"""
src/rag_conversational/conversational_recommender.py
====================================================
ConversationalRecommender: Pipeline chính tích hợp GNN+GCL + RAG + LLM.

Pipeline 3-Phase:
  Phase 1 (Parallel): RAG Retrieval + GNN+GCL Recommendation
  Phase 2 (Fusion)  : Xây dựng Prompt có cấu trúc
  Phase 3 (Generate): LLM sinh phản hồi + Feedback loop

Feedback Loop:
  - Khi user accept gợi ý → cập nhật interaction graph
  - Khi phiên hội thoại kết thúc → lưu vào ConversationStore
  - Định kỳ re-train model với dữ liệu mới (async)
"""

import time
import logging
import asyncio
from typing import Optional
from dataclasses import dataclass, field

from .conversation_store import ConversationStore, ConversationTurn
from .rag_retriever import RAGRetriever
from .gnn_gcl_adapter import GNNGCLAdapter
from .prompt_builder import PromptBuilder, PromptConfig
from .response_generator import ResponseGenerator, LLMResponse

logger = logging.getLogger(__name__)


# ============================================================
# SESSION & TURN DATA
# ============================================================

@dataclass
class TurnResult:
    """
    Kết quả của một lượt hội thoại hoàn chỉnh.

    Attributes:
        query           (str)        : Câu hỏi người dùng.
        response        (LLMResponse): Phản hồi từ LLM.
        retrieved_context(list)      : Các đoạn hội thoại RAG đã truy xuất.
        gnn_recommendations(list)    : Top-N items từ GNN+GCL.
        prompt          (str)        : Prompt đã gửi cho LLM.
        total_latency_ms(float)      : Tổng thời gian xử lý (ms).
        turn_id         (str)        : ID của lượt hội thoại đã lưu.
    """
    query               : str
    response            : LLMResponse
    retrieved_context   : list = field(default_factory=list)
    gnn_recommendations : list = field(default_factory=list)
    prompt              : str  = ''
    total_latency_ms    : float = 0.0
    turn_id             : str  = ''

    def to_display(self) -> str:
        """Format kết quả để hiển thị cho user."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"💬 BẠN: {self.query}")
        lines.append(f"{'─'*60}")
        lines.append(f"🤖 TRỢ LÝ: {self.response.text}")
        lines.append(f"{'─'*60}")

        if self.retrieved_context:
            lines.append(f"📚 Ngữ cảnh sử dụng: {len(self.retrieved_context)} đoạn hội thoại")

        lines.append(f"⏱  Thời gian: {self.total_latency_ms:.0f}ms")
        if not self.response.is_mock:
            lines.append(f"🔢 Token: {self.response.tokens_used}")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)


@dataclass
class Session:
    """
    Phiên hội thoại của một user.

    Attributes:
        user_id         (int)        : ID người dùng.
        session_id      (str)        : UUID phiên.
        turns           (list)       : Danh sách TurnResult trong phiên.
        accepted_items  (list[int])  : Items user đã chấp nhận (feedback).
        start_time      (float)      : Timestamp bắt đầu phiên.
    """
    user_id        : int
    session_id     : str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    turns          : list = field(default_factory=list)
    accepted_items : list = field(default_factory=list)
    start_time     : float = field(default_factory=time.time)

    def add_turn(self, result: TurnResult):
        self.turns.append(result)

    def accept_item(self, item_id: int):
        """User chấp nhận một gợi ý → feedback cho re-train."""
        if item_id not in self.accepted_items:
            self.accepted_items.append(item_id)

    @property
    def turn_count(self):
        return len(self.turns)


# ============================================================
# MAIN PIPELINE CLASS
# ============================================================

class ConversationalRecommender:
    """
    Pipeline tích hợp GNN+GCL + RAG + LLM cho Hội Thoại Khuyến Nghị.

    Kiến trúc:
        query (user_id, q_t)
            │
            ├── [Phase 1A] RAGRetriever.retrieve(q_t, user_id)
            │       → Top-K conversation segments
            │
            ├── [Phase 1B] GNNGCLAdapter.get_top_n_items(user_id)
            │       → Top-N personalized items
            │
            ↓
        [Phase 2] PromptBuilder.build(q_t, context, items, history)
            → Structured prompt
            │
            ↓
        [Phase 3] ResponseGenerator.generate(prompt)
            → LLMResponse
            │
            ↓
        ConversationStore.add_turn(turn)   (lưu vào CSDL)
        GNNGCLAdapter.update_interaction() (nếu có feedback)

    Args:
        store           (ConversationStore): CSDL hội thoại.
        gnn_adapter     (GNNGCLAdapter)    : Wrapper GNN+GCL model.
        prompt_builder  (PromptBuilder)    : Prompt constructor.
        response_generator (ResponseGenerator): LLM caller.
        top_n           (int)              : Số items từ GNN+GCL. Mặc định 10.
        top_k_rag       (int)              : Số đoạn từ RAG. Mặc định 5.
        save_turns      (bool)             : Tự động lưu mỗi lượt vào store.
        personalized_rag(bool)             : Chỉ dùng hội thoại của user đó.
    """

    def __init__(
        self,
        store: ConversationStore,
        gnn_adapter: GNNGCLAdapter,
        prompt_builder: Optional[PromptBuilder] = None,
        response_generator: Optional[ResponseGenerator] = None,
        top_n: int = 10,
        top_k_rag: int = 5,
        save_turns: bool = True,
        personalized_rag: bool = True,
    ):
        self.store       = store
        self.gnn_adapter = gnn_adapter
        self.top_n       = top_n
        self.top_k_rag   = top_k_rag
        self.save_turns  = save_turns
        self.personalized_rag = personalized_rag

        # Khởi tạo sub-modules với default nếu không truyền
        self.retriever = RAGRetriever(store, k=top_k_rag)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_generator = response_generator or ResponseGenerator()

        # Session management
        self._sessions: dict[int, Session] = {}

        logger.info(
            f"ConversationalRecommender initialized:\n"
            f"  GNN+GCL: {gnn_adapter}\n"
            f"  Store: {store}\n"
            f"  Generator: {self.response_generator}\n"
            f"  top_n={top_n}, top_k_rag={top_k_rag}"
        )

    # ----------------------------------------------------------
    # SESSION MANAGEMENT
    # ----------------------------------------------------------

    def start_session(self, user_id: int) -> Session:
        """Bắt đầu phiên hội thoại mới cho user."""
        session = Session(user_id=user_id)
        self._sessions[user_id] = session
        logger.info(f"Session started: user_id={user_id}, session_id={session.session_id}")
        return session

    def get_session(self, user_id: int) -> Optional[Session]:
        """Lấy phiên hội thoại hiện tại của user."""
        return self._sessions.get(user_id)

    def end_session(self, user_id: int) -> Optional[Session]:
        """
        Kết thúc phiên hội thoại và lưu thống kê.
        Trả về Session đã kết thúc.
        """
        session = self._sessions.pop(user_id, None)
        if session:
            duration = time.time() - session.start_time
            logger.info(
                f"Session ended: user_id={user_id}, "
                f"turns={session.turn_count}, "
                f"duration={duration:.1f}s, "
                f"accepted_items={session.accepted_items}"
            )
        return session

    # ----------------------------------------------------------
    # MAIN CHAT METHOD
    # ----------------------------------------------------------

    def chat(
        self,
        user_id: int,
        query: str,
        exclude_conv_items: bool = True,
    ) -> TurnResult:
        """
        Xử lý một lượt hội thoại và trả về kết quả.

        Đây là phương thức chính của pipeline 3-phase.

        Args:
            user_id             (int) : ID người dùng.
            query               (str) : Câu hỏi của người dùng.
            exclude_conv_items  (bool): Loại trừ items đã gợi ý trong phiên.

        Returns:
            TurnResult: Kết quả đầy đủ của lượt hội thoại.
        """
        t_start = time.time()
        logger.info(f"Chat: user_id={user_id}, query='{query[:60]}'")

        # Lấy phiên hiện tại (tạo mới nếu chưa có)
        session = self._sessions.get(user_id)
        if session is None:
            session = self.start_session(user_id)

        # -------------------------------------------------------
        # PHASE 1: DUAL-BRANCH PARALLEL EXECUTION (1A: RAG + 1B: GNN+GCL)
        # -------------------------------------------------------
        # Items đã gợi ý trong phiên hiện tại
        conv_items = set()
        if exclude_conv_items:
            for prev_turn in session.turns:
                conv_items.update(
                    item.get('item_id', 0)
                    for item in prev_turn.gnn_recommendations
                )

        user_id_for_rag = user_id if self.personalized_rag else None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_rag = executor.submit(
                self.retriever.retrieve,
                query=query,
                user_id=user_id_for_rag,
                k=self.top_k_rag,
            )
            future_gnn = executor.submit(
                self.gnn_adapter.get_top_n_items,
                user_id=user_id,
                n=self.top_n,
                exclude_seen=True,
                exclude_conv_items=conv_items if conv_items else None,
            )
            retrieved_context = future_rag.result()
            recommendations = future_gnn.result()

        t_parallel = time.time()
        logger.debug(
            f"Dual-branch parallel: RAG={len(retrieved_context)} turns, "
            f"GNN={len(recommendations)} items in {(t_parallel - t_start)*1000:.0f}ms"
        )

        # -------------------------------------------------------
        # PHASE 2: CONTEXT FUSION — Xây dựng Prompt
        # -------------------------------------------------------
        user_history = self.store.get_user_history(user_id, limit=10)

        prompt = self.prompt_builder.build(
            query           = query,
            context         = retrieved_context,
            recommendations = recommendations,
            user_history    = user_history,
        )
        t_prompt = time.time()
        logger.debug(f"Prompt: {len(prompt)} chars in {(t_prompt-t_parallel)*1000:.0f}ms")

        # -------------------------------------------------------
        # PHASE 3: RESPONSE GENERATION
        # -------------------------------------------------------
        llm_response = self.response_generator.generate(prompt)
        t_llm = time.time()
        logger.debug(f"LLM: {len(llm_response.text)} chars in {(t_llm-t_prompt)*1000:.0f}ms")

        total_ms = (t_llm - t_start) * 1000

        # -------------------------------------------------------
        # LƯU VÀO STORE + CẬP NHẬT SESSION
        # -------------------------------------------------------
        turn_id = ''
        if self.save_turns:
            items_shown = [r.get('item_id', 0) for r in recommendations[:10]]
            turn = ConversationTurn(
                user_id     = user_id,
                turn_index  = session.turn_count,
                query       = query,
                response    = llm_response.text,
                items_shown = items_shown,
                dataset_src = 'live',
            )
            turn_id = self.store.add_turn(turn)
            logger.debug(f"Turn saved: {turn_id}")

        # Tạo TurnResult
        result = TurnResult(
            query               = query,
            response            = llm_response,
            retrieved_context   = retrieved_context,
            gnn_recommendations = recommendations,
            prompt              = prompt,
            total_latency_ms    = total_ms,
            turn_id             = turn_id,
        )
        session.add_turn(result)

        logger.info(
            f"Turn complete: user_id={user_id}, "
            f"rag={len(retrieved_context)}, recs={len(recommendations)}, "
            f"total={total_ms:.0f}ms"
        )
        return result

    # ----------------------------------------------------------
    # FEEDBACK LOOP
    # ----------------------------------------------------------

    def accept_item(self, user_id: int, item_id: int):
        """
        Ghi nhận user chấp nhận một gợi ý (positive feedback).

        Tác động:
          1. Cập nhật interaction trong GNNGCLAdapter
          2. Ghi vào session hiện tại
          3. (Future) Trigger re-train nếu đủ new interactions

        Args:
            user_id (int): ID người dùng.
            item_id (int): Item ID được chấp nhận.
        """
        # Cập nhật GNN adapter
        self.gnn_adapter.update_interaction(user_id, item_id)
        self.gnn_adapter.invalidate_cache()

        # Cập nhật session
        session = self._sessions.get(user_id)
        if session:
            session.accept_item(item_id)

        logger.info(f"Feedback: user {user_id} accepted item {item_id}")

    def reject_item(self, user_id: int, item_id: int):
        """
        Ghi nhận user từ chối một gợi ý (negative feedback).
        (Không loại trừ item vĩnh viễn nhưng ghi nhận để phân tích.)
        """
        logger.info(f"Feedback: user {user_id} rejected item {item_id}")
        # TODO: Cập nhật negative feedback vào model

    # ----------------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------------

    def get_session_stats(self, user_id: int) -> dict:
        """Thống kê phiên hội thoại hiện tại."""
        session = self._sessions.get(user_id)
        if not session:
            return {'status': 'no_session', 'user_id': user_id}

        total_rag  = sum(len(t.retrieved_context) for t in session.turns)
        total_recs = sum(len(t.gnn_recommendations) for t in session.turns)
        avg_latency = (
            sum(t.total_latency_ms for t in session.turns) / session.turn_count
            if session.turn_count > 0 else 0
        )

        return {
            'user_id'        : user_id,
            'session_id'     : session.session_id,
            'turn_count'     : session.turn_count,
            'total_rag_results': total_rag,
            'total_recs'     : total_recs,
            'avg_latency_ms' : avg_latency,
            'accepted_items' : session.accepted_items,
            'duration_s'     : time.time() - session.start_time,
        }

    def get_system_stats(self) -> dict:
        """Thống kê toàn hệ thống."""
        return {
            'store_count'       : self.store.count(),
            'active_sessions'   : len(self._sessions),
            'gnn_adapter'       : str(self.gnn_adapter),
            'rag_retriever'     : str(self.retriever),
            'response_generator': str(self.response_generator),
        }

    def __repr__(self):
        return (
            f"ConversationalRecommender("
            f"top_n={self.top_n}, top_k_rag={self.top_k_rag}, "
            f"sessions={len(self._sessions)})"
        )
