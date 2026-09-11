"""
src/rag_conversational/prompt_builder.py
=========================================
Xây dựng Prompt có cấu trúc cho LLM (Gemini API).

Prompt gồm 5 phần:
  [SYSTEM]         : Vai trò và hành vi của AI
  [CONTEXT]        : Các đoạn hội thoại liên quan (từ RAG)
  [RECOMMENDATIONS]: Top-N items từ GNN+GCL
  [USER_HISTORY]   : Tóm tắt lịch sử ngắn gọn
  [QUERY]          : Câu hỏi hiện tại + nhiệm vụ LLM

Hỗ trợ 2 ngôn ngữ: Tiếng Việt (mặc định) và English.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT CONFIGURATION
# ============================================================

@dataclass
class PromptConfig:
    """
    Cấu hình cho Prompt Builder.

    Attributes:
        language        (str) : 'vi' (tiếng Việt) hoặc 'en'.
        max_context_len (int) : Độ dài tối đa của mỗi đoạn context (ký tự).
        max_response_len(int) : Độ dài tối đa của response cũ trong context.
        top_n_display   (int) : Số items hiển thị trong prompt.
        include_history (bool): Có thêm lịch sử ngắn gọn vào prompt không.
        style           (str) : 'formal' | 'friendly' | 'concise'.
    """
    language         : str  = 'vi'
    max_context_len  : int  = 300
    max_response_len : int  = 200
    top_n_display    : int  = 8
    include_history  : bool = True
    style            : str  = 'friendly'


# ============================================================
# SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPTS_VI = {
    'friendly': (
        "Bạn là trợ lý tư vấn thông minh, thân thiện và am hiểu về phim ảnh, âm nhạc "
        "và sản phẩm tiêu dùng. Nhiệm vụ của bạn là:\n"
        "1. Trả lời câu hỏi của người dùng một cách tự nhiên, thân thiện.\n"
        "2. Gợi ý 3-5 sản phẩm/nội dung phù hợp từ danh sách được cung cấp.\n"
        "3. Giải thích ngắn gọn tại sao mỗi gợi ý phù hợp với người dùng.\n"
        "4. Hỏi thêm nếu cần thông tin để cải thiện gợi ý.\n"
        "Luôn dùng danh sách gợi ý đã cho — KHÔNG tự bịa thêm sản phẩm không có trong danh sách."
    ),
    'formal': (
        "Bạn là hệ thống tư vấn chuyên nghiệp. Căn cứ trên ngữ cảnh hội thoại và "
        "danh sách gợi ý cá nhân hóa, hãy:\n"
        "1. Phân tích nhu cầu người dùng.\n"
        "2. Đề xuất các mục phù hợp từ danh sách cung cấp.\n"
        "3. Cung cấp lý do rõ ràng cho từng đề xuất.\n"
        "Chỉ sử dụng thông tin trong danh sách gợi ý — không bổ sung thông tin ngoài."
    ),
    'concise': (
        "Trợ lý tư vấn. Gợi ý 3-5 mục từ danh sách cho sẵn, giải thích ngắn gọn. "
        "Không bịa thêm sản phẩm."
    ),
}

SYSTEM_PROMPTS_EN = {
    'friendly': (
        "You are a friendly and knowledgeable recommendation assistant. Your task:\n"
        "1. Answer the user's question naturally and helpfully.\n"
        "2. Suggest 3-5 items from the provided list.\n"
        "3. Briefly explain why each suggestion fits the user.\n"
        "4. Ask follow-up questions if needed.\n"
        "Only recommend items from the provided list — DO NOT invent items not in the list."
    ),
    'formal': (
        "You are a professional recommendation system. Based on the conversation context "
        "and personalized item list:\n"
        "1. Analyze the user's needs.\n"
        "2. Recommend suitable items from the provided list.\n"
        "3. Provide clear reasoning for each recommendation.\n"
        "Only use items from the provided list."
    ),
    'concise': (
        "Recommendation assistant. Suggest 3-5 items from the list with brief explanations. "
        "Do not invent items."
    ),
}


# ============================================================
# PROMPT BUILDER
# ============================================================

class PromptBuilder:
    """
    Xây dựng Prompt có cấu trúc cho LLM.

    Sử dụng:
        builder = PromptBuilder(config=PromptConfig(language='vi'))
        prompt  = builder.build(
            query       = "Tôi muốn xem phim hành động hay",
            context     = retrieved_turns,      # Từ RAGRetriever
            recommendations = top_n_items,      # Từ GNNGCLAdapter
            user_history = recent_history,      # Từ ConversationStore
        )
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()

    # ----------------------------------------------------------
    # BUILD SECTIONS
    # ----------------------------------------------------------

    def _build_system_section(self) -> str:
        """Xây dựng phần [SYSTEM]."""
        lang  = self.config.language
        style = self.config.style
        if lang == 'vi':
            prompts = SYSTEM_PROMPTS_VI
        else:
            prompts = SYSTEM_PROMPTS_EN
        return prompts.get(style, prompts['friendly'])

    def _build_context_section(self, context: list[dict]) -> str:
        """
        Xây dựng phần [CONVERSATION CONTEXT] từ RAG results.

        Args:
            context (list[dict]): Kết quả từ RAGRetriever.retrieve()
        """
        if not context:
            if self.config.language == 'vi':
                return "[Không có ngữ cảnh hội thoại liên quan]"
            return "[No relevant conversation context]"

        if self.config.language == 'vi':
            header = "📚 NGỮ CẢNH HỘI THOẠI LIÊN QUAN:"
            q_label = "Người dùng"
            a_label = "Trợ lý"
        else:
            header = "📚 RELEVANT CONVERSATION CONTEXT:"
            q_label = "User"
            a_label = "Assistant"

        lines = [header]
        max_ctx = self.config.max_context_len
        max_resp = self.config.max_response_len

        for i, turn in enumerate(context, 1):
            score_pct = int(turn.get('score_semantic', turn.get('score', 0)) * 100)
            query_text = str(turn.get('query', ''))[:max_ctx]
            resp_text  = str(turn.get('response', ''))[:max_resp]

            lines.append(f"\n  [{i}] (Liên quan: {score_pct}%)")
            lines.append(f"  {q_label}: {query_text}")
            lines.append(f"  {a_label}: {resp_text}{'...' if len(turn.get('response','')) > max_resp else ''}")

        return '\n'.join(lines)

    def _build_recommendations_section(self, recommendations: list[dict]) -> str:
        """
        Xây dựng phần [PERSONALIZED RECOMMENDATIONS] từ GNN+GCL.

        Args:
            recommendations (list[dict]): Kết quả từ GNNGCLAdapter.get_top_n_items()
        """
        if not recommendations:
            if self.config.language == 'vi':
                return "[Không có gợi ý cá nhân hóa]"
            return "[No personalized recommendations]"

        if self.config.language == 'vi':
            header = "🎯 DANH SÁCH GỢI Ý CÁ NHÂN HÓA (từ GNN+GCL):"
        else:
            header = "🎯 PERSONALIZED RECOMMENDATIONS (from GNN+GCL):"

        lines = [header]
        top_n = recommendations[:self.config.top_n_display]

        for i, item in enumerate(top_n, 1):
            prompt_str = item.get('prompt_str', '')
            if not prompt_str:
                # Fallback: build manually
                title  = item.get('title', f"Item {item.get('item_id', i)}")
                score  = item.get('score', 0.0)
                genres = ', '.join(item.get('genres', [])[:2]) or 'N/A'
                tags   = ', '.join(item.get('tags', [])[:2])
                year   = item.get('year', '')
                year_s = f" ({year})" if year else ''
                tags_s = f" | {tags}" if tags else ''
                prompt_str = f"• {title}{year_s} [Score: {score:.3f}] [Genre: {genres}]{tags_s}"
            lines.append(f"  {i}. {prompt_str}")

        return '\n'.join(lines)

    def _build_history_section(self, user_history: list[dict]) -> str:
        """
        Xây dựng phần [USER HISTORY] — tóm tắt ngắn gọn.

        Args:
            user_history (list[dict]): Từ ConversationStore.get_user_history()
        """
        if not user_history or not self.config.include_history:
            return ""

        if self.config.language == 'vi':
            header = "📋 LỊCH SỬ GẦN ĐÂY CỦA NGƯỜI DÙNG:"
        else:
            header = "📋 USER'S RECENT HISTORY:"

        lines = [header]
        recent = user_history[:3]   # Chỉ lấy 3 lượt gần nhất để tránh prompt quá dài
        for turn in recent:
            q = str(turn.get('query', ''))[:100]
            lines.append(f"  • {q}")

        return '\n'.join(lines)

    def _build_query_section(self, query: str) -> str:
        """Xây dựng phần [CURRENT QUERY]."""
        if self.config.language == 'vi':
            return (
                f"💬 CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:\n"
                f"  \"{query}\"\n\n"
                f"📝 NHIỆM VỤ CỦA BẠN:\n"
                f"  1. Trả lời câu hỏi trên dựa trên ngữ cảnh và danh sách gợi ý.\n"
                f"  2. Đề xuất 3-5 mục phù hợp nhất từ DANH SÁCH GỢI Ý ở trên, kèm lý do ngắn gọn.\n"
                f"  3. Nếu cần thêm thông tin, hãy hỏi người dùng.\n"
                f"  ⚠️  Chỉ được gợi ý từ danh sách đã cho — KHÔNG được tự thêm mục khác."
            )
        return (
            f"💬 CURRENT USER QUERY:\n"
            f"  \"{query}\"\n\n"
            f"📝 YOUR TASK:\n"
            f"  1. Answer the question based on context and recommendations.\n"
            f"  2. Suggest 3-5 most relevant items from the RECOMMENDATIONS list above, with brief explanations.\n"
            f"  3. Ask follow-up if needed.\n"
            f"  ⚠️  Only suggest from the provided list — do NOT invent items."
        )

    # ----------------------------------------------------------
    # BUILD FULL PROMPT
    # ----------------------------------------------------------

    def build(
        self,
        query: str,
        context: Optional[list[dict]] = None,
        recommendations: Optional[list[dict]] = None,
        user_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Xây dựng prompt hoàn chỉnh từ 4 thành phần.

        Args:
            query           (str)            : Câu hỏi hiện tại của người dùng.
            context         (list[dict], opt): RAG results (từ RAGRetriever).
            recommendations (list[dict], opt): GNN+GCL results (từ GNNGCLAdapter).
            user_history    (list[dict], opt): Lịch sử hội thoại (từ ConversationStore).

        Returns:
            str: Prompt hoàn chỉnh, sẵn sàng gửi tới LLM.
        """
        sep = "\n" + "─" * 50 + "\n"

        sections = []

        # 1. System
        system_text = self._build_system_section()
        sections.append(system_text)
        sections.append(sep)

        # 2. User history (nếu có)
        history_text = self._build_history_section(user_history or [])
        if history_text:
            sections.append(history_text)
            sections.append(sep)

        # 3. RAG Context
        context_text = self._build_context_section(context or [])
        sections.append(context_text)
        sections.append(sep)

        # 4. GNN+GCL Recommendations
        rec_text = self._build_recommendations_section(recommendations or [])
        sections.append(rec_text)
        sections.append(sep)

        # 5. Query + Task
        query_text = self._build_query_section(query)
        sections.append(query_text)

        prompt = '\n'.join(sections)

        logger.debug(f"Prompt built: {len(prompt)} chars")
        return prompt

    def build_simple(
        self,
        query: str,
        recommendations: Optional[list[dict]] = None,
    ) -> str:
        """
        Phiên bản đơn giản của build() — không cần RAG context.
        Dùng khi user mới (cold start) hoặc test nhanh.
        """
        return self.build(query=query, recommendations=recommendations)

    def estimate_tokens(self, prompt: str) -> int:
        """
        Ước tính số token trong prompt (1 token ≈ 4 ký tự tiếng Anh / 2 ký tự tiếng Việt).
        """
        return len(prompt) // 3   # ước lượng trung bình

    def __repr__(self):
        return (
            f"PromptBuilder("
            f"lang='{self.config.language}', "
            f"style='{self.config.style}', "
            f"top_n={self.config.top_n_display})"
        )
