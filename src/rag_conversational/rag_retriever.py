"""
src/rag_conversational/rag_retriever.py
=======================================
RAG Retriever: Tìm kiếm ngữ cảnh hội thoại liên quan.

Hỗ trợ 3 chiến lược truy xuất:
  1. Semantic Search: Cosine similarity trên embeddings (mặc định)
  2. Recency-weighted: Ưu tiên hội thoại gần đây
  3. Hybrid: Kết hợp Semantic + Recency + diversity

Giao tiếp với ConversationStore thông qua interface chuẩn.
"""

import time
import math
import logging
from typing import Optional

from .conversation_store import ConversationStore

logger = logging.getLogger(__name__)


# ============================================================
# RAG RETRIEVER
# ============================================================

class RAGRetriever:
    """
    Truy xuất các đoạn hội thoại liên quan nhất với truy vấn hiện tại.

    Chiến lược hybrid retrieval:
        score_final = α × score_semantic + (1-α) × score_recency

    Trong đó:
        score_semantic : Cosine similarity từ ChromaDB
        score_recency  : Hàm giảm theo thời gian (exponential decay)
        α              : Trọng số semantic (mặc định 0.8)

    Args:
        store          (ConversationStore): CSDL hội thoại.
        k              (int)             : Số kết quả tối đa. Mặc định 5.
        alpha          (float)           : Trọng số semantic [0, 1]. Mặc định 0.8.
        recency_decay  (float)           : Hệ số giảm theo ngày. Mặc định 0.1.
        min_score      (float)           : Ngưỡng score tối thiểu. Mặc định 0.2.
        diversity_mmr  (bool)            : Dùng MMR để tăng diversity. Mặc định True.
        mmr_lambda     (float)           : Trade-off relevance/diversity trong MMR.
    """

    def __init__(
        self,
        store: ConversationStore,
        k: int = 5,
        alpha: float = 0.8,
        recency_decay: float = 0.1,
        min_score: float = 0.2,
        diversity_mmr: bool = True,
        mmr_lambda: float = 0.6,
    ):
        self.store         = store
        self.k             = k
        self.alpha         = alpha
        self.recency_decay = recency_decay
        self.min_score     = min_score
        self.diversity_mmr = diversity_mmr
        self.mmr_lambda    = mmr_lambda

    # ----------------------------------------------------------
    # RECENCY SCORE
    # ----------------------------------------------------------

    def _recency_score(self, timestamp: float) -> float:
        """
        Tính điểm theo độ mới của hội thoại.

        Công thức: score = exp(-λ × days_ago)
        Trong đó:
            λ = self.recency_decay (mặc định 0.1)
            days_ago = (now - timestamp) / 86400

        Ví dụ:
            - 0 ngày trước → score ≈ 1.0
            - 7 ngày trước → score ≈ exp(-0.7) ≈ 0.497
            - 30 ngày trước → score ≈ exp(-3.0) ≈ 0.050
        """
        now = time.time()
        days_ago = max(0.0, (now - timestamp) / 86400.0)
        return math.exp(-self.recency_decay * days_ago)

    # ----------------------------------------------------------
    # HYBRID RETRIEVAL
    # ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: Optional[int] = None,
        k: Optional[int] = None,
        dataset_src: Optional[str] = None,
    ) -> list[dict]:
        """
        Truy xuất K đoạn hội thoại liên quan nhất.

        Quy trình:
          1. Semantic search từ ConversationStore (lấy 2k candidates)
          2. Tính final_score = α×semantic + (1-α)×recency
          3. (Optional) MMR reranking để tăng diversity
          4. Trả về top-k

        Args:
            query       (str)          : Câu hỏi cần tìm context.
            user_id     (int, optional): Chỉ tìm trong hội thoại của user này.
            k           (int, optional): Override số kết quả. Mặc định self.k.
            dataset_src (str, optional): Lọc theo dataset ('movielens', 'lastfm', etc.)

        Returns:
            list[dict]: Danh sách kết quả đã xếp hạng, mỗi phần tử:
                {
                    'turn_id'       : str,
                    'query'         : str,
                    'response'      : str,
                    'items_shown'   : list[int],
                    'user_id'       : int,
                    'score_semantic': float,
                    'score_recency' : float,
                    'score_final'   : float,
                    'rank'          : int,
                }
        """
        n = k or self.k

        if self.store.count() == 0:
            logger.debug("Store trống — không có kết quả retrieval.")
            return []

        # Bước 1: Semantic search (lấy 2n candidates để có đủ sau lọc)
        candidates = self.store.search(
            query       = query,
            user_id     = user_id,
            k           = n * 2,
            dataset_src = dataset_src,
            min_score   = self.min_score,
        )

        if not candidates:
            logger.debug(f"Không tìm được candidates cho query: '{query[:50]}'")
            return []

        # Bước 2: Tính hybrid score
        for c in candidates:
            sem_score     = c.get('score', 0.0)
            rec_score     = self._recency_score(c.get('timestamp', 0))
            c['score_semantic'] = sem_score
            c['score_recency']  = rec_score
            c['score_final']    = self.alpha * sem_score + (1 - self.alpha) * rec_score

        # Bước 3: MMR Reranking (tăng diversity)
        if self.diversity_mmr and len(candidates) > 1:
            selected = self._mmr_rerank(candidates, n)
        else:
            candidates.sort(key=lambda x: -x['score_final'])
            selected = candidates[:n]

        # Cập nhật rank
        for rank, r in enumerate(selected):
            r['rank'] = rank + 1

        logger.debug(
            f"Retrieved {len(selected)} turns for query '{query[:40]}' "
            f"(user_id={user_id})"
        )
        return selected

    def _mmr_rerank(self, candidates: list[dict], k: int) -> list[dict]:
        """
        Maximal Marginal Relevance (MMR) reranking.

        Công thức MMR:
            next = argmax_d [ λ×score_final(d) - (1-λ)×max_{s∈S} sim(d, s) ]

        Trong đó:
            S    = tập đã chọn
            d    = candidate đang xét
            λ    = self.mmr_lambda (trade-off relevance vs diversity)

        Dùng query similarity làm proxy cho sim(d, s) khi không có
        embedding trực tiếp (tính dựa trên chênh lệch score_semantic).
        """
        if not candidates:
            return []

        candidates_sorted = sorted(candidates, key=lambda x: -x['score_final'])
        selected = [candidates_sorted[0]]
        remaining = candidates_sorted[1:]

        while len(selected) < k and remaining:
            mmr_scores = []
            for c in remaining:
                relevance = c['score_final']
                # Diversity: penalty từ item gần nhất đã chọn
                max_sim = max(
                    1.0 - abs(c['score_semantic'] - s['score_semantic'])
                    for s in selected
                )
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                mmr_scores.append((mmr, c))

            mmr_scores.sort(key=lambda x: -x[0])
            best = mmr_scores[0][1]
            selected.append(best)
            remaining.remove(best)

        return selected

    # ----------------------------------------------------------
    # FORMAT CHO PROMPT
    # ----------------------------------------------------------

    def format_context_for_prompt(self, retrieved: list[dict]) -> str:
        """
        Format các đoạn hội thoại đã truy xuất thành block text cho LLM prompt.

        Args:
            retrieved (list[dict]): Kết quả từ retrieve().

        Returns:
            str: Text đã format, sẵn sàng nhúng vào prompt.
        """
        if not retrieved:
            return "[Không có ngữ cảnh hội thoại liên quan]"

        lines = ["=== NGỮ CẢNH HỘI THOẠI LIÊN QUAN ==="]
        for i, r in enumerate(retrieved, 1):
            score_pct = int(r.get('score_semantic', 0) * 100)
            lines.append(f"\n[Hội thoại {i} | Độ liên quan: {score_pct}%]")
            lines.append(f"  Người dùng: {r.get('query', '')}")
            lines.append(f"  Hệ thống  : {r.get('response', '')[:300]}")
            items = r.get('items_shown', [])
            if items:
                lines.append(f"  Items gợi ý: {', '.join(str(i) for i in items[:5])}")
        lines.append("=" * 40)
        return '\n'.join(lines)

    def retrieve_and_format(
        self,
        query: str,
        user_id: Optional[int] = None,
        k: Optional[int] = None,
        dataset_src: Optional[str] = None,
    ) -> tuple[list[dict], str]:
        """
        Shortcut: retrieve + format trong một lần gọi.

        Returns:
            (list[dict], str): (retrieved_turns, formatted_context_text)
        """
        retrieved = self.retrieve(query, user_id, k, dataset_src)
        context_text = self.format_context_for_prompt(retrieved)
        return retrieved, context_text

    def __repr__(self):
        return (
            f"RAGRetriever("
            f"k={self.k}, alpha={self.alpha}, "
            f"mmr={self.diversity_mmr}, "
            f"store_count={self.store.count()})"
        )
