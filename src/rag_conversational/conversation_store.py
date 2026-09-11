"""
src/rag_conversational/conversation_store.py
============================================
CSDL Hội Thoại dựa trên ChromaDB (Vector Database).

Mỗi đoạn hội thoại được lưu dưới dạng:
  - embedding  : vector 384-dim (sentence-transformers MiniLM)
  - metadata   : user_id, turn_index, query, response, items_shown, timestamp
  - document   : text dùng để truy xuất (query + response tổng hợp)

Chức năng chính:
  - add_turn()      : Thêm một lượt hội thoại vào CSDL
  - search()        : Tìm K đoạn hội thoại tương đồng nhất với truy vấn
  - get_user_history(): Lấy toàn bộ lịch sử của một user
  - delete_user()   : Xóa lịch sử một user
"""

import os
import time
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logging.warning("chromadb chưa được cài. Chạy: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logging.warning("sentence-transformers chưa được cài. Chạy: pip install sentence-transformers")

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ConversationTurn:
    """
    Một lượt hội thoại (một cặp Query-Response).

    Attributes:
        user_id     (int)       : ID người dùng.
        turn_index  (int)       : Thứ tự lượt trong phiên hội thoại (bắt đầu từ 0).
        query       (str)       : Câu hỏi của người dùng.
        response    (str)       : Phản hồi của hệ thống.
        items_shown (list[int]) : Danh sách item IDs đã được gợi ý.
        timestamp   (float)     : Unix timestamp khi tạo.
        turn_id     (str)       : UUID duy nhất (tự sinh nếu không cung cấp).
        dataset_src (str)       : Nguồn dữ liệu ('movielens', 'lastfm', 'amazon', 'live').
    """
    user_id     : int
    query       : str
    response    : str
    turn_index  : int = 0
    items_shown : list = field(default_factory=list)
    timestamp   : float = field(default_factory=time.time)
    turn_id     : str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_src : str = 'live'

    def to_document_text(self) -> str:
        """Tạo text đại diện cho vector embedding (query + response)."""
        items_str = ', '.join(str(i) for i in self.items_shown[:5])
        return (
            f"Người dùng hỏi: {self.query} "
            f"Hệ thống trả lời: {self.response} "
            f"Gợi ý items: {items_str}"
        )

    def to_metadata(self) -> dict:
        """Chuyển thành dict metadata cho ChromaDB."""
        return {
            'user_id'    : self.user_id,
            'turn_index' : self.turn_index,
            'query'      : self.query[:500],        # ChromaDB giới hạn metadata string
            'response'   : self.response[:500],
            'items_shown': json.dumps(self.items_shown[:20]),
            'timestamp'  : self.timestamp,
            'dataset_src': self.dataset_src,
        }


# ============================================================
# FALLBACK STORE (khi không có ChromaDB)
# ============================================================

class InMemoryConversationStore:
    """
    Fallback store dùng numpy cosine similarity khi ChromaDB không có.
    Chỉ dùng cho development/testing.
    """

    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        self._turns   = []          # list[ConversationTurn]
        self._embeddings = []       # list[np.ndarray]
        self._encoder = None
        self._model_name = encoder_model

        if HAS_ST:
            logger.info(f"Loading encoder: {encoder_model}")
            self._encoder = SentenceTransformer(encoder_model)

    def _encode(self, text: str):
        if self._encoder:
            return self._encoder.encode(text, normalize_embeddings=True)
        return None

    def add_turn(self, turn: ConversationTurn) -> str:
        doc_text = turn.to_document_text()
        emb = self._encode(doc_text)
        self._turns.append(turn)
        self._embeddings.append(emb)
        return turn.turn_id

    def search(self, query: str, user_id: Optional[int] = None,
               k: int = 5) -> list[dict]:
        if not self._encoder or not self._embeddings:
            return []

        import numpy as np
        q_emb = self._encode(query)

        results = []
        for i, (turn, emb) in enumerate(zip(self._turns, self._embeddings)):
            if user_id is not None and turn.user_id != user_id:
                continue
            if emb is None:
                continue
            score = float(np.dot(q_emb, emb))
            results.append({'turn': turn, 'score': score, 'rank': 0})

        results.sort(key=lambda x: -x['score'])
        for rank, r in enumerate(results[:k]):
            r['rank'] = rank + 1
        return results[:k]

    def get_user_history(self, user_id: int) -> list[ConversationTurn]:
        return [t for t in self._turns if t.user_id == user_id]

    def count(self) -> int:
        return len(self._turns)

    def clear(self):
        self._turns.clear()
        self._embeddings.clear()


# ============================================================
# CHROMADB CONVERSATION STORE (Production)
# ============================================================

class ConversationStore:
    """
    CSDL Hội Thoại Vector dựa trên ChromaDB.

    Mỗi collection tương ứng với một dataset/domain.
    Tự động fallback sang InMemoryConversationStore nếu ChromaDB không có.

    Args:
        persist_dir     (str) : Thư mục lưu ChromaDB. Mặc định './chroma_db'.
        collection_name (str) : Tên collection. Mặc định 'conversations'.
        encoder_model   (str) : Tên model sentence-transformers.
                               Mặc định 'all-MiniLM-L6-v2' (384-dim, nhẹ, nhanh).
    """

    ENCODER_MODEL = 'all-MiniLM-L6-v2'

    def __init__(
        self,
        persist_dir: str = './chroma_db',
        collection_name: str = 'conversations',
        encoder_model: str = ENCODER_MODEL,
    ):
        self.persist_dir      = persist_dir
        self.collection_name  = collection_name
        self._encoder_model   = encoder_model
        self._encoder         = None
        self._client          = None
        self._collection      = None

        # Khởi tạo encoder
        if HAS_ST:
            logger.info(f"Loading sentence-transformer: {encoder_model}")
            self._encoder = SentenceTransformer(encoder_model)
        else:
            logger.warning("sentence-transformers không khả dụng — embedding bị vô hiệu.")

        # Khởi tạo ChromaDB hoặc fallback
        if HAS_CHROMA:
            self._init_chromadb()
        else:
            logger.warning("ChromaDB không khả dụng — dùng InMemory fallback.")
            self._fallback = InMemoryConversationStore(encoder_model)

    def _init_chromadb(self):
        """Khởi tạo ChromaDB client và collection."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={'hnsw:space': 'cosine'},   # Dùng cosine similarity
        )
        logger.info(
            f"ChromaDB initialized: {self.persist_dir}/{self.collection_name} "
            f"({self._collection.count()} records)"
        )

    def _encode(self, text: str) -> list[float]:
        """Encode text thành vector float list."""
        if self._encoder is None:
            raise RuntimeError("Encoder chưa được khởi tạo. Cài sentence-transformers.")
        emb = self._encoder.encode(text, normalize_embeddings=True)
        return emb.tolist()

    # ----------------------------------------------------------
    # THÊM MỘT LƯỢT HỘI THOẠI
    # ----------------------------------------------------------

    def add_turn(self, turn: ConversationTurn) -> str:
        """
        Thêm một ConversationTurn vào CSDL.

        Args:
            turn (ConversationTurn): Lượt hội thoại cần lưu.

        Returns:
            str: turn_id của lượt vừa thêm.
        """
        if not HAS_CHROMA:
            return self._fallback.add_turn(turn)

        doc_text = turn.to_document_text()
        embedding = self._encode(doc_text)
        metadata  = turn.to_metadata()

        self._collection.add(
            ids        = [turn.turn_id],
            documents  = [doc_text],
            embeddings = [embedding],
            metadatas  = [metadata],
        )
        logger.debug(f"Added turn {turn.turn_id} for user {turn.user_id}")
        return turn.turn_id

    def add_turns_batch(self, turns: list[ConversationTurn]) -> list[str]:
        """
        Thêm nhiều lượt hội thoại cùng lúc (hiệu quả hơn cho synthetic data).

        Args:
            turns (list[ConversationTurn]): Danh sách lượt hội thoại.

        Returns:
            list[str]: Danh sách turn_ids.
        """
        if not turns:
            return []

        if not HAS_CHROMA:
            return [self._fallback.add_turn(t) for t in turns]

        docs       = [t.to_document_text() for t in turns]
        ids        = [t.turn_id for t in turns]
        metadatas  = [t.to_metadata() for t in turns]
        embeddings = [self._encode(d) for d in docs]

        # ChromaDB có giới hạn batch size = 5000
        BATCH = 500
        for i in range(0, len(turns), BATCH):
            self._collection.add(
                ids        = ids[i:i+BATCH],
                documents  = docs[i:i+BATCH],
                embeddings = embeddings[i:i+BATCH],
                metadatas  = metadatas[i:i+BATCH],
            )
            logger.info(f"Added batch {i//BATCH + 1}: {min(i+BATCH, len(turns))} turns total")

        return ids

    # ----------------------------------------------------------
    # TÌM KIẾM NGỮ NGHĨA
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: Optional[int] = None,
        k: int = 5,
        dataset_src: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Tìm K đoạn hội thoại liên quan nhất với query.

        Hỗ trợ lọc theo:
          - user_id    : Chỉ tìm trong hội thoại của user đó (Personalized RAG)
          - dataset_src: Lọc theo nguồn ('movielens', 'lastfm', 'amazon', 'live')
          - min_score  : Ngưỡng similarity tối thiểu

        Args:
            query      (str)          : Câu hỏi cần tìm context.
            user_id    (int, optional): Lọc theo user.
            k          (int)          : Số kết quả tối đa. Mặc định 5.
            dataset_src (str, optional): Lọc theo nguồn dữ liệu.
            min_score  (float)        : Ngưỡng score tối thiểu [0, 1].

        Returns:
            list[dict]: Danh sách kết quả, mỗi phần tử gồm:
                {
                    'turn_id'    : str,
                    'query'      : str,
                    'response'   : str,
                    'items_shown': list[int],
                    'user_id'    : int,
                    'score'      : float,   # cosine similarity [0, 1]
                    'rank'       : int,     # 1-indexed
                }
        """
        if not HAS_CHROMA:
            return self._fallback.search(query, user_id, k)

        if self._collection.count() == 0:
            logger.debug("Collection trống — không có kết quả.")
            return []

        # Xây dựng filter (where clause)
        where = {}
        if user_id is not None and dataset_src is not None:
            where = {'$and': [
                {'user_id'    : {'$eq': user_id}},
                {'dataset_src': {'$eq': dataset_src}},
            ]}
        elif user_id is not None:
            where = {'user_id': {'$eq': user_id}}
        elif dataset_src is not None:
            where = {'dataset_src': {'$eq': dataset_src}}

        query_embedding = self._encode(query)

        # Thực hiện tìm kiếm
        kwargs = dict(
            query_embeddings = [query_embedding],
            n_results        = min(k * 2, self._collection.count()),  # lấy dư để lọc min_score
            include          = ['documents', 'metadatas', 'distances'],
        )
        if where:
            kwargs['where'] = where

        results_raw = self._collection.query(**kwargs)

        # Parse kết quả
        results = []
        ids_list       = results_raw.get('ids', [[]])[0]
        distances_list = results_raw.get('distances', [[]])[0]
        metadatas_list = results_raw.get('metadatas', [[]])[0]

        for turn_id, distance, meta in zip(ids_list, distances_list, metadatas_list):
            # ChromaDB cosine distance = 1 - cosine_similarity
            score = 1.0 - float(distance)
            if score < min_score:
                continue

            items_shown = json.loads(meta.get('items_shown', '[]'))
            results.append({
                'turn_id'    : turn_id,
                'query'      : meta.get('query', ''),
                'response'   : meta.get('response', ''),
                'items_shown': items_shown,
                'user_id'    : meta.get('user_id', -1),
                'timestamp'  : meta.get('timestamp', 0),
                'dataset_src': meta.get('dataset_src', ''),
                'score'      : score,
                'rank'       : 0,   # set below
            })

        # Sắp xếp theo score, lấy top k
        results.sort(key=lambda x: -x['score'])
        results = results[:k]
        for rank, r in enumerate(results):
            r['rank'] = rank + 1

        return results

    # ----------------------------------------------------------
    # TRUY VẤN THEO USER
    # ----------------------------------------------------------

    def get_user_history(self, user_id: int, limit: int = 50) -> list[dict]:
        """
        Lấy lịch sử hội thoại gần nhất của một user.

        Args:
            user_id (int) : ID người dùng.
            limit   (int) : Số lượt tối đa trả về. Mặc định 50.

        Returns:
            list[dict]: Danh sách lượt hội thoại, sắp xếp theo timestamp mới nhất.
        """
        if not HAS_CHROMA:
            turns = self._fallback.get_user_history(user_id)
            return [asdict(t) for t in turns[-limit:]]

        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            where   = {'user_id': {'$eq': user_id}},
            include = ['metadatas'],
            limit   = limit,
        )

        turns = []
        for meta_id, meta in zip(results.get('ids', []), results.get('metadatas', [])):
            items_shown = json.loads(meta.get('items_shown', '[]'))
            turns.append({
                'turn_id'    : meta_id,
                'user_id'    : meta.get('user_id', user_id),
                'turn_index' : meta.get('turn_index', 0),
                'query'      : meta.get('query', ''),
                'response'   : meta.get('response', ''),
                'items_shown': items_shown,
                'timestamp'  : meta.get('timestamp', 0),
                'dataset_src': meta.get('dataset_src', ''),
            })

        turns.sort(key=lambda x: -x['timestamp'])
        return turns

    def get_user_seen_items(self, user_id: int) -> set[int]:
        """Lấy tập hợp tất cả items đã được gợi ý cho user."""
        history = self.get_user_history(user_id, limit=200)
        seen = set()
        for turn in history:
            seen.update(turn.get('items_shown', []))
        return seen

    # ----------------------------------------------------------
    # QUẢN LÝ
    # ----------------------------------------------------------

    def count(self) -> int:
        """Số lượt hội thoại trong CSDL."""
        if not HAS_CHROMA:
            return self._fallback.count()
        return self._collection.count()

    def clear(self):
        """Xóa toàn bộ CSDL (dùng cẩn thận!)."""
        if not HAS_CHROMA:
            self._fallback.clear()
            return
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name     = self.collection_name,
            metadata = {'hnsw:space': 'cosine'},
        )
        logger.warning("Collection đã bị xóa và tạo lại.")

    def delete_user_history(self, user_id: int):
        """Xóa toàn bộ lịch sử hội thoại của một user."""
        if not HAS_CHROMA:
            return
        history = self.get_user_history(user_id, limit=10000)
        ids = [h['turn_id'] for h in history]
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} turns for user {user_id}")

    def __repr__(self):
        return (
            f"ConversationStore("
            f"collection='{self.collection_name}', "
            f"count={self.count()}, "
            f"backend={'chromadb' if HAS_CHROMA else 'in-memory'})"
        )
