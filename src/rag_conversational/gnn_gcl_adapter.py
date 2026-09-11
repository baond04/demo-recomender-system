"""
src/rag_conversational/gnn_gcl_adapter.py
=========================================
Wrapper/Adapter cho model GNN+GCL (SimGCL) hiện có.

Cung cấp interface chuẩn cho ConversationalRecommender:
  - get_top_n_items()   : Lấy Top-N items cho một user
  - get_user_embedding(): Lấy embedding vector của user
  - get_item_metadata() : Lấy thông tin item (title, genres, tags)

Hỗ trợ 2 mode:
  1. LIVE mode: Dùng model đã train (cần torch)
  2. DUMMY mode: Fallback khi model chưa train (test pipeline)
"""

import os
import sys
import csv
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

def _find_repo_root():
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / 'ml-latest-small').exists() or (current / 'hetrec2011-lastfm-2k').exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent

BASE_DIR = _find_repo_root()

# ============================================================
# ITEM METADATA LOADER
# ============================================================

class ItemMetadataStore:
    """
    Load và cung cấp metadata cho items từ MovieLens.
    Hỗ trợ: title, genres, tags, year.
    """

    def __init__(
        self,
        movies_csv: str = None,
        tags_csv:   str = None,
    ):
        if movies_csv is None:
            movies_csv = str(BASE_DIR / 'ml-latest-small' / 'movies.csv')
        if tags_csv is None:
            tags_csv = str(BASE_DIR / 'ml-latest-small' / 'tags.csv')

        self._movies = {}     # movie_id → {title, genres, year}
        self._tags   = defaultdict(list)   # movie_id → list[str]

        self._load_movies(movies_csv)
        self._load_tags(tags_csv)
        logger.info(f"ItemMetadataStore: {len(self._movies)} movies, {len(self._tags)} movies with tags")

    def _load_movies(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"movies.csv not found: {path}")
            return
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mid   = int(row['movieId'])
                title = row['title']
                # Trích năm từ title: "Toy Story (1995)" → 1995
                year = None
                if '(' in title and title.endswith(')'):
                    try:
                        year = int(title.split('(')[-1].strip(')'))
                    except ValueError:
                        pass
                genres = row['genres'].split('|') if row['genres'] != '(no genres listed)' else []
                self._movies[mid] = {
                    'title' : title,
                    'genres': genres,
                    'year'  : year,
                }

    def _load_tags(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mid = int(row['movieId'])
                self._tags[mid].append(row['tag'].lower())

    def get(self, item_id: int) -> dict:
        """Trả về metadata của một item."""
        if item_id in self._movies:
            info = self._movies[item_id].copy()
            info['tags'] = list(set(self._tags.get(item_id, [])))[:5]
            info['item_id'] = item_id
            return info
        return {
            'item_id': item_id,
            'title'  : f'Item {item_id}',
            'genres' : [],
            'tags'   : [],
            'year'   : None,
        }

    def format_for_prompt(self, item_id: int, score: float) -> str:
        """Format thông tin item thành string cho LLM prompt."""
        info = self.get(item_id)
        genres_str = ', '.join(info['genres'][:3]) if info['genres'] else 'N/A'
        tags_str   = ', '.join(info['tags'][:3]) if info['tags'] else ''
        year_str   = f"({info['year']})" if info['year'] else ''
        tags_part  = f" | Tags: {tags_str}" if tags_str else ''
        return (
            f"• {info['title']} {year_str} "
            f"[Score: {score:.3f}] "
            f"[Genre: {genres_str}]"
            f"{tags_part}"
        )

    def get_all_ids(self) -> list[int]:
        return list(self._movies.keys())


# ============================================================
# GNN+GCL ADAPTER
# ============================================================

class GNNGCLAdapter:
    """
    Adapter cho model SimGCL/LightGCN_GCL đã train.

    Args:
        model         : Instance SimGCL hoặc LightGCN_GCL đã train.
                        Nếu None → dùng dummy random recommendations.
        edge_index    : Đồ thị tương tác [2, 2|E|] (torch.Tensor).
        num_users (int): Tổng số users.
        num_items (int): Tổng số items.
        item_metadata (ItemMetadataStore, optional): Metadata cho items.
        user_item_train (dict, optional): {user_id: set[item_id]} đã tương tác
                                          (dùng để lọc exclude_seen).
    """

    def __init__(
        self,
        model=None,
        edge_index=None,
        num_users: int = 610,
        num_items: int = 9724,
        item_metadata: Optional[ItemMetadataStore] = None,
        user_item_train: Optional[dict] = None,
        user2id: Optional[dict] = None,
        id2user: Optional[dict] = None,
        item2id: Optional[dict] = None,
        id2item: Optional[dict] = None,
    ):
        self.model        = model
        self.edge_index   = edge_index
        self.num_users    = num_users
        self.num_items    = num_items
        self.metadata     = item_metadata or ItemMetadataStore()
        self.user_item_train = user_item_train or {}
        self.user2id      = user2id or {}
        self.id2user      = id2user or {}
        self.item2id      = item2id or {}
        self.id2item      = id2item or {}

        # Cache score matrix để tránh compute lại nhiều lần
        self._score_cache: Optional[list] = None

        if model is None:
            logger.warning(
                "GNNGCLAdapter: model=None → dùng dummy random recommendations. "
                "Truyền model đã train để có kết quả thực."
            )
        else:
            logger.info(
                f"GNNGCLAdapter: mode=live (SimGCL PyTorch, {num_users} users, {num_items} items)"
            )

    # ----------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------

    def _get_score_matrix(self) -> list:
        """
        Lấy ma trận điểm User×Item [M×N].
        Cache để tránh tính toán lại nhiều lần.
        """
        if self._score_cache is not None:
            return self._score_cache

        if self.model is None or self.edge_index is None:
            return None

        try:
            self._score_cache = self.model.predict_score_matrix(self.edge_index)
            logger.info(
                f"Score matrix computed via LightGCN+SimGCL: {len(self._score_cache)} users × "
                f"{len(self._score_cache[0]) if self._score_cache else 0} items"
            )
        except Exception as e:
            logger.error(f"Lỗi khi tính score matrix: {e}")
            self._score_cache = None

        return self._score_cache

    def get_top_n_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        exclude_conv_items: Optional[set] = None,
    ) -> list[dict]:
        """
        Lấy Top-N items gợi ý cho một user cụ thể bằng mô hình GNN+GCL thực tế.

        Args:
            user_id             (int)       : ID người dùng.
            n                   (int)       : Số items gợi ý. Mặc định 10.
            exclude_seen        (bool)      : Loại trừ items user đã tương tác.
            exclude_conv_items  (set, opt)  : Items đã được gợi ý trong hội thoại.

        Returns:
            list[dict]: Danh sách items đã xếp hạng theo điểm lan truyền đồ thị.
        """
        score_matrix = self._get_score_matrix()

        if score_matrix is not None:
            # Map raw user_id sang internal u_idx nếu có user2id
            u_idx = None
            if self.user2id:
                u_idx = self.user2id.get(str(user_id), self.user2id.get(user_id, None))
            if u_idx is None and 0 <= user_id < len(score_matrix):
                u_idx = user_id

            if u_idx is not None and 0 <= u_idx < len(score_matrix):
                scores_for_user = score_matrix[u_idx]
                item_scores = []
                for i_idx, score in enumerate(scores_for_user):
                    if self.id2item:
                        raw_id = self.id2item.get(i_idx, self.id2item.get(str(i_idx), i_idx))
                        try:
                            raw_id = int(raw_id)
                        except (ValueError, TypeError):
                            pass
                    else:
                        raw_id = i_idx
                    item_scores.append((raw_id, float(score)))
            else:
                item_scores = []
        else:
            item_scores = []

        if not item_scores:
            # Fallback nếu model chưa sẵn sàng
            all_item_ids = self.metadata.get_all_ids()
            if not all_item_ids:
                all_item_ids = list(range(self.num_items))
            item_scores = [
                (item_id, random.random())
                for item_id in all_item_ids
            ]

        # Exclude seen items
        exclude = set()
        if exclude_seen:
            u_train = self.user_item_train.get(user_id, set())
            if isinstance(u_train, list):
                u_train = set(u_train)
            exclude.update(u_train)
            if self.user2id and str(user_id) in self.user2id:
                u_train_idx = self.user_item_train.get(self.user2id[str(user_id)], set())
                if isinstance(u_train_idx, list):
                    u_train_idx = set(u_train_idx)
                exclude.update(u_train_idx)
        if exclude_conv_items:
            exclude.update(exclude_conv_items)

        # Lọc và sắp xếp
        filtered = [
            (item_id, score)
            for item_id, score in item_scores
            if item_id not in exclude
        ]
        filtered.sort(key=lambda x: -x[1])
        top_n = filtered[:n]

        # Gắn metadata
        results = []
        for item_id, score in top_n:
            meta = self.metadata.get(item_id)
            meta['score']      = float(score)
            meta['prompt_str'] = self.metadata.format_for_prompt(item_id, score)
            results.append(meta)

        return results

    def get_user_embedding(self, user_id: int):
        """
        Lấy embedding vector của user từ mô hình GNN+GCL thực tế.
        """
        if self.model is None or self.edge_index is None:
            return None

        try:
            import torch
            with torch.no_grad():
                if hasattr(self.model, '_propagate_raw'):
                    raw_stack = self.model._propagate_raw(self.edge_index)
                    user_emb, _ = self.model._get_final_emb(raw_stack)
                else:
                    user_emb, _ = self.model._propagate(self.edge_index)

                u_idx = user_id
                if self.user2id:
                    u_idx = self.user2id.get(str(user_id), self.user2id.get(user_id, user_id))
                if 0 <= u_idx < user_emb.size(0):
                    return user_emb[u_idx].cpu().numpy()
        except Exception as e:
            logger.error(f"Lỗi khi lấy user embedding: {e}")
        return None

    def invalidate_cache(self):
        """Xóa cache score matrix (gọi sau khi re-train model)."""
        self._score_cache = None
        logger.info("Score cache invalidated")

    def update_interaction(self, user_id: int, item_id: int):
        """
        Cập nhật tập items đã tương tác khi user click/accept gợi ý.
        Dùng cho feedback loop.
        """
        if user_id not in self.user_item_train:
            self.user_item_train[user_id] = set()
        if isinstance(self.user_item_train[user_id], list):
            self.user_item_train[user_id] = set(self.user_item_train[user_id])
        self.user_item_train[user_id].add(item_id)
        logger.debug(f"Interaction updated: user {user_id} → item {item_id}")

    def __repr__(self):
        mode = 'live (SimGCL PyTorch)' if self.model is not None else 'dummy'
        return (
            f"GNNGCLAdapter("
            f"mode={mode}, "
            f"users={self.num_users}, "
            f"items={self.num_items})"
        )


# ============================================================
# FACTORY FUNCTION
# ============================================================

def load_gnn_gcl_adapter(
    model_path: Optional[str] = None,
    data_dir: str = None,
    device: str = 'cpu',
) -> GNNGCLAdapter:
    """
    Factory function để tạo GNNGCLAdapter từ checkpoint đã lưu.
    Tự động nạp checkpoints/simgcl_movielens.pt nếu có.
    """
    if data_dir is None:
        data_dir = str(BASE_DIR / 'ml-latest-small')

    # Load item metadata
    metadata = ItemMetadataStore(
        movies_csv = os.path.join(data_dir, 'movies.csv'),
        tags_csv   = os.path.join(data_dir, 'tags.csv'),
    )

    # Load user-item training interactions
    user_item_train = defaultdict(set)
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    if os.path.exists(ratings_path):
        with open(ratings_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if float(row['rating']) >= 4.0:
                    user_item_train[int(row['userId'])].add(int(row['movieId']))
    num_users = max(user_item_train.keys()) + 1 if user_item_train else 610
    num_items = len(metadata.get_all_ids())

    # Tự động tìm checkpoint nếu chưa truyền
    if model_path is None:
        possible_paths = [
            os.path.join(str(BASE_DIR), 'checkpoints', 'simgcl_movielens.pt'),
            os.path.join('checkpoints', 'simgcl_movielens.pt'),
            os.path.join(os.getcwd(), 'checkpoints', 'simgcl_movielens.pt'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    # Load model nếu có
    model      = None
    edge_index = None
    user2id    = None
    id2user    = None
    item2id    = None
    id2item    = None

    if model_path and os.path.exists(model_path):
        try:
            import torch
            sys.path.insert(0, str(BASE_DIR / 'src'))
            from model_lightgcn_gcl import SimGCL

            checkpoint = torch.load(model_path, map_location=device)
            ckpt_num_users = checkpoint.get('num_users', num_users)
            ckpt_num_items = checkpoint.get('num_items', num_items)
            model = SimGCL(
                num_users     = ckpt_num_users,
                num_items     = ckpt_num_items,
                embedding_dim = checkpoint.get('embedding_dim', 32),
                K             = checkpoint.get('K', 2),
                temperature   = checkpoint.get('temperature', 0.2),
                lambda_cl     = checkpoint.get('lambda_cl', 0.2),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            edge_index = checkpoint.get('edge_index')
            user2id    = checkpoint.get('user2id')
            id2user    = checkpoint.get('id2user')
            item2id    = checkpoint.get('item2id')
            id2item    = checkpoint.get('id2item')
            if 'train_user_items' in checkpoint:
                user_item_train = checkpoint['train_user_items']
            num_users  = ckpt_num_users
            num_items  = ckpt_num_items
            logger.info(f"Model GNN+GCL loaded successfully from {model_path} (mode=live)")
        except Exception as e:
            logger.error(f"Không thể load model: {e}. Dùng dummy mode.")

    return GNNGCLAdapter(
        model           = model,
        edge_index      = edge_index,
        num_users       = num_users,
        num_items       = num_items,
        item_metadata   = metadata,
        user_item_train = dict(user_item_train),
        user2id         = user2id,
        id2user         = id2user,
        item2id         = item2id,
        id2item         = id2item,
    )
