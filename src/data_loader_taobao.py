"""
data_loader_taobao.py
Module nạp và tiền xử lý bộ dữ liệu Taobao User Behavior.
"""

import os
import sys
import glob
import math
import random
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class DataLoaderTaobao:
    def __init__(self, dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.user2id = {}
        self.item2id = {}

    def load_raw_data(self):
        parquet_file = None
        if os.path.isdir(self.dataset_path):
            files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
            if files:
                parquet_file = files[0]
        elif os.path.isfile(self.dataset_path):
            parquet_file = self.dataset_path

        if not parquet_file or not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Không tìm thấy file Parquet Taobao tại {self.dataset_path}")

        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_file)
            data_dict = table.to_pydict()
        except Exception:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            data_dict = df.to_dict(orient="list")

        seq_indices = data_dict.get("seq_idx", [])
        events = data_dict.get("type_event", [])
        n_rows = len(seq_indices)

        interactions = []
        for idx in range(n_rows):
            u_key = f"user_{seq_indices[idx]}"
            event_list = events[idx] if idx < len(events) else []
            for ev_idx, ev in enumerate(event_list):
                i_key = f"item_category_{ev}"
                if u_key not in self.user2id:
                    self.user2id[u_key] = len(self.user2id)
                if i_key not in self.item2id:
                    self.item2id[i_key] = len(self.item2id)

                u_id = self.user2id[u_key]
                i_id = self.item2id[i_key]
                interactions.append((u_id, i_id))

        return interactions

    def build_tfidf_features(self, num_items):
        import numpy as np
        num_features = 50
        np.random.seed(self.seed)
        features = np.random.randn(num_items, num_features).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
        genre_list = [f"taobao_cat_{i}" for i in range(num_features)]
        return features, genre_list

    def prepare_data(self):
        random.seed(self.seed)
        interactions = self.load_raw_data()
        num_users = len(self.user2id)
        num_items = len(self.item2id)

        user_interactions = defaultdict(list)
        for u, i in interactions:
            user_interactions[u].append(i)

        train_pairs, val_pairs, test_pairs = [], [], []
        train_user_items, val_user_items, test_user_items = defaultdict(set), defaultdict(set), defaultdict(set)

        for u, items in user_interactions.items():
            random.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * self.train_ratio))
            n_val = int(n * self.val_ratio)

            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:]

            for i in train_items:
                train_pairs.append((u, i))
                train_user_items[u].add(i)
            for i in val_items:
                val_pairs.append((u, i))
                val_user_items[u].add(i)
            for i in test_items:
                test_pairs.append((u, i))
                test_user_items[u].add(i)

        edge_index = [[], []]
        for u, i in train_pairs:
            item_node_id = num_users + i
            edge_index[0].append(u); edge_index[1].append(item_node_id)
            edge_index[0].append(item_node_id); edge_index[1].append(u)

        tfidf_features, genre_list = self.build_tfidf_features(num_items)

        return {
            "num_users": num_users, "num_items": num_items,
            "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs,
            "train_user_items": train_user_items, "val_user_items": val_user_items, "test_user_items": test_user_items,
            "edge_index": edge_index, "tfidf_features": tfidf_features, "genre_list": genre_list,
            "user2id": self.user2id, "item2id": self.item2id
        }
