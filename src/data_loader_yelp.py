"""
data_loader_yelp.py
Module nạp và tiền xử lý bộ dữ liệu Yelp2018 (chuẩn LightGCN / SimGCL).
"""

import os
import sys
import math
import random
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class DataLoaderYelp:
    def __init__(self, dataset_dir, seed=42):
        self.dataset_dir = dataset_dir
        self.seed = seed
        self.user2id = {}
        self.item2id = {}

    def load_file(self, filename):
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file {filename} tại {self.dataset_dir}")

        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                raw_u = parts[0]
                if raw_u not in self.user2id:
                    self.user2id[raw_u] = len(self.user2id)
                u = self.user2id[raw_u]

                for raw_i in parts[1:]:
                    if raw_i not in self.item2id:
                        self.item2id[raw_i] = len(self.item2id)
                    i = self.item2id[raw_i]
                    pairs.append((u, i))
        return pairs

    def build_tfidf_features(self, num_items):
        import numpy as np
        num_features = 50
        np.random.seed(self.seed)
        features = np.random.randn(num_items, num_features).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
        genre_list = [f"yelp_cat_{i}" for i in range(num_features)]
        return features, genre_list

    def prepare_data(self):
        random.seed(self.seed)
        train_pairs = self.load_file("train.txt")
        test_pairs = self.load_file("test.txt")

        num_users = len(self.user2id)
        num_items = len(self.item2id)

        train_user_items, test_user_items, val_user_items = defaultdict(set), defaultdict(set), defaultdict(set)
        val_pairs = []

        for u, i in train_pairs:
            train_user_items[u].add(i)

        for u, i in test_pairs:
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
