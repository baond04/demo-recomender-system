"""
data_loader_netflix.py
Module nạp và tiền xử lý bộ dữ liệu Netflix Shows/Movies.
"""

import os
import sys
import glob
import math
import random
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class DataLoaderNetflix:
    def __init__(self, dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.user2id = {}
        self.item2id = {}
        self.item_texts = defaultdict(list)

    def load_raw_data(self):
        parquet_file = None
        if os.path.isdir(self.dataset_path):
            files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
            if files:
                parquet_file = files[0]
        elif os.path.isfile(self.dataset_path):
            parquet_file = self.dataset_path

        if not parquet_file or not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Không tìm thấy file Parquet Netflix tại {self.dataset_path}")

        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_file)
            data_dict = table.to_pydict()
        except Exception:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            data_dict = df.to_dict(orient="list")

        titles = data_dict.get("title", [])
        show_ids = data_dict.get("show_id", [])
        descs = data_dict.get("description", [])
        n_rows = len(titles)

        interactions = []
        for idx in range(n_rows):
            raw_item_id = str(show_ids[idx]) if idx < len(show_ids) else f"show_{idx}"
            txt = str(descs[idx]) if idx < len(descs) else ""

            if raw_item_id not in self.item2id:
                i_idx = len(self.item2id)
                self.item2id[raw_item_id] = i_idx

            i_id = self.item2id[raw_item_id]

            # Phân bổ các user ảo tương tác với item để dựng đồ thị bipartite
            raw_user_id = f"user_{idx % 400}"
            if raw_user_id not in self.user2id:
                u_idx = len(self.user2id)
                self.user2id[raw_user_id] = u_idx

            u_id = self.user2id[raw_user_id]
            interactions.append((u_id, i_id))

            if txt.strip():
                self.item_texts[i_id].append(txt.strip().lower())

        return interactions

    def build_tfidf_features(self, num_items):
        word_counts = defaultdict(int)
        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = set(w for w in full_txt.split() if len(w) > 3 and w.isalpha())
            for w in words:
                word_counts[w] += 1

        top_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)[:50]
        if not top_words:
            top_words = ["movie", "show", "series", "family", "drama", "action", "comedy", "love"]

        import numpy as np
        word2id = {w: idx for idx, w in enumerate(top_words)}
        num_words = len(top_words)

        tfidf_matrix = np.zeros((num_items, num_words), dtype=np.float32)
        df = defaultdict(int)
        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = set(w for w in full_txt.split() if w in word2id)
            for w in words:
                df[w] += 1

        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = [w for w in full_txt.split() if w in word2id]
            if not words:
                continue
            tf = 1.0 / len(words)
            for w in words:
                w_idx = word2id[w]
                idf = math.log((num_items + 1.0) / (df[w] + 1.0)) + 1.0
                tfidf_matrix[i_id, w_idx] = tf * idf

        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms

        return tfidf_matrix, top_words

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
