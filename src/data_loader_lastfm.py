"""
data_loader_lastfm.py
Module nạp và tiền xử lý bộ dữ liệu Last.fm (hetrec2011-lastfm-2k) cho 4 mô hình RecSys.
"""

import os
import sys
import random
import math
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class DataLoaderLastFM:
    def __init__(self, dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.user2id = {}
        self.id2user = {}
        self.item2id = {}
        self.id2item = {}
        
        self.artist_meta = {} # item_id -> {"title": ..., "tags": [...]}
        
    def load_raw_data(self):
        user_artists_path = os.path.join(self.dataset_dir, "user_artists.dat")
        artists_path = os.path.join(self.dataset_dir, "artists.dat")
        user_tags_path = os.path.join(self.dataset_dir, "user_taggedartists.dat")
        tags_path = os.path.join(self.dataset_dir, "tags.dat")
        
        if not os.path.exists(user_artists_path):
            raise FileNotFoundError(f"Không tìm thấy user_artists.dat tại {self.dataset_dir}")
            
        # 1. Đọc tên nghệ sĩ
        if os.path.exists(artists_path):
            with open(artists_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        raw_artist_id = parts[0].strip()
                        artist_name = parts[1].strip()
                        if raw_artist_id not in self.item2id:
                            idx = len(self.item2id)
                            self.item2id[raw_artist_id] = idx
                            self.id2item[idx] = raw_artist_id
                        i_id = self.item2id[raw_artist_id]
                        self.artist_meta[i_id] = {"title": artist_name, "tags": []}

        # 2. Đọc thẻ gắn (tags)
        tag_dict = {}
        if os.path.exists(tags_path):
            with open(tags_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        tag_dict[parts[0].strip()] = parts[1].strip()
                        
        if os.path.exists(user_tags_path):
            with open(user_tags_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        raw_art_id = parts[1].strip()
                        tag_id = parts[2].strip()
                        if raw_art_id in self.item2id:
                            i_id = self.item2id[raw_art_id]
                            t_name = tag_dict.get(tag_id, "")
                            if t_name and i_id in self.artist_meta:
                                self.artist_meta[i_id]["tags"].append(t_name)

        # 3. Đọc tương tác user_artists
        interactions = []
        with open(user_artists_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    raw_user_id = parts[0].strip()
                    raw_artist_id = parts[1].strip()
                    weight = float(parts[2].strip())
                    
                    if weight >= 1.0: # Lọc tương tác nghe nhạc
                        if raw_artist_id not in self.item2id:
                            idx = len(self.item2id)
                            self.item2id[raw_artist_id] = idx
                            self.id2item[idx] = raw_artist_id
                            
                        if raw_user_id not in self.user2id:
                            idx = len(self.user2id)
                            self.user2id[raw_user_id] = idx
                            self.id2user[idx] = raw_user_id
                            
                        u_id = self.user2id[raw_user_id]
                        i_id = self.item2id[raw_artist_id]
                        interactions.append((u_id, i_id, weight))
                        
        return interactions

    def build_tfidf_features(self, num_items):
        import numpy as np
        all_tags = sorted(list(set(t for meta in self.artist_meta.values() for t in meta.get("tags", []))))
        if not all_tags:
            all_tags = ["pop", "rock", "metal", "electronic", "indie"]
            
        tag2id = {t: idx for idx, t in enumerate(all_tags)}
        num_tags = len(all_tags)
        
        tfidf_matrix = np.zeros((num_items, num_tags), dtype=np.float32)
        df = defaultdict(int)
        for i_id in range(num_items):
            tags = set(self.artist_meta.get(i_id, {}).get("tags", []))
            for t in tags:
                df[t] += 1
                
        for i_id in range(num_items):
            tags = self.artist_meta.get(i_id, {}).get("tags", [])
            if not tags:
                continue
            tf = 1.0 / len(tags)
            for t in tags:
                t_idx = tag2id[t]
                idf = math.log((num_items + 1.0) / (df[t] + 1.0)) + 1.0
                tfidf_matrix[i_id, t_idx] = tf * idf
                
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms
                
        return tfidf_matrix, all_tags

    def prepare_data(self):
        random.seed(self.seed)
        interactions = self.load_raw_data()
        num_users = len(self.user2id)
        num_items = len(self.item2id)
        
        user_interactions = defaultdict(list)
        for u, i, r in interactions:
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
