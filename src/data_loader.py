"""
data_loader.py
Module 1: Tiền xử lý dữ liệu và Xây dựng Đồ thị Lưỡng phân cho 3 bộ dữ liệu:
1. MovieLens 100K (DataLoaderMovieLens)
2. Last.fm (DataLoaderLastFM)
3. Amazon Musical Instruments (DataLoaderAmazon)

Thực hiện mã hóa định danh hai chiều (user2id, item2id), trích xuất ma trận đặc trưng L2-normalized TF-IDF,
chia tập Train/Val/Test (80/10/10) và xây dựng ma trận kề edge_index cho mô hình GNN (LightGCN).
"""

import os
import sys
import glob
import json
import random
import math
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from data_loader_tripadvisor import DataLoaderTripAdvisor
    from data_loader_yelp import DataLoaderYelp
    from data_loader_taobao import DataLoaderTaobao
    from data_loader_netflix import DataLoaderNetflix
except ImportError:
    pass




# ==============================================================================
# 1. DATA LOADER CHO MOVIELENS 100K
# ==============================================================================
class DataLoaderMovieLens:
    def __init__(self, dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, min_rating=3.0, seed=42):
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_rating = min_rating
        self.seed = seed
        
        self.user2id = {}
        self.id2user = {}
        self.item2id = {}
        self.id2item = {}
        
        self.movies_meta = {} # item_id -> {"title": ..., "genres": [...]}
        
    def load_raw_data(self):
        movies_path = os.path.join(self.dataset_dir, "movies.csv")
        ratings_path = os.path.join(self.dataset_dir, "ratings.csv")
        
        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Không tìm thấy file dataset tại {self.dataset_dir}. Cần có movies.csv và ratings.csv.")
            
        # 1. Đọc file movies.csv
        with open(movies_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    raw_movie_id = parts[0].strip()
                    genres_str = parts[-1].strip()
                    title = ",".join(parts[1:-1]).strip('"')
                    
                    if raw_movie_id not in self.item2id:
                        idx = len(self.item2id)
                        self.item2id[raw_movie_id] = idx
                        self.id2item[idx] = raw_movie_id
                        
                    item_id = self.item2id[raw_movie_id]
                    genres = genres_str.split("|") if genres_str != "(no genres listed)" else []
                    self.movies_meta[item_id] = {"title": title, "genres": genres}
                    
        # 2. Đọc file ratings.csv
        interactions = []
        with open(ratings_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    raw_user_id = parts[0].strip()
                    raw_movie_id = parts[1].strip()
                    rating = float(parts[2].strip())
                    
                    # Lọc tương tác dương (rating >= min_rating)
                    if rating >= self.min_rating and raw_movie_id in self.item2id:
                        if raw_user_id not in self.user2id:
                            idx = len(self.user2id)
                            self.user2id[raw_user_id] = idx
                            self.id2user[idx] = raw_user_id
                            
                        u_id = self.user2id[raw_user_id]
                        i_id = self.item2id[raw_movie_id]
                        interactions.append((u_id, i_id, rating))
                        
        return interactions

    def build_tfidf_features(self, num_items):
        """Trích xuất ma trận TF-IDF thể loại phim cho Content-Based Filtering"""
        all_genres = sorted(list(set(g for meta in self.movies_meta.values() for g in meta["genres"])))
        genre2id = {g: idx for idx, g in enumerate(all_genres)}
        num_genres = len(all_genres)
        
        tfidf_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
        
        df = defaultdict(int)
        for i_id in range(num_items):
            genres = self.movies_meta.get(i_id, {}).get("genres", [])
            for g in set(genres):
                df[g] += 1
                
        for i_id in range(num_items):
            genres = self.movies_meta.get(i_id, {}).get("genres", [])
            if not genres:
                continue
            tf = 1.0 / len(genres)
            for g in genres:
                g_idx = genre2id[g]
                idf = math.log((num_items + 1.0) / (df[g] + 1.0)) + 1.0
                tfidf_matrix[i_id, g_idx] = tf * idf
                
        # L2 Normalization theo hàng Item
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms
                
        return tfidf_matrix, all_genres

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
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            if n_train == 0 and n > 0:
                n_train = 1
                
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
        
        print("=== DA TAI THANH CONG MOVIELENS ===")
        print(f"* So luong Users (M): {num_users}")
        print(f"* So luong Items (N): {num_items}")
        print(f"* So tuong tac Train: {len(train_pairs)}")
        print(f"* So tuong tac Val:   {len(val_pairs)}")
        print(f"* So tuong tac Test:  {len(test_pairs)}")
        print(f"* Do thua ma tran:    {1.0 - (len(interactions)/(num_users*num_items)):.4%}")
        
        return {
            "num_users": num_users, "num_items": num_items,
            "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs,
            "train_user_items": train_user_items, "val_user_items": val_user_items, "test_user_items": test_user_items,
            "edge_index": edge_index, "tfidf_features": tfidf_features, "genre_list": genre_list,
            "user2id": self.user2id, "item2id": self.item2id
        }


# ==============================================================================
# 2. DATA LOADER CHO LAST.FM
# ==============================================================================
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


# ==============================================================================
# 3. DATA LOADER CHO AMAZON MUSICAL INSTRUMENTS
# ==============================================================================
class DataLoaderAmazon:
    def __init__(self, dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, min_rating=3.0, seed=42):
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_rating = min_rating
        self.seed = seed
        
        self.user2id = {}
        self.id2user = {}
        self.item2id = {}
        self.id2item = {}
        self.item_texts = defaultdict(list)
        
    def load_raw_data(self):
        json_file = None
        if os.path.isdir(self.dataset_dir):
            files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
            if files:
                json_file = files[0]
        elif os.path.isfile(self.dataset_dir):
            json_file = self.dataset_dir
            
        if not json_file or not os.path.exists(json_file):
            raise FileNotFoundError(f"Không tìm thấy file JSON dataset tại {self.dataset_dir}")
            
        interactions = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                raw_user_id = data.get("reviewerID")
                raw_item_id = data.get("asin")
                rating = float(data.get("overall", 0.0))
                text = data.get("summary", "") + " " + data.get("reviewText", "")
                
                if rating >= self.min_rating and raw_user_id and raw_item_id:
                    if raw_item_id not in self.item2id:
                        idx = len(self.item2id)
                        self.item2id[raw_item_id] = idx
                        self.id2item[idx] = raw_item_id
                    if raw_user_id not in self.user2id:
                        idx = len(self.user2id)
                        self.user2id[raw_user_id] = idx
                        self.id2user[idx] = raw_user_id
                        
                    u_id = self.user2id[raw_user_id]
                    i_id = self.item2id[raw_item_id]
                    interactions.append((u_id, i_id, rating))
                    
                    if text.strip():
                        self.item_texts[i_id].append(text.strip().lower())
                        
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
            top_words = ["quality", "sound", "guitar", "great", "price", "easy"]
            
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
