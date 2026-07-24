"""
data_loader.py
Module 1: Tiền xử lý dữ liệu và Xây dựng Đồ thị Lưỡng phân cho Recommender System.
Xử lý dữ liệu MovieLens (ratings.csv, movies.csv), ánh xạ ID, trích xuất TF-IDF,
chia tập Train/Val/Test (80/10/10) và xây dựng danh sách cạnh edge_index cho GNN.
"""

import os
import sys
import random
import math
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
        
        # Ma trận TF-IDF: num_items x num_genres
        tfidf_matrix = [[0.0] * num_genres for _ in range(num_items)]
        
        # Tính DF (Document Frequency) cho từng thể loại
        df = defaultdict(int)
        for i_id in range(num_items):
            genres = self.movies_meta.get(i_id, {}).get("genres", [])
            for g in set(genres):
                df[g] += 1
                
        # Tính TF-IDF
        for i_id in range(num_items):
            genres = self.movies_meta.get(i_id, {}).get("genres", [])
            if not genres:
                continue
            tf = 1.0 / len(genres)
            for g in genres:
                g_idx = genre2id[g]
                idf = math.log((num_items + 1.0) / (df[g] + 1.0)) + 1.0
                tfidf_matrix[i_id][g_idx] = tf * idf
                
        # Chuẩn hóa L2 norm theo từng hàng Item
        for i_id in range(num_items):
            norm = math.sqrt(sum(v**2 for v in tfidf_matrix[i_id]))
            if norm > 0:
                tfidf_matrix[i_id] = [v / norm for v in tfidf_matrix[i_id]]
                
        return tfidf_matrix, all_genres

    def prepare_data(self):
        random.seed(self.seed)
        interactions = self.load_raw_data()
        
        num_users = len(self.user2id)
        num_items = len(self.item2id)
        
        # Gom nhóm tương tác theo User
        user_interactions = defaultdict(list)
        for u, i, r in interactions:
            user_interactions[u].append(i)
            
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        train_user_items = defaultdict(set)
        val_user_items = defaultdict(set)
        test_user_items = defaultdict(set)
        
        for u, items in user_interactions.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Đảm bảo ít nhất 1 sản phẩm ở tập Train nếu có dữ liệu
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
                
        # Dựng đồ thị lưỡng phân edge_index cho LightGCN (vô hướng: User <-> Item)
        edge_index = [[], []]
        for u, i in train_pairs:
            item_node_id = num_users + i
            # Cạnh u -> i
            edge_index[0].append(u)
            edge_index[1].append(item_node_id)
            # Cạnh i -> u (vô hướng)
            edge_index[0].append(item_node_id)
            edge_index[1].append(u)
            
        tfidf_features, genre_list = self.build_tfidf_features(num_items)
        
        data_dict = {
            "num_users": num_users,
            "num_items": num_items,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "train_user_items": train_user_items,
            "val_user_items": val_user_items,
            "test_user_items": test_user_items,
            "edge_index": edge_index,
            "tfidf_features": tfidf_features,
            "genre_list": genre_list,
            "user2id": self.user2id,
            "item2id": self.item2id
        }
        
        print("=== DA TAI THANH CONG MOVIELENS ===")
        print(f"* So luong Users (M): {num_users}")
        print(f"* So luong Items (N): {num_items}")
        print(f"* So tuong tac Train: {len(train_pairs)}")
        print(f"* So tuong tac Val:   {len(val_pairs)}")
        print(f"* So tuong tac Test:  {len(test_pairs)}")
        print(f"* Do thua ma tran:    {1.0 - (len(interactions)/(num_users*num_items)):.4%}")
        
        return data_dict

if __name__ == "__main__":
    loader = DataLoaderMovieLens(r"d:\DoAnTotNghiep\recommender-system\ml-latest-small")
    data = loader.prepare_data()
