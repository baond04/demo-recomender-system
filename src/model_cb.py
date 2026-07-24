"""
model_cb.py
Module 2: Mô hình Content-Based Filtering (CB)
Biến đổi thuộc tính TF-IDF, tổng hợp User Profile Vector và tính Cosine Similarity.
"""

import math

class ContentBasedRecommender:
    def __init__(self, tfidf_features, num_users, num_items):
        """
        :param tfidf_features: Ma trận TF-IDF đặc trưng Item [num_items x d]
        :param num_users: Tổng số User M
        :param num_items: Tổng số Item N
        """
        self.tfidf_features = tfidf_features
        self.num_users = num_users
        self.num_items = num_items
        self.d_features = len(tfidf_features[0]) if tfidf_features else 0
        self.user_profiles = None

    def fit(self, train_user_items):
        """
        Xây dựng vector User Profile bằng cách lấy trung bình cộng vector TF-IDF
        của các sản phẩm người dùng đã tương tác trong tập Train.
        """
        self.user_profiles = [[0.0] * self.d_features for _ in range(self.num_users)]
        
        for u in range(self.num_users):
            items = train_user_items.get(u, set())
            if not items:
                continue
            
            # Sum TF-IDF vectors of liked items
            for i in items:
                for k in range(self.d_features):
                    self.user_profiles[u][k] += self.tfidf_features[i][k]
                    
            # Average vector
            n_items = len(items)
            for k in range(self.d_features):
                self.user_profiles[u][k] /= n_items
                
            # L2 Normalization for User Profile
            norm = math.sqrt(sum(v**2 for v in self.user_profiles[u]))
            if norm > 0:
                self.user_profiles[u] = [v / norm for v in self.user_profiles[u]]

    def predict_score_matrix(self):
        """
        Tính ma trận điểm Cosine Similarity S_CB giữa toàn bộ User Profile và tất cả Item features.
        S_CB[u][i] = Cosine(U_u, V_i)
        :return: Ma trận điểm S_CB kích thước [num_users x num_items]
        """
        score_matrix = [[0.0] * self.num_items for _ in range(self.num_users)]
        
        for u in range(self.num_users):
            u_vec = self.user_profiles[u]
            u_norm = math.sqrt(sum(v**2 for v in u_vec))
            
            if u_norm == 0:
                continue
                
            for i in range(self.num_items):
                i_vec = self.tfidf_features[i]
                # Dot product (vì cả u_vec và i_vec đã được L2 normalized nên Dot Product chính là Cosine Similarity)
                dot = sum(u_vec[k] * i_vec[k] for k in range(self.d_features))
                score_matrix[u][i] = dot
                
        return score_matrix
