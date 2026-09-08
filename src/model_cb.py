import numpy as np

class ContentBasedRecommender:
    def __init__(self, tfidf_features, num_users, num_items):
        """
        :param tfidf_features: Ma trận TF-IDF đặc trưng Item [num_items x d]
        :param num_users: Tổng số User M
        :param num_items: Tổng số Item N
        """
        if isinstance(tfidf_features, np.ndarray):
            self.tfidf_features = tfidf_features.astype(np.float32)
        else:
            self.tfidf_features = np.array(tfidf_features, dtype=np.float32)
            
        self.num_users = num_users
        self.num_items = num_items
        self.d_features = self.tfidf_features.shape[1] if self.tfidf_features.ndim > 1 else 0
        self.user_profiles = None

    def fit(self, train_user_items):
        """
        Xây dựng vector User Profile bằng cách lấy trung bình cộng vector TF-IDF
        của các sản phẩm người dùng đã tương tác trong tập Train.
        """
        self.user_profiles = np.zeros((self.num_users, self.d_features), dtype=np.float32)
        
        for u in range(self.num_users):
            items = list(train_user_items.get(u, set()))
            if not items:
                continue
            
            # Tính tổng và trung bình cộng vector TF-IDF của các item đã tương tác
            u_vec = self.tfidf_features[items].sum(axis=0)
            u_vec = u_vec / len(items)
            
            # L2 Normalization for User Profile
            norm = np.linalg.norm(u_vec)
            if norm > 0:
                u_vec = u_vec / norm
            self.user_profiles[u] = u_vec

    def predict_score_matrix(self):
        """
        Tính ma trận điểm Cosine Similarity S_CB giữa toàn bộ User Profile và tất cả Item features.
        S_CB[u][i] = Cosine(U_u, V_i)
        :return: Ma trận điểm S_CB kích thước [num_users x num_items]
        """
        # Sử dụng phép nhân ma trận NumPy (BLAS vectorized) cực nhanh thay cho vòng lặp lồng
        score_matrix = np.dot(self.user_profiles, self.tfidf_features.T)
        return score_matrix

