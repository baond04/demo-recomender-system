"""
model_hybrid.py
Module 4: Mô hình Lai Recommender System (Hybrid - Late Fusion)
Thực hiện chuẩn hóa Min-Max điểm số thô theo từng hàng User,
sau đó kết hợp cộng trọng số Late Fusion: S_Hybrid = α * S̃_MF + (1 - α) * S̃_CB.
"""

class HybridRecommender:
    def __init__(self, alpha=0.6):
        """
        :param alpha: Trọng số ưu tiên mô hình MF (thường chọn 0.6)
        """
        self.alpha = alpha

    def user_wise_min_max_scale(self, score_matrix):
        """
        Chuẩn hóa Min-Max theo từng hàng User (User-wise Scaling)
        Đưa điểm dự đoán của từng User về cùng khoảng [0, 1]
        """
        num_users = len(score_matrix)
        num_items = len(score_matrix[0]) if num_users > 0 else 0
        
        scaled_matrix = [[0.0] * num_items for _ in range(num_users)]
        
        for u in range(num_users):
            row = score_matrix[u]
            min_val = min(row)
            max_val = max(row)
            diff = max_val - min_val
            
            if diff < 1e-8:
                scaled_matrix[u] = [0.0] * num_items
            else:
                scaled_matrix[u] = [(v - min_val) / diff for v in row]
                
        return scaled_matrix

    def predict_score_matrix(self, score_matrix_mf, score_matrix_cb):
        """
        Kết hợp Late Fusion giữa 2 ma trận điểm thô từ MF và CB
        :param score_matrix_mf: Ma trận điểm thô từ mô hình Matrix Factorization
        :param score_matrix_cb: Ma trận điểm thô từ mô hình Content-Based Filtering
        :return: Ma trận điểm kết hợp S_Hybrid [num_users x num_items]
        """
        num_users = len(score_matrix_mf)
        num_items = len(score_matrix_mf[0]) if num_users > 0 else 0
        
        # 1. Chuẩn hóa Min-Max 2 ma trận điểm theo hàng User
        scaled_mf = self.user_wise_min_max_scale(score_matrix_mf)
        scaled_cb = self.user_wise_min_max_scale(score_matrix_cb)
        
        # 2. Cộng kết hợp có trọng số Late Fusion
        score_hybrid = [[0.0] * num_items for _ in range(num_users)]
        for u in range(num_users):
            for i in range(num_items):
                score_hybrid[u][i] = self.alpha * scaled_mf[u][i] + (1.0 - self.alpha) * scaled_cb[u][i]
                
        return score_hybrid
