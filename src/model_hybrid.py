"""
model_hybrid.py
Module 4: Mô hình Lai Recommender System (Hybrid - Late Fusion)
Thực hiện chuẩn hóa Min-Max điểm số thô theo từng hàng User,
sau đó kết hợp cộng trọng số Late Fusion: S_Hybrid = α * S̃_MF + (1 - α) * S̃_CB.
"""

import numpy as np

class HybridRecommender:
    def __init__(self, alpha=0.6):
        """
        :param alpha: Trọng số ưu tiên mô hình MF (thường chọn 0.6)
        """
        self.alpha = alpha

    def user_wise_min_max_scale(self, score_matrix):
        """
        Chuẩn hóa Min-Max theo từng hàng User (User-wise Scaling) bằng NumPy vectorization.
        Đưa điểm dự đoán của từng User về cùng khoảng [0, 1]
        """
        if not isinstance(score_matrix, np.ndarray):
            score_matrix = np.array(score_matrix, dtype=np.float32)
            
        min_vals = score_matrix.min(axis=1, keepdims=True)
        max_vals = score_matrix.max(axis=1, keepdims=True)
        diffs = max_vals - min_vals
        
        # Xử lý trường hợp diff == 0 để tránh chia cho 0
        valid_mask = diffs > 1e-8
        scaled = np.zeros_like(score_matrix, dtype=np.float32)
        np.divide(score_matrix - min_vals, diffs, out=scaled, where=valid_mask)
        
        return scaled

    def predict_score_matrix(self, score_matrix_mf, score_matrix_cb):
        """
        Kết hợp Late Fusion giữa 2 ma trận điểm thô từ MF và CB
        :param score_matrix_mf: Ma trận điểm thô từ mô hình Matrix Factorization
        :param score_matrix_cb: Ma trận điểm thô từ mô hình Content-Based Filtering
        :return: Ma trận điểm kết hợp S_Hybrid [num_users x num_items]
        """
        # 1. Chuẩn hóa Min-Max 2 ma trận điểm theo hàng User bằng NumPy
        scaled_mf = self.user_wise_min_max_scale(score_matrix_mf)
        scaled_cb = self.user_wise_min_max_scale(score_matrix_cb)
        
        # 2. Cộng kết hợp có trọng số Late Fusion
        score_hybrid = self.alpha * scaled_mf + (1.0 - self.alpha) * scaled_cb
        return score_hybrid
