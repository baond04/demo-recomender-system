"""
model_mf.py
Module 3: Mô hình Collaborative Filtering - Matrix Factorization (MF)
Quản lý 2 bảng nhúng user_embedding và item_embedding, tính điểm Dot Product
và huấn luyện tối ưu với hàm mất mát xếp hạng BPR Loss.
"""

import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class MatrixFactorization(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=32):
            super(MatrixFactorization, self).__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            
            # 2 Bảng nhúng độc lập cho User và Item
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # Khởi tạo trọng số ngẫu nhiên chuẩn hoá Xavier/Glorot Normal
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)

        def forward(self, users, items):
            """Tính điểm dự đoán tích vô hướng p_u^T q_i"""
            u_embed = self.user_embedding(users)
            i_embed = self.item_embedding(items)
            scores = (u_embed * i_embed).sum(dim=-1)
            return scores

        def compute_bpr_loss(self, users, pos_items, neg_items, reg_weight=1e-4):
            """Hàm mất mát xếp hạng BPR Loss: -ln sigmoid(ŷ_pos - ŷ_neg) + reg"""
            u_embed = self.user_embedding(users)
            pos_embed = self.item_embedding(pos_items)
            neg_embed = self.item_embedding(neg_items)
            
            pos_scores = (u_embed * pos_embed).sum(dim=-1)
            neg_scores = (u_embed * neg_embed).sum(dim=-1)
            
            diff = pos_scores - neg_scores
            bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            
            # L2 Regularization
            reg_loss = reg_weight * (u_embed.norm(2).pow(2) + pos_embed.norm(2).pow(2) + neg_embed.norm(2).pow(2)) / users.size(0)
            
            return bpr_loss + reg_loss

        def predict_score_matrix(self):
            """Tính toàn bộ ma trận điểm S_MF = P * Q^T"""
            with torch.no_grad():
                P = self.user_embedding.weight # [M x d]
                Q = self.item_embedding.weight # [N x d]
                scores = torch.matmul(P, Q.t()) # [M x N]
                return scores.cpu().numpy()

else:
    # Class fallback thuần Python/NumPy khi chưa cài PyTorch
    class MatrixFactorization:
        def __init__(self, num_users, num_items, embedding_dim=32):
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            
            # Khởi tạo bảng nhúng thuần Python
            self.user_embedding = [[random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(num_users)]
            self.item_embedding = [[random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(num_items)]

        def predict_score_matrix(self):
            score_matrix = [[0.0] * self.num_items for _ in range(self.num_users)]
            for u in range(self.num_users):
                for i in range(self.num_items):
                    score_matrix[u][i] = sum(self.user_embedding[u][k] * self.item_embedding[i][k] for k in range(self.embedding_dim))
            return score_matrix
