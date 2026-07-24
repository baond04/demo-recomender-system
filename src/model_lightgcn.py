"""
model_lightgcn.py
Module 5: Mô hình Mạng nơ-ron đồ thị LightGCN (Light Graph Convolutional Network)
Định nghĩa Class LightGCNConv (Lớp lan truyền Message Passing mức thấp chuẩn hóa độ bậc 1/√( |N_u| |N_i| ))
và Class LightGCNRecommender (Module bọc mức cao quản lý bảng nhúng e^{(0)}, lặp qua K tầng và gom tụ trung bình).
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
    class LightGCNConv(nn.Module):
        """
        Class LightGCNConv: Lớp lan truyền Message Passing mức thấp.
        Đại số tuyến tính lan truyền thông điệp trên đồ thị lưỡng phân theo hệ số chuẩn hóa độ bậc đối xứng:
        e_u^{(k+1)} = ∑_{i ∈ N_u} ( 1 / √( |N_u| |N_i| ) ) e_i^{(k)}
        (Không dùng ma trận trọng số W hay hàm ReLU)
        """
        def __init__(self):
            super(LightGCNConv, self).__init__()

        def forward(self, x, edge_index):
            """
            :param x: Vector nhúng của tất cả các nút (Users + Items) [N_total x d]
            :param edge_index: Danh sách cạnh 2 chiều [2 x 2*|E|]
            :return: Vector nhúng mới sau 1 tầng lan truyền Message Passing
            """
            src, dst = edge_index[0], edge_index[1]
            
            # Tính độ bậc (degree) cho từng nút
            num_nodes = x.size(0)
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float, device=x.device))
            
            # Hệ số chuẩn hóa đối xứng: norm = 1 / sqrt(deg[src] * deg[dst])
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
            
            # Message Passing: Gom tụ có trọng số norm
            msg = x[src] * norm.view(-1, 1)
            out = torch.zeros_like(x)
            out.scatter_add_(0, dst.view(-1, 1).expand(-1, x.size(1)), msg)
            
            return out


    class LightGCNRecommender(nn.Module):
        """
        Class LightGCNRecommender: Module quản lý mức cao.
        Quản lý bảng nhúng gốc e^{(0)}, điều phối gọi lặp Class LightGCNConv qua K tầng,
        tính trung bình cộng nhúng đa tầng e^{(final)} = (1/(K+1)) ∑ e^{(k)} và tính Dot Product.
        """
        def __init__(self, num_users, num_items, embedding_dim=32, K=2):
            super(LightGCNRecommender, self).__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.K = K
            
            # Bảng nhúng gốc e^{(0)} hợp nhất cho cả Users và Items [ (M+N) x d ]
            self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)
            
            # Lớp lan truyền thông điệp
            self.conv = LightGCNConv()

        def propagate(self, edge_index):
            """Lan truyền thông điệp qua K tầng và gom tụ trung bình đa tầng"""
            x = self.embedding.weight
            layer_embeddings = [x]
            
            # Gọi lặp qua K tầng
            for k in range(self.K):
                x = self.conv(x, edge_index)
                layer_embeddings.append(x)
                
            # Gom tụ trung bình đa tầng (Layer Average Pooling)
            final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
            
            user_embeddings = final_embeddings[:self.num_users]
            item_embeddings = final_embeddings[self.num_users:]
            
            return user_embeddings, item_embeddings

        def forward(self, users, pos_items, edge_index):
            user_embeds, item_embeds = self.propagate(edge_index)
            u_embed = user_embeds[users]
            i_embed = item_embeds[pos_items]
            scores = (u_embed * i_embed).sum(dim=-1)
            return scores

        def compute_bpr_loss(self, users, pos_items, neg_items, edge_index, reg_weight=1e-4):
            user_embeds, item_embeds = self.propagate(edge_index)
            
            u_embed = user_embeds[users]
            pos_embed = item_embeds[pos_items]
            neg_embed = item_embeds[neg_items]
            
            pos_scores = (u_embed * pos_embed).sum(dim=-1)
            neg_scores = (u_embed * neg_embed).sum(dim=-1)
            
            diff = pos_scores - neg_scores
            bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            
            # Regularization trên nhúng ban đầu e^{(0)}
            u_e0 = self.embedding(users)
            pos_e0 = self.embedding(self.num_users + pos_items)
            neg_e0 = self.embedding(self.num_users + neg_items)
            reg_loss = reg_weight * (u_e0.norm(2).pow(2) + pos_e0.norm(2).pow(2) + neg_e0.norm(2).pow(2)) / users.size(0)
            
            return bpr_loss + reg_loss

        def predict_score_matrix(self, edge_index):
            with torch.no_grad():
                user_embeds, item_embeds = self.propagate(edge_index)
                scores = torch.matmul(user_embeds, item_embeds.t())
                return scores.cpu().numpy().tolist()

else:
    # Fallback khi chưa cài PyTorch
    class LightGCNRecommender:
        def __init__(self, num_users, num_items, embedding_dim=32, K=2):
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.user_embedding = [[random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(num_users)]
            self.item_embedding = [[random.gauss(0, 0.1) for _ in range(embedding_dim)] for _ in range(num_items)]

        def predict_score_matrix(self, edge_index=None):
            score_matrix = [[0.0] * self.num_items for _ in range(self.num_users)]
            for u in range(self.num_users):
                for i in range(self.num_items):
                    score_matrix[u][i] = sum(self.user_embedding[u][k] * self.item_embedding[i][k] for k in range(self.embedding_dim))
            return score_matrix
