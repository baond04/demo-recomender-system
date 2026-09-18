"""
model_lightgcn_gcl.py
Module: Mô hình GNN + GCL (Graph Contrastive Learning) cho Hệ Tư Vấn
=======================================================================
Tích hợp Graph Contrastive Learning (Self-Supervised Learning) vào LightGCN
nhằm khắc phục hạn chế dữ liệu thưa (Sparsity) và nhiễu (Noise).

Kiến trúc:
    Bước 1 — Augmentation: Tạo 2 views G1, G2 từ đồ thị gốc G
              bằng Edge Dropout và Node Dropout.
    Bước 2 — Dual Encoder: Chạy LightGCN trên G1 và G2 (shared weights).
    Bước 3 — InfoNCE Loss: Tối thiểu hóa khoảng cách giữa cùng 1 node
              ở 2 views (Positive Pairs), tối đa hóa khoảng cách với
              các node khác (Negative Pairs).
    Bước 4 — Tổng hợp: L_total = L_BPR + λ * L_CL

Tham khảo:
    SGL:      Wu et al. (SIGIR 2021) - Self-supervised Graph Learning for RS
    SimGCL:   Yu et al.  (SIGIR 2022) - Are GCL Augmentations Necessary?
    LightGCL: Cai et al. (ICLR 2023) - Simple yet Effective GCL for RS
"""

import random
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# HÀM AUGMENTATION — TẠO VIEWS TỪ ĐỒ THỊ GỐC
# ============================================================

def edge_dropout(edge_index, drop_rate=0.2):
    """
    Edge Dropout Augmentation (Phương pháp SGL).
    Xóa ngẫu nhiên drop_rate% số cạnh trong đồ thị, tạo view G1.

    Args:
        edge_index (Tensor): Ma trận cạnh [2, 2*|E|] (đồ thị vô hướng).
        drop_rate  (float) : Tỷ lệ cạnh bị xóa. Mặc định 0.2 (20%).

    Returns:
        Tensor: edge_index mới sau khi xóa cạnh [2, 2*|E_kept|].

    Công thức:
        G1 = (U ∪ I, E \ ΔE),  |ΔE| = drop_rate × |E|
    """
    if not HAS_TORCH:
        return edge_index

    num_edges = edge_index.size(1)
    # Tạo mask ngẫu nhiên: True = giữ lại, False = xóa
    keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
    return edge_index[:, keep_mask]


def node_dropout(edge_index, num_nodes, drop_rate=0.1):
    """
    Node Dropout Augmentation (Phương pháp SGL).
    Ẩn ngẫu nhiên drop_rate% node, xóa tất cả cạnh liên quan đến node đó.
    Tạo view G2.

    Args:
        edge_index (Tensor): Ma trận cạnh [2, 2*|E|].
        num_nodes  (int)   : Tổng số node (Users + Items).
        drop_rate  (float) : Tỷ lệ node bị ẩn. Mặc định 0.1 (10%).

    Returns:
        Tensor: edge_index mới sau khi ẩn node.

    Công thức:
        G2 = (Ũ ∪ Ĩ, E), |Ũ|=0.9|U|, |Ĩ|=0.9|I|
    """
    if not HAS_TORCH:
        return edge_index

    # Mask node: True = giữ lại, False = ẩn
    node_mask = torch.rand(num_nodes, device=edge_index.device) >= drop_rate
    # Giữ lại cạnh nếu cả 2 đầu node đều được giữ
    src_mask = node_mask[edge_index[0]]
    dst_mask = node_mask[edge_index[1]]
    keep_mask = src_mask & dst_mask
    return edge_index[:, keep_mask]


def random_walk_subgraph(edge_index, num_nodes, walk_length=3, num_starts=None):
    """
    Random Walk Subgraph Augmentation.
    Lấy mẫu đồ thị con dựa trên Random Walk từ các node khởi đầu ngẫu nhiên.
    Phù hợp với đồ thị lớn, nhiều cụm.

    Args:
        edge_index  (Tensor): Ma trận cạnh [2, 2*|E|].
        num_nodes   (int)   : Tổng số node.
        walk_length (int)   : Độ dài mỗi đường đi ngẫu nhiên.
        num_starts  (int)   : Số node khởi đầu. Mặc định = 50% tổng node.

    Returns:
        Tensor: edge_index của đồ thị con.
    """
    if not HAS_TORCH:
        return edge_index

    if num_starts is None:
        num_starts = max(1, num_nodes // 2)

    # Xây dựng danh sách kề
    adj = [[] for _ in range(num_nodes)]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    for s, d in zip(src_list, dst_list):
        adj[s].append(d)

    # Random Walk
    visited_nodes = set()
    start_nodes = random.sample(range(num_nodes), min(num_starts, num_nodes))
    for start in start_nodes:
        cur = start
        for _ in range(walk_length):
            visited_nodes.add(cur)
            neighbors = adj[cur]
            if not neighbors:
                break
            cur = random.choice(neighbors)
        visited_nodes.add(cur)

    visited_tensor = torch.tensor(list(visited_nodes), device=edge_index.device)
    node_in_walk = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    node_in_walk[visited_tensor] = True

    src_mask = node_in_walk[edge_index[0]]
    dst_mask = node_in_walk[edge_index[1]]
    keep_mask = src_mask & dst_mask
    return edge_index[:, keep_mask]


# ============================================================
# CORE LAYERS — LightGCN Message Passing (tái sử dụng từ model_lightgcn.py)
# ============================================================

if HAS_TORCH:
    class LightGCNConv(nn.Module):
        """
        LightGCN Convolution Layer — Lớp lan truyền thông điệp mức thấp.
        Chuẩn hóa đối xứng theo độ bậc:
            e_u^(k+1) = Σ_{i∈N_u} (1/√(|N_u||N_i|)) · e_i^(k)
        Không dùng ma trận trọng số W hay hàm kích hoạt phi tuyến.
        """
        def __init__(self):
            super().__init__()

        def forward(self, x, edge_index):
            """
            Args:
                x          (Tensor): Embedding tất cả node [N_total × d].
                edge_index (Tensor): Danh sách cạnh [2 × 2|E|].
            Returns:
                Tensor: Embedding sau 1 bước lan truyền [N_total × d].
            """
            src, dst = edge_index[0], edge_index[1]
            num_nodes = x.size(0)

            # Tính degree và hệ số chuẩn hóa đối xứng
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, src, torch.ones(src.size(0), dtype=torch.float, device=x.device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

            # Gom tụ có trọng số
            msg = x[src] * norm.unsqueeze(1)
            out = torch.zeros_like(x)
            out.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), msg)
            return out


    # ============================================================
    # MODULE CHÍNH — LightGCN + GCL
    # ============================================================

    class LightGCN_GCL(nn.Module):
        """
        LightGCN_GCL: Mô hình GNN + Graph Contrastive Learning cho Hệ Tư Vấn.

        Kiến trúc tổng thể:
            G (đồ thị gốc)
            ├── Augment_1 (Edge Dropout)  → G1 → LightGCN Encoder → z^(1)
            └── Augment_2 (Node Dropout)  → G2 → LightGCN Encoder → z^(2)
                                                         ↓
                                               InfoNCE Contrastive Loss (L_CL)
                                               + BPR Recommendation Loss (L_BPR)
                                                         ↓
                                           L_total = L_BPR + λ · L_CL

        Tham số:
            num_users     (int)   : Số lượng User M.
            num_items     (int)   : Số lượng Item N.
            embedding_dim (int)   : Chiều không gian nhúng d. Mặc định 32.
            K             (int)   : Số tầng lan truyền. Mặc định 2.
            aug_type      (str)   : Loại augmentation —
                                    'edge_dropout' | 'node_dropout' | 'both'.
            edge_drop_rate (float): Tỷ lệ xóa cạnh cho Edge Dropout. Mặc định 0.2.
            node_drop_rate (float): Tỷ lệ ẩn node cho Node Dropout. Mặc định 0.1.
            temperature   (float) : Hệ số τ trong InfoNCE. Mặc định 0.2.
            lambda_cl     (float) : Hệ số λ cân bằng L_BPR và L_CL. Mặc định 0.2.
        """

        def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=32,
            K=2,
            aug_type='both',
            edge_drop_rate=0.2,
            node_drop_rate=0.1,
            temperature=0.2,
            lambda_cl=0.2
        ):
            super().__init__()
            self.num_users     = num_users
            self.num_items     = num_items
            self.embedding_dim = embedding_dim
            self.K             = K
            self.aug_type      = aug_type
            self.edge_drop_rate = edge_drop_rate
            self.node_drop_rate = node_drop_rate
            self.temperature   = temperature
            self.lambda_cl     = lambda_cl

            # Bảng nhúng gốc e^(0): chung cho cả User và Item [(M+N) × d]
            self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)

            # Lớp lan truyền dùng chung (Shared Weights) cho cả 2 views
            self.conv = LightGCNConv()

        # -------------------------------------------------------
        # PHƯƠNG THỨC PROPAGATE
        # -------------------------------------------------------

        def _propagate(self, edge_index):
            """
            Lan truyền thông điệp qua K tầng và gom tụ trung bình đa tầng.
            e* = (1/(K+1)) · Σ_{k=0}^{K} e^(k)

            Args:
                edge_index (Tensor): Danh sách cạnh có thể đã bị augment.
            Returns:
                Tuple(Tensor, Tensor): (user_embeddings, item_embeddings)
            """
            x = self.embedding.weight
            layer_embs = [x]
            for _ in range(self.K):
                x = self.conv(x, edge_index)
                layer_embs.append(x)

            # Mean Pooling: lấy trung bình tất cả tầng
            final = torch.stack(layer_embs, dim=0).mean(dim=0)
            user_emb = final[:self.num_users]
            item_emb = final[self.num_users:]
            return user_emb, item_emb

        # -------------------------------------------------------
        # PHƯƠNG THỨC AUGMENTATION
        # -------------------------------------------------------

        def _augment(self, edge_index):
            """
            Tạo 2 augmented views từ đồ thị gốc theo aug_type đã cấu hình.

            Returns:
                Tuple(Tensor, Tensor): (edge_index_view1, edge_index_view2)
            """
            num_nodes = self.num_users + self.num_items

            if self.aug_type == 'edge_dropout':
                # Cả 2 views đều dùng Edge Dropout với random seed khác nhau
                ei1 = edge_dropout(edge_index, drop_rate=self.edge_drop_rate)
                ei2 = edge_dropout(edge_index, drop_rate=self.edge_drop_rate)

            elif self.aug_type == 'node_dropout':
                # Cả 2 views đều dùng Node Dropout
                ei1 = node_dropout(edge_index, num_nodes, drop_rate=self.node_drop_rate)
                ei2 = node_dropout(edge_index, num_nodes, drop_rate=self.node_drop_rate)

            else:
                # aug_type == 'both' (mặc định, theo SGL paper)
                # View 1: Edge Dropout — View 2: Node Dropout
                ei1 = edge_dropout(edge_index, drop_rate=self.edge_drop_rate)
                ei2 = node_dropout(edge_index, num_nodes, drop_rate=self.node_drop_rate)

            return ei1, ei2

        # -------------------------------------------------------
        # HÀM MẤT MÁT InfoNCE (CONTRASTIVE LOSS)
        # -------------------------------------------------------

        def _info_nce_loss(self, z1_user, z2_user, z1_item, z2_item):
            """
            Tính InfoNCE Contrastive Loss cho cả User và Item.

            Công thức:
                L_CL = -Σ_u log [ exp(sim(z_u^(1), z_u^(2)) / τ) ]
                                  ─────────────────────────────────
                                  Σ_{u'} exp(sim(z_u^(1), z_{u'}^(2)) / τ)

            Trong đó:
                sim(·) = Cosine Similarity
                τ = self.temperature (temperature parameter)
                Positive pair: cùng node từ 2 views khác nhau
                Negative pair: các node khác trong cùng mini-batch

            Args:
                z1_user (Tensor): User embeddings từ View 1 [M × d].
                z2_user (Tensor): User embeddings từ View 2 [M × d].
                z1_item (Tensor): Item embeddings từ View 1 [N × d].
                z2_item (Tensor): Item embeddings từ View 2 [N × d].

            Returns:
                Tensor: Scalar loss value.
            """
            def nce_loss_single(z1, z2):
                # Chuẩn hóa L2 để tính Cosine Similarity
                z1_norm = F.normalize(z1, dim=1)
                z2_norm = F.normalize(z2, dim=1)

                # Ma trận similarity [N × N]: sim[i][j] = cos(z1_i, z2_j)
                logits = torch.mm(z1_norm, z2_norm.t()) / self.temperature

                # Label: phần tử diagonal là positive pairs
                labels = torch.arange(z1.size(0), device=z1.device)

                # CrossEntropy = InfoNCE Loss
                loss = F.cross_entropy(logits, labels)
                return loss

            # Tính cho cả 2 hướng và trung bình
            loss_user = (nce_loss_single(z1_user, z2_user) +
                         nce_loss_single(z2_user, z1_user)) / 2.0
            loss_item = (nce_loss_single(z1_item, z2_item) +
                         nce_loss_single(z2_item, z1_item)) / 2.0

            return (loss_user + loss_item) / 2.0

        # -------------------------------------------------------
        # HÀM MẤT MÁT BPR (RECOMMENDATION LOSS)
        # -------------------------------------------------------

        def _bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb, reg_weight=1e-4):
            """
            Tính BPR (Bayesian Personalized Ranking) Loss.

            Công thức:
                L_BPR = -Σ_{(u,i,j)} ln σ(ŷ_ui - ŷ_uj) + λ‖Θ‖²

            Trong đó:
                ŷ_ui = e_u · e_i  (Dot Product — điểm dự đoán)
                i: Item dương (User đã tương tác trong Train set)
                j: Item âm  (User chưa tương tác — Negative Sampling)
                λ: L2 regularization weight

            Args:
                users     (Tensor): Indices của User trong batch.
                pos_items (Tensor): Indices của Item dương tương ứng.
                neg_items (Tensor): Indices của Item âm (sampled).
                user_emb  (Tensor): User embeddings đã propagate [M × d].
                item_emb  (Tensor): Item embeddings đã propagate [N × d].
                reg_weight (float): Hệ số L2 regularization.

            Returns:
                Tensor: Scalar BPR loss.
            """
            u_e  = user_emb[users]
            pi_e = item_emb[pos_items]
            ni_e = item_emb[neg_items]

            pos_scores = (u_e * pi_e).sum(dim=-1)
            neg_scores = (u_e * ni_e).sum(dim=-1)

            bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            # L2 regularization trên embedding ban đầu e^(0)
            u_e0  = self.embedding(users)
            pi_e0 = self.embedding(self.num_users + pos_items)
            ni_e0 = self.embedding(self.num_users + neg_items)
            reg = reg_weight * (
                u_e0.norm(2).pow(2) +
                pi_e0.norm(2).pow(2) +
                ni_e0.norm(2).pow(2)
            ) / users.size(0)

            return bpr + reg

        # -------------------------------------------------------
        # HÀM MẤT MÁT TỔNG HỢP
        # -------------------------------------------------------

        def compute_total_loss(self, users, pos_items, neg_items, edge_index, reg_weight=1e-4):
            """
            Tính hàm mất mát tổng hợp GCL:
                L_total = L_BPR + λ · L_CL

            Quy trình:
                1. Augment đồ thị G → G1, G2
                2. Propagate LightGCN trên G1 → z^(1)
                   Propagate LightGCN trên G2 → z^(2)
                3. Propagate LightGCN trên G gốc → dùng cho BPR
                4. Tính L_CL = InfoNCE(z^(1), z^(2))
                5. Tính L_BPR dùng embedding từ G gốc
                6. Trả về L_total = L_BPR + λ · L_CL

            Args:
                users     (Tensor): User indices trong batch.
                pos_items (Tensor): Item dương indices.
                neg_items (Tensor): Item âm indices.
                edge_index (Tensor): Đồ thị gốc [2, 2|E|].
                reg_weight (float) : L2 regularization.

            Returns:
                Tuple(Tensor, float, float):
                    (L_total, L_BPR.item(), L_CL.item())
            """
            # Bước 1: Augmentation tạo 2 views
            ei1, ei2 = self._augment(edge_index)

            # Bước 2: Dual Encoder (Shared Weights)
            z1_user, z1_item = self._propagate(ei1)
            z2_user, z2_item = self._propagate(ei2)

            # Bước 3: Propagate trên đồ thị gốc cho BPR
            user_emb, item_emb = self._propagate(edge_index)

            # Bước 4: InfoNCE Loss
            loss_cl = self._info_nce_loss(z1_user, z2_user, z1_item, z2_item)

            # Bước 5: BPR Loss
            loss_bpr = self._bpr_loss(users, pos_items, neg_items,
                                      user_emb, item_emb, reg_weight)

            # Bước 6: Tổng hợp
            loss_total = loss_bpr + self.lambda_cl * loss_cl

            return loss_total, loss_bpr.item(), loss_cl.item()

        # -------------------------------------------------------
        # DỰ ĐOÁN MA TRẬN ĐIỂM (INFERENCE)
        # -------------------------------------------------------

        def predict_score_matrix(self, edge_index):
            """
            Dự đoán ma trận điểm User×Item cho Inference/Evaluation.
            Dùng đồ thị gốc (không augment) và trả về Dot Product.

            Args:
                edge_index (Tensor): Đồ thị gốc.

            Returns:
                list[list[float]]: Ma trận điểm [M × N] dưới dạng Python list.
            """
            with torch.no_grad():
                user_emb, item_emb = self._propagate(edge_index)
                scores = torch.matmul(user_emb, item_emb.t())
                return scores.cpu().numpy().tolist()


    # ============================================================
    # MODULE SimGCL — Augmentation trên Embedding Space
    # ============================================================

    class SimGCL(nn.Module):
        """
        SimGCL (Simple Graph Contrastive Learning) — Yu et al., SIGIR 2022.
        Thay vì augment cấu trúc đồ thị, SimGCL thêm nhiễu ngẫu nhiên
        trực tiếp vào không gian embedding để tạo các views.

        Ưu điểm so với SGL:
            - Không cần augment đồ thị (ít tốn kém hơn)
            - Đơn giản hơn, hiệu quả tương đương hoặc tốt hơn
            - Phù hợp với đồ thị thưa, tránh mất thông tin cấu trúc

        Công thức tạo View:
            z' = z + ε · sign(Δz),  ε ~ Uniform(0, ε_max)
            Δz ~ N(0, I) — nhiễu ngẫu nhiên chuẩn hóa
        """

        def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=32,
            K=2,
            epsilon=0.1,
            temperature=0.2,
            lambda_cl=0.2
        ):
            super().__init__()
            self.num_users     = num_users
            self.num_items     = num_items
            self.embedding_dim = embedding_dim
            self.K             = K
            self.epsilon       = epsilon      # Cường độ nhiễu
            self.temperature   = temperature
            self.lambda_cl     = lambda_cl

            self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)
            self.conv = LightGCNConv()

        def _propagate_raw(self, edge_index):
            """Propagate và trả về embedding RAW tất cả các tầng."""
            x = self.embedding.weight
            layer_embs = [x]
            for _ in range(self.K):
                x = self.conv(x, edge_index)
                layer_embs.append(x)
            # Trả về stacked tensor [K+1, N_total, d]
            return torch.stack(layer_embs, dim=0)

        def _add_noise_view(self, layer_stack):
            """
            Thêm nhiễu có hướng vào từng tầng để tạo augmented view.
            Công thức: z' = z + ε · sign(Δ),  Δ ~ N(0, I)
            """
            noise = torch.randn_like(layer_stack)
            noise = F.normalize(noise, p=2, dim=-1)
            return layer_stack + self.epsilon * noise.sign()

        def _get_final_emb(self, layer_stack):
            """Mean Pooling: e* = mean(e^(0), ..., e^(K))."""
            final = layer_stack.mean(dim=0)
            return final[:self.num_users], final[self.num_users:]

        def _info_nce_loss(self, z1, z2):
            """InfoNCE Loss (Cosine Similarity + Cross Entropy)."""
            z1_n = F.normalize(z1, dim=1)
            z2_n = F.normalize(z2, dim=1)
            logits = torch.mm(z1_n, z2_n.t()) / self.temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            return F.cross_entropy(logits, labels)

        def compute_total_loss(self, users, pos_items, neg_items, edge_index, reg_weight=1e-4):
            """
            Tính L_total = L_BPR + λ · L_CL cho SimGCL.

            Quy trình SimGCL:
                1. Propagate LightGCN trên đồ thị GỐC (1 lần duy nhất)
                2. Tạo View 1: thêm noise vào embedding stack
                3. Tạo View 2: thêm noise khác vào embedding stack
                4. Tính InfoNCE trên User embeddings
                5. Tính BPR trên embedding gốc (không có noise)
            """
            # Propagate 1 lần trên đồ thị gốc
            raw_stack = self._propagate_raw(edge_index)
            user_emb, item_emb = self._get_final_emb(raw_stack)

            # Tạo 2 views bằng noise
            stack1 = self._add_noise_view(raw_stack)
            stack2 = self._add_noise_view(raw_stack)
            z1_user, z1_item = self._get_final_emb(stack1)
            z2_user, z2_item = self._get_final_emb(stack2)

            # InfoNCE Loss (trên User và Item)
            loss_cl_user = (self._info_nce_loss(z1_user, z2_user) +
                            self._info_nce_loss(z2_user, z1_user)) / 2.0
            loss_cl_item = (self._info_nce_loss(z1_item, z2_item) +
                            self._info_nce_loss(z2_item, z1_item)) / 2.0
            loss_cl = (loss_cl_user + loss_cl_item) / 2.0

            # BPR Loss
            u_e  = user_emb[users]
            pi_e = item_emb[pos_items]
            ni_e = item_emb[neg_items]
            bpr = -torch.log(
                torch.sigmoid((u_e * pi_e).sum(-1) - (u_e * ni_e).sum(-1)) + 1e-8
            ).mean()
            u_e0  = self.embedding(users)
            pi_e0 = self.embedding(self.num_users + pos_items)
            ni_e0 = self.embedding(self.num_users + neg_items)
            reg = reg_weight * (
                u_e0.norm(2).pow(2) + pi_e0.norm(2).pow(2) + ni_e0.norm(2).pow(2)
            ) / users.size(0)
            loss_bpr = bpr + reg

            loss_total = loss_bpr + self.lambda_cl * loss_cl
            return loss_total, loss_bpr.item(), loss_cl.item()

        def predict_score_matrix(self, edge_index):
            """Dự đoán ma trận điểm (dùng embedding gốc, không noise)."""
            with torch.no_grad():
                raw_stack = self._propagate_raw(edge_index)
                user_emb, item_emb = self._get_final_emb(raw_stack)
                scores = torch.matmul(user_emb, item_emb.t())
                return scores.cpu().numpy().tolist()


# ============================================================
# FALLBACK — Khi chưa cài PyTorch
# ============================================================

else:
    class LightGCN_GCL:
        """Fallback class khi PyTorch chưa được cài đặt."""
        def __init__(self, num_users, num_items, embedding_dim=32, K=2, **kwargs):
            self.num_users     = num_users
            self.num_items     = num_items
            self.embedding_dim = embedding_dim
            self.user_emb = [[random.gauss(0, 0.1) for _ in range(embedding_dim)]
                              for _ in range(num_users)]
            self.item_emb = [[random.gauss(0, 0.1) for _ in range(embedding_dim)]
                              for _ in range(num_items)]

        def predict_score_matrix(self, edge_index=None):
            d = self.embedding_dim
            return [
                [sum(self.user_emb[u][k] * self.item_emb[i][k] for k in range(d))
                 for i in range(self.num_items)]
                for u in range(self.num_users)
            ]

    class SimGCL(LightGCN_GCL):
        """Fallback SimGCL."""
        pass


# ============================================================
# VÒNG LẶP HUẤN LUYỆN — GNN + GCL
# ============================================================

def train_gcl_model(
    model,
    train_user_items,
    train_pairs,
    edge_index_tensor,
    num_items,
    epochs=20,
    batch_size=2048,
    lr=0.01,
    reg_weight=1e-4,
    model_name="GNN+GCL"
):
    """
    Vòng lặp huấn luyện cho mô hình GNN+GCL (hoặc SimGCL).

    Args:
        model            : Instance của LightGCN_GCL hoặc SimGCL.
        train_user_items (dict): {user_id: set(item_ids)} — tập Train.
        train_pairs      (list): [(user, item), ...] — cặp tương tác Train.
        edge_index_tensor (Tensor): Đồ thị lưỡng phân [2, 2|E|].
        num_items        (int) : Số lượng Item N.
        epochs           (int) : Số epoch huấn luyện. Mặc định 20.
        batch_size       (int) : Kích thước batch. Mặc định 2048.
        lr               (float): Learning rate. Mặc định 0.01.
        reg_weight       (float): L2 regularization. Mặc định 1e-4.
        model_name       (str) : Tên hiển thị log.

    Returns:
        list[dict]: Lịch sử loss theo từng epoch
                    [{'epoch', 'total_loss', 'bpr_loss', 'cl_loss'}, ...]
    """
    if not HAS_TORCH:
        print(f"  [WARNING] PyTorch chưa được cài đặt. Bỏ qua huấn luyện {model_name}.")
        return []

    import torch

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = []

    print(f"\n[GCL] Bắt đầu huấn luyện {model_name}...")
    print(f"      Config: epochs={epochs}, batch={batch_size}, lr={lr}, "
          f"λ_cl={model.lambda_cl}, τ={model.temperature}")

    for epoch in range(1, epochs + 1):
        model.train()

        # Negative Sampling
        users_list, pos_list, neg_list = [], [], []
        for u, i in train_pairs:
            users_list.append(u)
            pos_list.append(i)
            liked = train_user_items.get(u, set())
            j = random.randint(0, num_items - 1)
            while j in liked:
                j = random.randint(0, num_items - 1)
            neg_list.append(j)

        # Mini-batch training
        total_loss_sum = 0.0
        bpr_loss_sum   = 0.0
        cl_loss_sum    = 0.0
        n_batches      = 0

        indices = list(range(len(users_list)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            u_b  = torch.tensor([users_list[k] for k in batch_idx],
                                 dtype=torch.long, device=device)
            pi_b = torch.tensor([pos_list[k]   for k in batch_idx],
                                 dtype=torch.long, device=device)
            ni_b = torch.tensor([neg_list[k]   for k in batch_idx],
                                 dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss, bpr_val, cl_val = model.compute_total_loss(
                u_b, pi_b, ni_b, edge_index_tensor, reg_weight=reg_weight
            )
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            bpr_loss_sum   += bpr_val
            cl_loss_sum    += cl_val
            n_batches      += 1

        avg_total = total_loss_sum / n_batches
        avg_bpr   = bpr_loss_sum   / n_batches
        avg_cl    = cl_loss_sum    / n_batches

        history.append({
            'epoch'     : epoch,
            'total_loss': avg_total,
            'bpr_loss'  : avg_bpr,
            'cl_loss'   : avg_cl
        })

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  * Epoch {epoch:02d}/{epochs:02d} | "
                  f"L_total={avg_total:.4f} | "
                  f"L_BPR={avg_bpr:.4f} | "
                  f"L_CL={avg_cl:.4f}")

    print(f"  [OK] Huấn luyện {model_name} hoàn tất!")
    return history
