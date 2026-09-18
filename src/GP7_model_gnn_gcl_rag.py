"""
src/GP7_model_gnn_gcl_rag.py
============================
GIẢI PHÁP 7 (GP7): GNN + GCL + RAG (GIẢI PHÁP ĐỀ XUẤT TOÀN DIỆN CỦA ĐỒ ÁN TỐT NGHIỆP)
Mô hình gợi ý cá nhân hóa đa phương thức kết hợp Mạng nơ-ron đồ thị (LightGCN), 
Học tương phản đồ thị (SimGCL) và Truy xuất tăng cường ngữ cảnh văn bản (RAG) từ CSDL Item.

Đánh giá toàn diện trên 11 độ đo Top-K:
  - Nhóm độ đo theo yêu cầu của Cô: Precision@K, Recall@K, NDCG@K, MRR@K, Coverage@K, Improvement (%)
  - Nhóm độ đo mở rộng bổ sung: HitRate@K, F1@K, MAP@K, Novelty@K, Diversity@K
"""

import os
import sys
import random
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoaderMovieLens
from GP5_model_lightgcn_gcl import SimGCL, HAS_TORCH
from GP6_model_rag import VanillaRAGRecommender
from metrics import evaluate_model

if HAS_TORCH:
    import torch


class GNN_GCL_RAG_Recommender:
    """
    Mô hình đề xuất cốt lõi: GNN + GCL + RAG (PCRS Recommender).
    Kết hợp:
      - Nhánh 1 (GNN + GCL - SimGCL): Nắm bắt tín hiệu cộng tác bậc cao và cấu trúc đồ thị tương tác người dùng - sản phẩm.
      - Nhánh 2 (RAG Retrieval): Nắm bắt độ tương đồng ngữ nghĩa văn bản phong phú từ CSDL Item.
      - Fusion Layer: Kết hợp điểm thích ứng (Adaptive Score Fusion) để tạo danh sách Top-K chính xác và đa dạng nhất.
    """
    def __init__(self, num_users, num_items, item_features, embedding_dim=32, k_layers=2, alpha=0.7):
        self.num_users = num_users
        self.num_items = num_items
        self.item_features = item_features
        self.alpha = alpha  # Trọng số kết hợp GNN+GCL vs RAG
        
        # Nhánh 1: GNN + GCL
        if HAS_TORCH:
            self.model_gcl = SimGCL(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                K=k_layers,
                epsilon=0.1,
                temperature=0.2,
                lambda_cl=0.2
            )
        else:
            self.model_gcl = SimGCL(num_users, num_items, embedding_dim=embedding_dim)

        # Nhánh 2: RAG Semantic Retrieval
        self.model_rag = VanillaRAGRecommender(item_features, num_users, num_items)

    def fit(self, train_user_items, train_pairs=None, edge_index=None, epochs=15, batch_size=2048, lr=0.01):
        # 1. Huấn luyện Nhánh RAG (Truy xuất CSDL Item)
        print("    [1/2] Đang khớp không gian biểu diễn ngữ nghĩa RAG...")
        self.model_rag.fit(train_user_items)

        # 2. Huấn luyện Nhánh GNN + GCL (SimGCL)
        print("    [2/2] Đang huấn luyện mô hình đồ thị tương phản SimGCL (GNN + GCL)...")
        if HAS_TORCH and edge_index is not None and train_pairs is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_gcl = self.model_gcl.to(device)
            from GP5_model_lightgcn_gcl import train_gcl_model
            train_gcl_model(
                self.model_gcl, train_user_items, train_pairs, edge_index,
                self.num_items, epochs=epochs, batch_size=batch_size, lr=lr,
                model_name="GNN+GCL+RAG (SimGCL Backbone)"
            )
            self.model_gcl.eval()
        else:
            print("          (Đang dùng biểu diễn đồ thị ma trận hoặc fallback)")

    def predict_score_matrix(self, edge_index=None):
        """
        Dự đoán ma trận điểm kết hợp (GNN + GCL + RAG)
        Score_PCRS = alpha * Score_SimGCL + (1 - alpha) * Score_RAG
        """
        # Điểm RAG
        score_rag = self.model_rag.predict_score_matrix()
        
        # Chuẩn hóa min-max RAG theo từng dòng
        rag_min = score_rag.min(axis=1, keepdims=True)
        rag_max = score_rag.max(axis=1, keepdims=True)
        rag_denom = np.where((rag_max - rag_min) == 0, 1.0, (rag_max - rag_min))
        norm_score_rag = (score_rag - rag_min) / rag_denom

        # Điểm SimGCL
        if HAS_TORCH and edge_index is not None:
            score_gcl = self.model_gcl.predict_score_matrix(edge_index)
        else:
            score_gcl = self.model_gcl.predict_score_matrix()

        score_gcl = np.asarray(score_gcl, dtype=np.float32)

        # Chuẩn hóa min-max GCL theo từng dòng
        gcl_min = score_gcl.min(axis=1, keepdims=True)
        gcl_max = score_gcl.max(axis=1, keepdims=True)
        gcl_denom = np.where((gcl_max - gcl_min) == 0, 1.0, (gcl_max - gcl_min))
        norm_score_gcl = (score_gcl - gcl_min) / gcl_denom

        # Hợp nhất đa tầng (Fusion Layer)
        final_score = self.alpha * norm_score_gcl + (1.0 - self.alpha) * norm_score_rag
        return final_score


def run_gp7(data=None, dataset_name="MovieLens 100K", k_list=[10, 20]):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data is None:
        print(f"\n[GP7] Đang nạp dữ liệu {dataset_name}...")
        data = DataLoaderMovieLens(os.path.join(project_dir, "ml-latest-small")).prepare_data()

    print(f"\n{'='*80}")
    print(f"  GIẢI PHÁP 7 (GP7): GNN + GCL + RAG (GIẢI PHÁP ĐỀ XUẤT CHÍNH CỦA ĐỒ ÁN)")
    print(f"  Dataset: {dataset_name} | Users={data['num_users']}, Items={data['num_items']}")
    print(f"{'='*80}")

    num_users = data["num_users"]
    num_items = data["num_items"]
    train_user_items = data["train_user_items"]
    test_user_items = data["test_user_items"]
    train_pairs = data["train_pairs"]
    edge_index = data.get("edge_index", None)
    tfidf_features = data["tfidf_features"]

    model = GNN_GCL_RAG_Recommender(
        num_users=num_users,
        num_items=num_items,
        item_features=tfidf_features,
        embedding_dim=32,
        k_layers=2,
        alpha=0.75
    )

    print("\n  -> Bắt đầu huấn luyện mô hình GNN + GCL + RAG...")
    model.fit(
        train_user_items=train_user_items,
        train_pairs=train_pairs,
        edge_index=edge_index,
        epochs=15,
        batch_size=2048,
        lr=0.01
    )

    print("\n  -> Đang tính toán điểm hợp nhất và đánh giá 11 độ đo Top-K...")
    score_matrix = model.predict_score_matrix(edge_index=edge_index)

    results = evaluate_model(
        score_matrix, train_user_items, test_user_items, num_items,
        k_list=k_list, item_features=tfidf_features
    )

    for k in k_list:
        print(f"\n  --- KẾT QUẢ GNN + GCL + RAG (GP7) — TOP-{k} ---")
        print(f"  [Đánh giá các độ đo]")
        print(f"    • Precision@{k:<2} : {results.get(f'Precision@{k}', 0.0):.4f}")
        print(f"    • Recall@{k:<2}    : {results.get(f'Recall@{k}', 0.0):.4f}")
        print(f"    • NDCG@{k:<2}      : {results.get(f'NDCG@{k}', 0.0):.4f}")
        print(f"    • MRR@{k:<2}       : {results.get(f'MRR@{k}', 0.0):.4f}")
        print(f"    • Coverage@{k:<2}  : {results.get(f'Coverage@{k}', 0.0):.4f}")
        print(f"  [Độ đo mở rộng bổ sung]")
        print(f"    • HitRate@{k:<2}   : {results.get(f'HitRate@{k}', 0.0):.4f}")
        print(f"    • F1-Score@{k:<2}  : {results.get(f'F1@{k}', 0.0):.4f}")
        print(f"    • MAP@{k:<2}       : {results.get(f'MAP@{k}', 0.0):.4f}")
        print(f"    • Novelty@{k:<2}   : {results.get(f'Novelty@{k}', 0.0):.4f}")
        print(f"    • Diversity@{k:<2} : {results.get(f'Diversity@{k}', 0.0):.4f}")

    return results, score_matrix


if __name__ == "__main__":
    run_gp7()
