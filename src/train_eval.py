"""
train_eval.py
Module 7: Vòng lặp Huấn luyện & Đánh giá so sánh 4 mô hình Recommender System.
Điều phối chạy cả 4 mô hình (Content-Based, Matrix Factorization, Hybrid, LightGCN)
trên cùng bộ dữ liệu MovieLens, tính BPR Loss, Negative Sampling, Masking và xuất bảng kết quả.
"""

import sys
import os
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import các mô-đun trong src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoaderMovieLens
from model_cb import ContentBasedRecommender
from model_mf import MatrixFactorization, HAS_TORCH
from model_hybrid import HybridRecommender
from model_lightgcn import LightGCNRecommender
from metrics import evaluate_model

if HAS_TORCH:
    import torch

def sample_negative_items(train_user_items, num_users, num_items, train_pairs):
    """Hàm Negative Sampling ngẫu nhiên cho BPR Loss"""
    users = []
    pos_items = []
    neg_items = []
    
    for u, i in train_pairs:
        users.append(u)
        pos_items.append(i)
        
        # Rút ngẫu nhiên 1 sản phẩm âm chưa tương tác
        u_liked = train_user_items.get(u, set())
        neg = random.randint(0, num_items - 1)
        while neg in u_liked:
            neg = random.randint(0, num_items - 1)
        neg_items.append(neg)
        
    return users, pos_items, neg_items

def main():
    print("=" * 85)
    print("      CHUONG TRINH HUAN LUYEN & DANH GIA SO SANH 4 MO HINH RECSYS      ")
    print("=" * 85)

    # 1. Tải và tiền xử lý dữ liệu từ folder ml-latest-small
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml-latest-small")
    loader = DataLoaderMovieLens(dataset_dir=dataset_dir, min_rating=3.0, seed=42)
    data = loader.prepare_data()
    
    num_users = data["num_users"]
    num_items = data["num_items"]
    train_user_items = data["train_user_items"]
    test_user_items = data["test_user_items"]
    train_pairs = data["train_pairs"]
    edge_index = data["edge_index"]
    tfidf_features = data["tfidf_features"]

    all_results = {}

    # -------------------------------------------------------------
    # MÔ HÌNH 1: Content-Based Filtering (CB)
    # -------------------------------------------------------------
    print("\n[1/4] Dang huan luyen Mo hinh Content-Based Filtering...")
    model_cb = ContentBasedRecommender(tfidf_features, num_users, num_items)
    model_cb.fit(train_user_items)
    score_cb = model_cb.predict_score_matrix()
    all_results["Content-Based"] = evaluate_model(score_cb, train_user_items, test_user_items, num_items)
    print("  * Hoan thanh danh gia Content-Based!")

    # -------------------------------------------------------------
    # MÔ HÌNH 2: Matrix Factorization (MF)
    # -------------------------------------------------------------
    print("\n[2/4] Dang huan luyen Mo hinh Matrix Factorization (MF)...")
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=32).to(device)
        optimizer = torch.optim.Adam(model_mf.parameters(), lr=0.01)
        
        epochs = 10
        batch_size = 2048
        
        for epoch in range(1, epochs + 1):
            model_mf.train()
            users, pos_items, neg_items = sample_negative_items(train_user_items, num_users, num_items, train_pairs)
            
            total_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, len(users), batch_size):
                end_idx = min(start_idx + batch_size, len(users))
                u_b = torch.tensor(users[start_idx:end_idx], dtype=torch.long, device=device)
                p_b = torch.tensor(pos_items[start_idx:end_idx], dtype=torch.long, device=device)
                n_b = torch.tensor(neg_items[start_idx:end_idx], dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                loss = model_mf.compute_bpr_loss(u_b, p_b, n_b)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
            if epoch % 5 == 0 or epoch == epochs:
                print(f"  * Epoch {epoch:02d}/{epochs:02d} | BPR Loss: {total_loss / n_batches:.4f}")
                
        model_mf.eval()
        score_mf = model_mf.predict_score_matrix()
    else:
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=32)
        score_mf = model_mf.predict_score_matrix()
        
    all_results["Matrix Factorization"] = evaluate_model(score_mf, train_user_items, test_user_items, num_items)
    print("  * Hoan thanh danh gia Matrix Factorization!")

    # -------------------------------------------------------------
    # MÔ HÌNH 3: Hybrid Recommender (Late Fusion: MF + CB)
    # -------------------------------------------------------------
    print("\n[3/4] Dang huan luyen Mo hinh Lai (Hybrid Late Fusion alpha=0.6)...")
    model_hybrid = HybridRecommender(alpha=0.6)
    score_hybrid = model_hybrid.predict_score_matrix(score_mf, score_cb)
    all_results["Hybrid (MF + CB)"] = evaluate_model(score_hybrid, train_user_items, test_user_items, num_items)
    print("  * Hoan thanh danh gia Hybrid!")

    # -------------------------------------------------------------
    # MÔ HÌNH 4: LightGCN (Graph Neural Network tren Do thi Luong phan)
    # -------------------------------------------------------------
    print("\n[4/4] Dang huan luyen Mo hinh LightGCN (Graph Neural Network)...")
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=32, K=2).to(device)
        optimizer = torch.optim.Adam(model_lgcn.parameters(), lr=0.01)
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
        
        epochs = 10
        batch_size = 2048
        
        for epoch in range(1, epochs + 1):
            model_lgcn.train()
            users, pos_items, neg_items = sample_negative_items(train_user_items, num_users, num_items, train_pairs)
            
            total_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, len(users), batch_size):
                end_idx = min(start_idx + batch_size, len(users))
                u_b = torch.tensor(users[start_idx:end_idx], dtype=torch.long, device=device)
                p_b = torch.tensor(pos_items[start_idx:end_idx], dtype=torch.long, device=device)
                n_b = torch.tensor(neg_items[start_idx:end_idx], dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                loss = model_lgcn.compute_bpr_loss(u_b, p_b, n_b, edge_index_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
            if epoch % 5 == 0 or epoch == epochs:
                print(f"  * Epoch {epoch:02d}/{epochs:02d} | LightGCN Loss: {total_loss / n_batches:.4f}")
                
        model_lgcn.eval()
        score_lgcn = model_lgcn.predict_score_matrix(edge_index_tensor)
    else:
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=32, K=2)
        score_lgcn = model_lgcn.predict_score_matrix()
        
    all_results["LightGCN (GNN)"] = evaluate_model(score_lgcn, train_user_items, test_user_items, num_items)
    print("  * Hoan thanh danh gia LightGCN!")

    # -------------------------------------------------------------
    # XUẤT BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH TRÊN TẬP TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                    BANG TONG HOP KET QUA DANH GIA SO SANH (Top-20)                    ")
    print("=" * 90)
    print(f"{'Mo hinh':<25} | {'Precision@20':<12} | {'Recall@20':<12} | {'NDCG@20':<12} | {'MRR@20':<12} | {'Coverage@20':<12}")
    print("-" * 90)
    
    for model_name, res in all_results.items():
        p20 = res.get("Precision@20", 0.0)
        r20 = res.get("Recall@20", 0.0)
        n20 = res.get("NDCG@20", 0.0)
        m20 = res.get("MRR@20", 0.0)
        c20 = res.get("Coverage@20", 0.0)
        print(f"{model_name:<25} | {p20:<12.4f} | {r20:<12.4f} | {n20:<12.4f} | {m20:<12.4f} | {c20:<12.4f}")
        
    print("=" * 90)
    print("* Da hoan tat thuc nghiem va tinh toan ket qua quy chuan!")

if __name__ == "__main__":
    main()
