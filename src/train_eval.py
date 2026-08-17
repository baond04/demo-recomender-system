"""
train_eval_all_datasets.py
Module điều phối chạy thực nghiệm và xuất bảng so sánh 4 mô hình (CB, MF, Hybrid, LightGCN)
trên cả 3 bộ dữ liệu: MovieLens 100K, Last.fm và Amazon Musical Instruments.
"""

import sys
import os
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoaderMovieLens, DataLoaderLastFM, DataLoaderAmazon

from model_cb import ContentBasedRecommender
from model_mf import MatrixFactorization, HAS_TORCH
from model_hybrid import HybridRecommender
from model_lightgcn import LightGCNRecommender
from metrics import evaluate_model

if HAS_TORCH:
    import torch

def sample_negative_items(train_user_items, num_items, train_pairs):
    users, pos_items, neg_items = [], [], []
    for u, i in train_pairs:
        users.append(u)
        pos_items.append(i)
        u_liked = train_user_items.get(u, set())
        neg = random.randint(0, num_items - 1)
        while neg in u_liked:
            neg = random.randint(0, num_items - 1)
        neg_items.append(neg)
    return users, pos_items, neg_items

def run_experiment_on_dataset(dataset_name, data):
    print(f"\n" + "=" * 80)
    print(f"      ĐANG CHẠY THỰC NGHIỆM TRÊN BỘ DỮ LIỆU: {dataset_name.upper()}      ")
    print("=" * 80)

    num_users = data["num_users"]
    num_items = data["num_items"]
    train_user_items = data["train_user_items"]
    test_user_items = data["test_user_items"]
    train_pairs = data["train_pairs"]
    edge_index = data["edge_index"]
    tfidf_features = data["tfidf_features"]

    results = {}

    # 1. CB
    print("  [1/4] Content-Based...")
    model_cb = ContentBasedRecommender(tfidf_features, num_users, num_items)
    model_cb.fit(train_user_items)
    score_cb = model_cb.predict_score_matrix()
    results["Content-Based"] = evaluate_model(score_cb, train_user_items, test_user_items, num_items, k_list=[10, 20])

    # 2. MF
    print("  [2/4] Matrix Factorization (MF)...", flush=True)
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=32).to(device)
        optimizer = torch.optim.Adam(model_mf.parameters(), lr=0.01)
        epochs = 50
        
        for epoch in range(1, epochs + 1):
            model_mf.train()
            u_l, p_l, n_l = sample_negative_items(train_user_items, num_items, train_pairs)
            total_loss = 0.0
            n_batches = 0
            for start_idx in range(0, len(u_l), 2048):
                end_idx = min(start_idx + 2048, len(u_l))
                u_b = torch.tensor(u_l[start_idx:end_idx], dtype=torch.long, device=device)
                p_b = torch.tensor(p_l[start_idx:end_idx], dtype=torch.long, device=device)
                n_b = torch.tensor(n_l[start_idx:end_idx], dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = model_mf.compute_bpr_loss(u_b, p_b, n_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                
            if epoch % 5 == 0 or epoch == epochs:
                print(f"    * [{dataset_name}] MF Epoch {epoch:02d}/{epochs:02d} | BPR Loss: {total_loss / n_batches:.4f}", flush=True)
                
        model_mf.eval()
        score_mf = model_mf.predict_score_matrix()
    else:
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=32)
        score_mf = model_mf.predict_score_matrix()
        
    results["Matrix Factorization"] = evaluate_model(score_mf, train_user_items, test_user_items, num_items, k_list=[10, 20])

    # 3. Hybrid
    print("  [3/4] Hybrid (MF + CB)...", flush=True)
    model_hybrid = HybridRecommender(alpha=0.6)
    score_hybrid = model_hybrid.predict_score_matrix(score_mf, score_cb)
    results["Hybrid (MF + CB)"] = evaluate_model(score_hybrid, train_user_items, test_user_items, num_items, k_list=[10, 20])

    # 4. LightGCN
    print("  [4/4] LightGCN (GNN)...", flush=True)
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=32, K=2).to(device)
        optimizer = torch.optim.Adam(model_lgcn.parameters(), lr=0.01)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
        epochs = 50
        
        for epoch in range(1, epochs + 1):
            model_lgcn.train()
            u_l, p_l, n_l = sample_negative_items(train_user_items, num_items, train_pairs)
            total_loss = 0.0
            n_batches = 0
            for start_idx in range(0, len(u_l), 2048):
                end_idx = min(start_idx + 2048, len(u_l))
                u_b = torch.tensor(u_l[start_idx:end_idx], dtype=torch.long, device=device)
                p_b = torch.tensor(p_l[start_idx:end_idx], dtype=torch.long, device=device)
                n_b = torch.tensor(n_l[start_idx:end_idx], dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = model_lgcn.compute_bpr_loss(u_b, p_b, n_b, edge_index_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                
            if epoch % 5 == 0 or epoch == epochs:
                print(f"    * [{dataset_name}] LightGCN Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss / n_batches:.4f}", flush=True)
                
        model_lgcn.eval()
        score_lgcn = model_lgcn.predict_score_matrix(edge_index_tensor)
    else:
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=32, K=2)
        score_lgcn = model_lgcn.predict_score_matrix()
        
    results["LightGCN (GNN)"] = evaluate_model(score_lgcn, train_user_items, test_user_items, num_items, k_list=[10, 20])

    for k in [10, 20]:
        print(f"\n  ---> BẢNG KẾT QUẢ CHO BỘ DỮ LIỆU {dataset_name.upper()} (Top-{k} Metrics):", flush=True)
        print(f"  {'Mô Hình':<22} | {f'Precision@{k}':<12} | {f'Recall@{k}':<12} | {f'NDCG@{k}':<12} | {f'MRR@{k}':<12} | {f'HitRate@{k}':<12} | {f'Coverage@{k}':<12}", flush=True)
        print("  " + "-" * 102, flush=True)
        for m_name, res in results.items():
            pk = res.get(f"Precision@{k}", 0.0)
            rk = res.get(f"Recall@{k}", 0.0)
            nk = res.get(f"NDCG@{k}", 0.0)
            mk = res.get(f"MRR@{k}", 0.0)
            hk = res.get(f"HitRate@{k}", 0.0)
            ck = res.get(f"Coverage@{k}", 0.0)
            print(f"  {m_name:<22} | {pk:<12.4f} | {rk:<12.4f} | {nk:<12.4f} | {mk:<12.4f} | {hk:<12.4f} | {ck:<12.4f}", flush=True)
        print("  " + "-" * 102, flush=True)
    print("", flush=True)

    return results

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    datasets = {
        "MovieLens 100K": DataLoaderMovieLens(os.path.join(project_dir, "ml-latest-small")).prepare_data(),
        "Last.fm": DataLoaderLastFM(os.path.join(project_dir, "hetrec2011-lastfm-2k")).prepare_data(),
        "Amazon Instruments": DataLoaderAmazon(os.path.join(project_dir, "reviews_Musical_Instruments_5.json")).prepare_data()
    }

    all_dataset_results = {}
    for name, data in datasets.items():
        all_dataset_results[name] = run_experiment_on_dataset(name, data)

    for k in [10, 20]:
        print("\n" + "=" * 95)
        print(f"      BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐA BỘ DỮ LIỆU (TOP-{k} METRICS)      ")
        print("=" * 95)
        print(f"{'Bộ Dữ Liệu':<20} | {'Mô Hình':<22} | {f'Recall@{k}':<11} | {f'NDCG@{k}':<11} | {f'MRR@{k}':<11} | {f'Coverage@{k}':<11}")
        print("-" * 95)

        for ds_name, res_dict in all_dataset_results.items():
            for m_name, res in res_dict.items():
                rk = res.get(f"Recall@{k}", 0.0)
                nk = res.get(f"NDCG@{k}", 0.0)
                mk = res.get(f"MRR@{k}", 0.0)
                ck = res.get(f"Coverage@{k}", 0.0)
                print(f"{ds_name:<20} | {m_name:<22} | {rk:<11.4f} | {nk:<11.4f} | {mk:<11.4f} | {ck:<11.4f}")
            print("-" * 95)

if __name__ == "__main__":
    main()
