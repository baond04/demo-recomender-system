"""
run_all_experiments.py
Chạy toàn bộ 6 mô hình trên 3 bộ dữ liệu và lưu kết quả thực tế.
Models: CB, MF, Hybrid, LightGCN, LightGCN+GCL(SGL), SimGCL
Datasets: MovieLens, Amazon Musical Instruments, Last.fm
"""

import sys
import os
import random
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_loader import DataLoaderMovieLens, DataLoaderLastFM, DataLoaderAmazon
from model_cb import ContentBasedRecommender
from model_mf import MatrixFactorization, HAS_TORCH
from model_hybrid import HybridRecommender
from model_lightgcn import LightGCNRecommender
from model_lightgcn_gcl import LightGCN_GCL, SimGCL, train_gcl_model
from metrics import evaluate_model

if HAS_TORCH:
    import torch

EPOCHS_BASE = 30     # CB, MF, Hybrid, LightGCN
EPOCHS_GCL  = 15     # GCL, SimGCL (nặng hơn, CPU chậm)
BATCH_SIZE  = 4096
EMBED_DIM   = 32
K_LAYERS    = 2
LR          = 0.01
LAMBDA_CL   = 0.2
TEMPERATURE = 0.2
K_LIST      = [10, 20]

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


def run_all_models(dataset_name, data):
    print(f"\n{'='*80}")
    print(f"  DATASET: {dataset_name}")
    print(f"  Users={data['num_users']}, Items={data['num_items']}, Interactions={len(data['train_pairs'])}")
    print(f"{'='*80}")

    num_users       = data["num_users"]
    num_items       = data["num_items"]
    train_user_items = data["train_user_items"]
    test_user_items  = data["test_user_items"]
    train_pairs      = data["train_pairs"]
    edge_index       = data["edge_index"]
    tfidf_features   = data["tfidf_features"]

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None

    # ──────────────────────────────
    # 1. Content-Based Filtering
    # ──────────────────────────────
    print(f"\n  [1/6] Content-Based Filtering...", flush=True)
    model_cb = ContentBasedRecommender(tfidf_features, num_users, num_items)
    model_cb.fit(train_user_items)
    score_cb = model_cb.predict_score_matrix()
    results["Content-Based"] = evaluate_model(score_cb, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['Content-Based'].get('Recall@20',0):.4f}  NDCG@20={results['Content-Based'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # 2. Matrix Factorization
    # ──────────────────────────────
    print(f"\n  [2/6] Matrix Factorization (BPR)...", flush=True)
    if HAS_TORCH:
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=EMBED_DIM).to(device)
        opt_mf   = torch.optim.Adam(model_mf.parameters(), lr=LR)
        for epoch in range(1, EPOCHS_BASE + 1):
            model_mf.train()
            u_l, p_l, n_l = sample_negative_items(train_user_items, num_items, train_pairs)
            total_loss, n_batches = 0.0, 0
            for s in range(0, len(u_l), BATCH_SIZE):
                e = min(s + BATCH_SIZE, len(u_l))
                ub = torch.tensor(u_l[s:e], dtype=torch.long, device=device)
                pb = torch.tensor(p_l[s:e], dtype=torch.long, device=device)
                nb = torch.tensor(n_l[s:e], dtype=torch.long, device=device)
                opt_mf.zero_grad()
                loss = model_mf.compute_bpr_loss(ub, pb, nb)
                loss.backward(); opt_mf.step()
                total_loss += loss.item(); n_batches += 1
            if epoch % 5 == 0 or epoch == EPOCHS_BASE:
                print(f"      Epoch {epoch:02d}/{EPOCHS_BASE} | BPR Loss={total_loss/n_batches:.4f}", flush=True)
        model_mf.eval()
        score_mf = model_mf.predict_score_matrix()
    else:
        model_mf = MatrixFactorization(num_users, num_items, embedding_dim=EMBED_DIM)
        score_mf = model_mf.predict_score_matrix()
    results["Matrix Factorization"] = evaluate_model(score_mf, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['Matrix Factorization'].get('Recall@20',0):.4f}  NDCG@20={results['Matrix Factorization'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # 3. Hybrid
    # ──────────────────────────────
    print(f"\n  [3/6] Hybrid (MF + CB, α=0.6)...", flush=True)
    model_hybrid = HybridRecommender(alpha=0.6)
    score_hybrid = model_hybrid.predict_score_matrix(score_mf, score_cb)
    results["Hybrid (MF+CB)"] = evaluate_model(score_hybrid, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['Hybrid (MF+CB)'].get('Recall@20',0):.4f}  NDCG@20={results['Hybrid (MF+CB)'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # 4. LightGCN Baseline
    # ──────────────────────────────
    print(f"\n  [4/6] LightGCN (K={K_LAYERS} layers)...", flush=True)
    if HAS_TORCH:
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=EMBED_DIM, K=K_LAYERS).to(device)
        opt_lgcn   = torch.optim.Adam(model_lgcn.parameters(), lr=LR)
        ei_tensor  = torch.tensor(edge_index, dtype=torch.long, device=device)
        for epoch in range(1, EPOCHS_BASE + 1):
            model_lgcn.train()
            u_l, p_l, n_l = sample_negative_items(train_user_items, num_items, train_pairs)
            total_loss, n_batches = 0.0, 0
            for s in range(0, len(u_l), BATCH_SIZE):
                e = min(s + BATCH_SIZE, len(u_l))
                ub = torch.tensor(u_l[s:e], dtype=torch.long, device=device)
                pb = torch.tensor(p_l[s:e], dtype=torch.long, device=device)
                nb = torch.tensor(n_l[s:e], dtype=torch.long, device=device)
                opt_lgcn.zero_grad()
                loss = model_lgcn.compute_bpr_loss(ub, pb, nb, ei_tensor)
                loss.backward(); opt_lgcn.step()
                total_loss += loss.item(); n_batches += 1
            if epoch % 5 == 0 or epoch == EPOCHS_BASE:
                print(f"      Epoch {epoch:02d}/{EPOCHS_BASE} | BPR Loss={total_loss/n_batches:.4f}", flush=True)
        model_lgcn.eval()
        score_lgcn = model_lgcn.predict_score_matrix(ei_tensor)
    else:
        model_lgcn = LightGCNRecommender(num_users, num_items, embedding_dim=EMBED_DIM, K=K_LAYERS)
        score_lgcn = model_lgcn.predict_score_matrix()
    results["LightGCN"] = evaluate_model(score_lgcn, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['LightGCN'].get('Recall@20',0):.4f}  NDCG@20={results['LightGCN'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # 5. LightGCN + GCL (SGL-style)
    # ──────────────────────────────
    print(f"\n  [5/6] LightGCN + GCL (SGL: Edge+Node Dropout)...", flush=True)
    if HAS_TORCH:
        model_gcl = LightGCN_GCL(
            num_users, num_items,
            embedding_dim=EMBED_DIM, K=K_LAYERS,
            aug_type='both',
            edge_drop_rate=0.2, node_drop_rate=0.1,
            temperature=TEMPERATURE, lambda_cl=LAMBDA_CL
        ).to(device)
        train_gcl_model(model_gcl, train_user_items, train_pairs, ei_tensor,
                        num_items, epochs=EPOCHS_GCL, batch_size=BATCH_SIZE, lr=LR,
                        model_name="GCL-SGL")
        model_gcl.eval()
        score_gcl = model_gcl.predict_score_matrix(ei_tensor)
    else:
        model_gcl = LightGCN_GCL(num_users, num_items, embedding_dim=EMBED_DIM)
        score_gcl = model_gcl.predict_score_matrix()
    results["LightGCN+GCL (SGL)"] = evaluate_model(score_gcl, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['LightGCN+GCL (SGL)'].get('Recall@20',0):.4f}  NDCG@20={results['LightGCN+GCL (SGL)'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # 6. SimGCL (Noise Perturbation)
    # ──────────────────────────────
    print(f"\n  [6/6] SimGCL (Noise Perturbation, ε=0.1)...", flush=True)
    if HAS_TORCH:
        model_simgcl = SimGCL(
            num_users, num_items,
            embedding_dim=EMBED_DIM, K=K_LAYERS,
            epsilon=0.1, temperature=TEMPERATURE, lambda_cl=LAMBDA_CL
        ).to(device)
        train_gcl_model(model_simgcl, train_user_items, train_pairs, ei_tensor,
                        num_items, epochs=EPOCHS_GCL, batch_size=BATCH_SIZE, lr=LR,
                        model_name="SimGCL")
        model_simgcl.eval()
        score_simgcl = model_simgcl.predict_score_matrix(ei_tensor)
    else:
        model_simgcl = SimGCL(num_users, num_items, embedding_dim=EMBED_DIM)
        score_simgcl = model_simgcl.predict_score_matrix()
    results["SimGCL"] = evaluate_model(score_simgcl, train_user_items, test_user_items, num_items, k_list=K_LIST)
    print(f"        Recall@20={results['SimGCL'].get('Recall@20',0):.4f}  NDCG@20={results['SimGCL'].get('NDCG@20',0):.4f}")

    # ──────────────────────────────
    # In bảng kết quả
    # ──────────────────────────────
    for k in K_LIST:
        print(f"\n  {'─'*100}")
        print(f"  KẾT QUẢ {dataset_name} — Top-{k}")
        print(f"  {'─'*100}")
        hdr = f"  {'Mô hình':<22} | {'Precision':<10} | {'Recall':<10} | {'NDCG':<10} | {'MRR':<10} | {'Coverage':<10}"
        print(hdr)
        print(f"  {'─'*100}")
        for mname, res in results.items():
            p = res.get(f"Precision@{k}", 0.0)
            r = res.get(f"Recall@{k}", 0.0)
            n = res.get(f"NDCG@{k}", 0.0)
            m = res.get(f"MRR@{k}", 0.0)
            c = res.get(f"Coverage@{k}", 0.0)
            print(f"  {mname:<22} | {p:<10.4f} | {r:<10.4f} | {n:<10.4f} | {m:<10.4f} | {c:<10.4f}")
        print(f"  {'─'*100}")

    return results


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(42)

    print("="*80)
    print("  CHẠY THỰC NGHIỆM TOÀN BỘ — GNN + GCL")
    print(f"  PyTorch: {'CÓ (' + ('GPU' if (HAS_TORCH and torch.cuda.is_available()) else 'CPU') + ')' if HAS_TORCH else 'KHÔNG (chạy numpy fallback)'}")
    print("="*80)

    datasets = {}

    print("\n[Loading] MovieLens ml-latest-small...", flush=True)
    try:
        datasets["MovieLens"] = DataLoaderMovieLens(
            os.path.join(project_dir, "ml-latest-small")
        ).prepare_data()
        print(f"  OK: {datasets['MovieLens']['num_users']} users, {datasets['MovieLens']['num_items']} items")
    except Exception as ex:
        print(f"  LỖI: {ex}")

    print("\n[Loading] Amazon Musical Instruments...", flush=True)
    try:
        datasets["Amazon Instruments"] = DataLoaderAmazon(
            os.path.join(project_dir, "reviews_Musical_Instruments_5.json")
        ).prepare_data()
        print(f"  OK: {datasets['Amazon Instruments']['num_users']} users, {datasets['Amazon Instruments']['num_items']} items")
    except Exception as ex:
        print(f"  LỖI: {ex}")

    print("\n[Loading] Last.fm HetRec 2011...", flush=True)
    try:
        datasets["Last.fm"] = DataLoaderLastFM(
            os.path.join(project_dir, "hetrec2011-lastfm-2k")
        ).prepare_data()
        print(f"  OK: {datasets['Last.fm']['num_users']} users, {datasets['Last.fm']['num_items']} items")
    except Exception as ex:
        print(f"  LỖI: {ex}")

    all_results = {}
    for ds_name, data in datasets.items():
        all_results[ds_name] = run_all_models(ds_name, data)

    # Lưu kết quả JSON
    out_path = os.path.join(project_dir, "experiment_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Kết quả đã lưu tại: {out_path}")

    # In bảng tổng hợp cuối
    print("\n" + "="*100)
    print("  BẢNG TỔNG HỢP — RECALL@20 & NDCG@20 TRÊN 3 BỘ DỮ LIỆU")
    print("="*100)
    print(f"  {'Bộ dữ liệu':<22} | {'Mô hình':<22} | {'Recall@20':<11} | {'NDCG@20':<11} | {'MRR@20':<11}")
    print("  " + "-"*82)
    for ds_name, res_dict in all_results.items():
        for mname, res in res_dict.items():
            r20 = res.get("Recall@20", 0.0)
            n20 = res.get("NDCG@20", 0.0)
            m20 = res.get("MRR@20", 0.0)
            print(f"  {ds_name:<22} | {mname:<22} | {r20:<11.4f} | {n20:<11.4f} | {m20:<11.4f}")
        print("  " + "-"*82)


if __name__ == "__main__":
    main()
