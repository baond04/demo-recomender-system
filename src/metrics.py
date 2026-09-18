"""
metrics.py
Module tính toán các độ đo đánh giá tiêu chuẩn Top-K cho Hệ gợi ý (Recommender Systems).
Bao gồm 11 độ đo:
  Phần 1 - Các độ đo theo yêu cầu của cô:
    1. Precision@K
    2. Recall@K
    3. NDCG@K
    4. MRR@K
    5. Coverage@K
    6. Improvement (%)
  Phần 2 - Các độ đo khảo sát bổ sung:
    7. HitRate@K
    8. F1-Score@K
    9. MAP@K
    10. Novelty@K
    11. Diversity@K (Intra-List Diversity)
"""

import math
from collections import defaultdict
import numpy as np

# =====================================================================
# PHẦN 1: CÁC ĐỘ ĐO THEO YÊU CẦU CỦA CÔ (GỐC)
# =====================================================================

def precision_at_k(actual_set, predicted_list, k):
    """1. Precision@K: Tỷ lệ đoán trúng trong K sản phẩm gợi ý."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    hits = sum(1 for i in top_k if i in actual_set)
    return hits / float(k)

def recall_at_k(actual_set, predicted_list, k):
    """2. Recall@K: Tỷ lệ tìm được sản phẩm thực tế mà User thích trên tập Test."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    hits = sum(1 for i in top_k if i in actual_set)
    return hits / float(len(actual_set))

def ndcg_at_k(actual_set, predicted_list, k):
    """3. NDCG@K: Đánh giá chất lượng thứ hạng (vị trí trúng càng cao điểm càng tốt)."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(top_k):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank_idx + 2) # rank_idx 0 -> log2(2)
            
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(actual_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(actual_set, predicted_list, k):
    """4. MRR@K (Mean Reciprocal Rank): Nghịch đảo vị trí trúng đầu tiên."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    for rank_idx, item in enumerate(top_k):
        if item in actual_set:
            return 1.0 / (rank_idx + 1)
    return 0.0

def calculate_improvement(base_val, target_val):
    """
    6. Improvement (%): Tỷ lệ phần trăm cải thiện hiệu năng của giải pháp mục tiêu so với baseline:
    Improvement (%) = ((Target - Base) / Base) * 100%
    """
    if base_val <= 0:
        return 0.0
    return ((target_val - base_val) / float(base_val)) * 100.0


# =====================================================================
# PHẦN 2: CÁC ĐỘ ĐO KHẢO SÁT BỔ SUNG (DÙNG ĐƯỢC CHO TẤT CẢ GIẢI PHÁP)
# =====================================================================

def hit_rate_at_k(actual_set, predicted_list, k):
    """7. HitRate@K: Trả về 1.0 nếu User nhận được ít nhất 1 sản phẩm đúng trong Top-K, ngược lại 0.0."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    for item in top_k:
        if item in actual_set:
            return 1.0
    return 0.0

def f1_score_at_k(precision, recall):
    """8. F1-Score@K: Trung bình điều hòa giữa Precision và Recall."""
    if (precision + recall) <= 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

def map_at_k(actual_set, predicted_list, k):
    """9. MAP@K (Mean Average Precision): Độ chính xác trung bình có trọng số theo thứ hạng."""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    hits = 0
    sum_prec = 0.0
    for rank_idx, item in enumerate(top_k):
        if item in actual_set:
            hits += 1
            sum_prec += hits / (rank_idx + 1.0)
    return sum_prec / float(min(len(actual_set), k))

def novelty_at_k(predicted_list, item_popularity, total_interactions, k):
    """10. Novelty@K: Đo khả năng gợi ý sản phẩm ngách/ít phổ biến (-log2 P(i))."""
    if k == 0 or not predicted_list or total_interactions <= 0:
        return 0.0
    top_k = predicted_list[:k]
    nov = 0.0
    for item in top_k:
        pop = item_popularity.get(item, 1)
        prob = max(pop, 1) / float(total_interactions)
        nov += -math.log2(prob)
    return nov / float(len(top_k))

def diversity_at_k(predicted_list, item_features, k):
    """11. Diversity@K (Intra-List Diversity): Đo khoảng cách thuộc tính giữa các item trong Top-K."""
    if k <= 1 or not predicted_list or item_features is None:
        return 1.0
    top_k = predicted_list[:k]
    n = len(top_k)
    if n <= 1:
        return 1.0
    
    try:
        vecs = item_features[top_k]
        sim_matrix = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
        dist_sum = (n * n - np.sum(sim_matrix)) / 2.0
        num_pairs = (n * (n - 1)) / 2.0
        return float(max(0.0, dist_sum / num_pairs)) if num_pairs > 0 else 1.0
    except Exception:
        return 1.0


# =====================================================================
# HÀM ĐÁNH GIÁ TỔNG HỢP CHO MA TRẬN ĐIỂM
# =====================================================================

def evaluate_model(score_matrix, train_user_items, test_user_items, num_total_items, 
                   k_list=[10, 20], item_features=None, item_popularity=None, total_interactions=None):
    """
    Hàm đánh giá tổng thể ma trận điểm dự đoán trên toàn bộ tập Test Users.
    Thực hiện Masking che các món đã tương tác ở tập Train trước khi xếp hạng Top-K.
    Trả về dictionary theo thứ tự chuẩn:
      - 6 độ đo cô yêu cầu: Precision, Recall, NDCG, MRR, Coverage
      - 5 độ đo bổ sung: HitRate, F1-Score, MAP, Novelty, Diversity
    """
    results = {}
    num_users = len(score_matrix)
    
    # Tính item popularity nếu chưa có
    if item_popularity is None:
        item_popularity = defaultdict(int)
        tot_cnt = 0
        for u, items in train_user_items.items():
            for i in items:
                item_popularity[i] += 1
                tot_cnt += 1
        total_interactions = tot_cnt if tot_cnt > 0 else 1
    elif total_interactions is None:
        total_interactions = sum(item_popularity.values()) or 1

    recommended_items_all_users = {k: set() for k in k_list}
    
    metrics_sum = {
        k: {
            "precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0,
            "hit_rate": 0.0, "map": 0.0, "novelty": 0.0, "diversity": 0.0
        } for k in k_list
    }
    eval_user_count = 0
    max_k = max(k_list) if k_list else 20

    for u in range(num_users):
        actual_test_items = test_user_items.get(u, set())
        if not actual_test_items:
            continue
            
        eval_user_count += 1
        
        # 1. Masking: Che các món đã tương tác trong tập Train
        train_items = list(train_user_items.get(u, set()))
        if isinstance(score_matrix, np.ndarray):
            u_scores = score_matrix[u].copy()
        else:
            u_scores = np.array(score_matrix[u], dtype=np.float32)

        if train_items:
            u_scores[train_items] = -np.inf
            
        # 2. Xếp hạng Top-K
        if len(u_scores) > max_k:
            top_part = np.argpartition(-u_scores, max_k)[:max_k]
            top_sorted = top_part[np.argsort(-u_scores[top_part])]
        else:
            top_sorted = np.argsort(-u_scores)
            
        predicted_rank_list = [int(item_id) for item_id in top_sorted if u_scores[item_id] != -np.inf]
        
        # 3. Tính toán các độ đo
        for k in k_list:
            top_k_list = predicted_rank_list[:k]
            metrics_sum[k]["precision"] += precision_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["recall"] += recall_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["ndcg"] += ndcg_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["mrr"] += mrr_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["hit_rate"] += hit_rate_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["map"] += map_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["novelty"] += novelty_at_k(top_k_list, item_popularity, total_interactions, k)
            metrics_sum[k]["diversity"] += diversity_at_k(top_k_list, item_features, k)
            
            recommended_items_all_users[k].update(top_k_list)

    if eval_user_count == 0:
        return results

    # Đưa kết quả vào dictionary theo thứ tự chuẩn hóa
    for k in k_list:
        p = metrics_sum[k]["precision"] / eval_user_count
        r = metrics_sum[k]["recall"] / eval_user_count
        
        # Nhóm cô yêu cầu
        results[f"Precision@{k}"] = p
        results[f"Recall@{k}"]    = r
        results[f"NDCG@{k}"]      = metrics_sum[k]["ndcg"] / eval_user_count
        results[f"MRR@{k}"]       = metrics_sum[k]["mrr"] / eval_user_count
        results[f"Coverage@{k}"]  = len(recommended_items_all_users[k]) / float(num_total_items)
        
        # Nhóm khảo sát bổ sung
        results[f"HitRate@{k}"]   = metrics_sum[k]["hit_rate"] / eval_user_count
        results[f"F1@{k}"]        = f1_score_at_k(p, r)
        results[f"MAP@{k}"]       = metrics_sum[k]["map"] / eval_user_count
        results[f"Novelty@{k}"]   = metrics_sum[k]["novelty"] / eval_user_count
        results[f"Diversity@{k}"] = metrics_sum[k]["diversity"] / eval_user_count

    return results


