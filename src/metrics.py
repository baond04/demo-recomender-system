"""
metrics.py
Module 6: Thư viện tính toán các độ đo đánh giá tiêu chuẩn Top-K.
Bao gồm: Precision@K, Recall@K, NDCG@K, MRR@K, HitRate@K và Coverage.
Áp dụng cơ chế Masking (che các món đã tương tác ở tập Train bằng -∞).
"""

import math

def precision_at_k(actual_set, predicted_list, k):
    """Precision@K: Tỷ lệ đoán trúng trong K sản phẩm gợi ý"""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    hits = sum(1 for i in top_k if i in actual_set)
    return hits / float(k)

def recall_at_k(actual_set, predicted_list, k):
    """Recall@K: Tỷ lệ bao phủ các sản phẩm thực tế mà User thích"""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    hits = sum(1 for i in top_k if i in actual_set)
    return hits / float(len(actual_set))

def ndcg_at_k(actual_set, predicted_list, k):
    """NDCG@K: Đánh giá chất lượng thứ hạng (vị trí trúng càng cao điểm càng tốt)"""
    if not actual_set or k == 0:
        return 0.0
    
    top_k = predicted_list[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(top_k):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank_idx + 2) # rank_idx 0 -> log2(2)
            
    # IDCG (Ideal DCG)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(actual_set), k)))
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(actual_set, predicted_list, k):
    """MRR@K (Mean Reciprocal Rank): Nghịch đảo vị trí trúng đầu tiên"""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    for rank_idx, item in enumerate(top_k):
        if item in actual_set:
            return 1.0 / (rank_idx + 1)
    return 0.0

def hit_rate_at_k(actual_set, predicted_list, k):
    """HitRate@K: Trả về 1.0 nếu User nhận được ít nhất 1 sản phẩm đúng trong Top-K, ngược lại 0.0"""
    if not actual_set or k == 0:
        return 0.0
    top_k = predicted_list[:k]
    for item in top_k:
        if item in actual_set:
            return 1.0
    return 0.0

def evaluate_model(score_matrix, train_user_items, test_user_items, num_total_items, k_list=[10, 20]):
    """
    Hàm đánh giá tổng thể ma trận điểm dự đoán trên toàn bộ tập Test Users.
    Thực hiện Masking che các món đã tương tác ở tập Train trước khi xếp hạng Top-K.
    """
    results = {}
    num_users = len(score_matrix)
    
    # Chuẩn bị chứa danh sách Top-K cho từng User để tính Coverage
    recommended_items_all_users = {k: set() for k in k_list}
    
    metrics_sum = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "hit_rate": 0.0} for k in k_list}
    eval_user_count = 0
    
    import numpy as np
    max_k = max(k_list) if k_list else 20

    for u in range(num_users):
        actual_test_items = test_user_items.get(u, set())
        if not actual_test_items:
            continue # Bỏ qua nếu User không có dữ liệu ở tập Test
            
        eval_user_count += 1
        
        # 1. Cơ chế Masking: Che các sản phẩm đã tương tác trong tập Train bằng -infinity
        train_items = list(train_user_items.get(u, set()))
        if isinstance(score_matrix, np.ndarray):
            u_scores = score_matrix[u].copy()
        else:
            u_scores = np.array(score_matrix[u], dtype=np.float32)

        if train_items:
            u_scores[train_items] = -np.inf
            
        # 2. Lấy danh sách Top-K bằng np.argpartition (nhanh gấp 50 lần sort toàn bộ list)
        if len(u_scores) > max_k:
            top_part = np.argpartition(-u_scores, max_k)[:max_k]
            top_sorted = top_part[np.argsort(-u_scores[top_part])]
        else:
            top_sorted = np.argsort(-u_scores)
            
        predicted_rank_list = [int(item_id) for item_id in top_sorted if u_scores[item_id] != -np.inf]
        
        # 3. Tính toán các độ đo tại các ngưỡng K
        for k in k_list:
            top_k_list = predicted_rank_list[:k]
            metrics_sum[k]["precision"] += precision_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["recall"] += recall_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["ndcg"] += ndcg_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["mrr"] += mrr_at_k(actual_test_items, top_k_list, k)
            metrics_sum[k]["hit_rate"] += hit_rate_at_k(actual_test_items, top_k_list, k)
            
            # Ghi nhận các sản phẩm được mang đi gợi ý
            recommended_items_all_users[k].update(top_k_list)

    if eval_user_count == 0:
        return results

    # Averaging metrics across test users
    for k in k_list:
        results[f"Precision@{k}"] = metrics_sum[k]["precision"] / eval_user_count
        results[f"Recall@{k}"] = metrics_sum[k]["recall"] / eval_user_count
        results[f"NDCG@{k}"] = metrics_sum[k]["ndcg"] / eval_user_count
        results[f"MRR@{k}"] = metrics_sum[k]["mrr"] / eval_user_count
        results[f"HitRate@{k}"] = metrics_sum[k]["hit_rate"] / eval_user_count
        # Coverage = |Unique Recommended Items| / Total Catalog Items
        results[f"Coverage@{k}"] = len(recommended_items_all_users[k]) / float(num_total_items)

    return results
