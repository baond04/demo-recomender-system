"""
src/GP6_model_rag.py
Giải pháp 6: Vanilla RAG (Gợi ý dựa trên truy xuất tăng cường ngữ cảnh văn bản thuần túy từ CSDL Item).
Không sử dụng mạng nơ-ron đồ thị cá nhân hóa (Đóng vai trò mô hình đối chứng cốt lõi trong Ablation Study).
Đánh giá trên toàn bộ 11 độ đo Top-K.
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoaderMovieLens
from metrics import evaluate_model

class VanillaRAGRecommender:
    """
    Mô hình RAG thuần túy (Vanilla RAG Recommender):
    Truy xuất Top-K sản phẩm chỉ dựa trên độ tương đồng ngữ nghĩa văn bản (Dense text vectors/TF-IDF)
    của CSDL Item kết hợp với lịch sử truy vấn/từ khóa chung, không sử dụng đồ thị học tương phản (GNN+GCL).
    """
    def __init__(self, item_features, num_users, num_items):
        self.item_features = item_features # (num_items, d)
        self.num_users = num_users
        self.num_items = num_items
        self.score_matrix = None

    def fit(self, train_user_items):
        self.score_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        
        # Vector đặc trưng toàn cục (Global semantic prior)
        global_prior = np.mean(self.item_features, axis=0, keepdims=True)
        global_scores = np.dot(self.item_features, global_prior.T).flatten()
        
        for u in range(self.num_users):
            items = list(train_user_items.get(u, []))
            if items:
                u_vec = np.mean(self.item_features[items], axis=0, keepdims=True)
                semantic_sim = np.dot(self.item_features, u_vec.T).flatten()
                self.score_matrix[u] = 0.7 * semantic_sim + 0.3 * global_scores
            else:
                self.score_matrix[u] = global_scores

    def predict_score_matrix(self):
        return self.score_matrix

def run_gp6(data=None, dataset_name="MovieLens 100K", k_list=[10, 20]):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data is None:
        print(f"\n[GP6] Đang nạp dữ liệu {dataset_name}...")
        data = DataLoaderMovieLens(os.path.join(project_dir, "ml-latest-small")).prepare_data()

    print(f"\n{'='*80}")
    print(f"  GIẢI PHÁP 6 (GP6): VANILLA RAG (TRUY XUẤT CSDL VĂN BẢN THUẦN TÚY)")
    print(f"  Dataset: {dataset_name} | Users={data['num_users']}, Items={data['num_items']}")
    print(f"{'='*80}")

    num_users = data["num_users"]
    num_items = data["num_items"]
    train_user_items = data["train_user_items"]
    test_user_items = data["test_user_items"]
    tfidf_features = data["tfidf_features"]

    print("  -> Đang truy xuất tương đồng ngữ nghĩa qua CSDL vector Item...")
    model_rag = VanillaRAGRecommender(tfidf_features, num_users, num_items)
    model_rag.fit(train_user_items)
    score_matrix = model_rag.predict_score_matrix()

    print("  -> Đang đánh giá trên 11 độ đo Top-K...")
    results = evaluate_model(
        score_matrix, train_user_items, test_user_items, num_items,
        k_list=k_list, item_features=tfidf_features
    )

    for k in k_list:
        print(f"\n  --- KẾT QUẢ VANILLA RAG (TRUY XUẤT CSDL VĂN BẢN) — TOP-{k} ---")
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
    run_gp6()
