"""
evaluate_pcrs.py
================
Đánh giá PCRS (Personalized Conversational Recommender System).

Metrics:
  1. GNN+GCL Metrics  : Recall@K, NDCG@K (từ offline evaluation)
  2. RAG Metrics       : Retrieval precision, recall
  3. Pipeline Metrics  : End-to-end latency, throughput
  4. Diversity         : Coverage, serendipity của gợi ý trong hội thoại

Cách dùng:
    python evaluate_pcrs.py --dataset movielens --top-k 10
    python evaluate_pcrs.py --latency-test --n-queries 50
"""

import sys
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# METRICS
# ============================================================

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall@K: Tỷ lệ items liên quan xuất hiện trong top-K gợi ý.

    Recall@K = |{i ∈ recommended[:K]} ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = Σ_{i=1}^{min(|R|,K)} 1 / log2(i+1)  (perfect ranking)
    NDCG@K = DCG@K / IDCG@K
    """
    import math
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended[:k], 1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: list[list], num_items: int) -> float:
    """
    Item Coverage: Tỷ lệ items được gợi ý ít nhất 1 lần.
    Coverage = |∪ recommendations| / num_items
    """
    all_items = set()
    for recs in all_recommendations:
        all_items.update(recs)
    return len(all_items) / num_items if num_items > 0 else 0.0


def diversity_intra(recommendations: list) -> float:
    """
    Intra-list Diversity: Đo độ đa dạng trong danh sách gợi ý của 1 user.
    Dựa trên tỷ lệ genres khác nhau.
    """
    # Placeholder: thực tế cần metadata genres
    return len(set(str(i)[-1] for i in recommendations)) / len(recommendations) if recommendations else 0.0


# ============================================================
# EVALUATION RUNNER
# ============================================================

def evaluate_gnn_gcl(adapter, test_interactions: dict, top_k: int = 10) -> dict:
    """
    Đánh giá GNN+GCL recommendation quality.

    Args:
        adapter          : GNNGCLAdapter.
        test_interactions: {user_id: set[item_id]} — ground truth test set.
        top_k            : K cho Recall@K và NDCG@K.

    Returns:
        dict: {'recall@k': float, 'ndcg@k': float, 'coverage': float}
    """
    logger.info(f"Evaluating GNN+GCL: {len(test_interactions)} test users, top_k={top_k}")

    recalls = []
    ndcgs   = []
    all_recs = []

    for user_id, relevant_items in list(test_interactions.items())[:200]:  # Lấy tối đa 200 users
        recs = adapter.get_top_n_items(user_id, n=top_k, exclude_seen=True)
        rec_ids = [r['item_id'] for r in recs]

        recalls.append(recall_at_k(rec_ids, relevant_items, top_k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant_items, top_k))
        all_recs.append(rec_ids)

    num_items = len(adapter.metadata.get_all_ids()) if adapter.metadata else 10000
    cov = coverage(all_recs, num_items)

    return {
        f'recall@{top_k}': sum(recalls) / len(recalls) if recalls else 0.0,
        f'ndcg@{top_k}'  : sum(ndcgs)   / len(ndcgs)   if ndcgs   else 0.0,
        'coverage'        : cov,
        'num_test_users'  : len(recalls),
    }


def evaluate_rag(retriever, test_queries: list[dict], top_k: int = 5) -> dict:
    """
    Đánh giá RAG retrieval quality.

    test_queries: list[{'query': str, 'relevant_turn_ids': list[str]}]
    """
    logger.info(f"Evaluating RAG: {len(test_queries)} queries, top_k={top_k}")
    precisions = []
    latencies  = []

    for q in test_queries:
        t0 = time.time()
        results = retriever.retrieve(q['query'], k=top_k)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = set(r['turn_id'] for r in results)
        relevant_ids  = set(q.get('relevant_turn_ids', []))

        if relevant_ids:
            precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids) if retrieved_ids else 0.0
            precisions.append(precision)

    return {
        'rag_precision@k'    : sum(precisions) / len(precisions) if precisions else 0.0,
        'rag_avg_latency_ms' : sum(latencies) / len(latencies) if latencies else 0.0,
        'rag_num_queries'    : len(test_queries),
    }


def evaluate_pipeline_latency(recommender, user_id: int = 1, n_queries: int = 20) -> dict:
    """
    Đo latency end-to-end của toàn bộ pipeline.

    Đo:
      - RAG latency
      - GNN+GCL latency
      - LLM latency
      - Total latency
    """
    logger.info(f"Latency test: {n_queries} queries...")

    test_queries = [
        "Tôi muốn xem phim hành động",
        "Gợi ý phim hài hay",
        "Phim khoa học viễn tưởng nào tốt?",
        "Có phim hoạt hình nào hay không?",
        "Tôi thích phim tâm lý, hãy gợi ý",
        "Phim kinh dị đặc sắc?",
        "Gợi ý phim drama cho tôi",
        "Phim lịch sử nào đáng xem?",
    ]

    results = []
    recommender.start_session(user_id + 9000)   # Session riêng cho test

    for i in range(n_queries):
        query = test_queries[i % len(test_queries)]
        t0 = time.time()
        result = recommender.chat(user_id=user_id + 9000, query=query)
        total_ms = (time.time() - t0) * 1000

        results.append({
            'total_ms'     : total_ms,
            'llm_ms'       : result.response.latency_ms,
            'rag_turns'    : len(result.retrieved_context),
            'gnn_items'    : len(result.gnn_recommendations),
            'is_mock'      : result.response.is_mock,
        })

        if (i + 1) % 5 == 0:
            logger.info(f"  {i+1}/{n_queries} done")

    recommender.end_session(user_id + 9000)

    total_times = [r['total_ms'] for r in results]
    llm_times   = [r['llm_ms']   for r in results]

    return {
        'n_queries'           : n_queries,
        'avg_total_ms'        : sum(total_times) / len(total_times),
        'p50_total_ms'        : sorted(total_times)[len(total_times)//2],
        'p95_total_ms'        : sorted(total_times)[int(len(total_times)*0.95)],
        'avg_llm_ms'          : sum(llm_times) / len(llm_times),
        'avg_rag_turns'       : sum(r['rag_turns'] for r in results) / len(results),
        'avg_gnn_items'       : sum(r['gnn_items'] for r in results) / len(results),
        'mock_mode'           : results[0]['is_mock'] if results else True,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate PCRS System')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--latency-test', action='store_true')
    parser.add_argument('--n-queries', type=int, default=20)
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db')
    args = parser.parse_args()

    # Import và khởi tạo
    from src.rag_conversational.conversation_store import ConversationStore
    from src.rag_conversational.gnn_gcl_adapter import load_gnn_gcl_adapter
    from src.rag_conversational.rag_retriever import RAGRetriever
    from src.rag_conversational.conversational_recommender import ConversationalRecommender
    from src.rag_conversational.response_generator import ResponseGenerator
    from src.rag_conversational.prompt_builder import PromptBuilder

    logger.info("=" * 60)
    logger.info("PCRS Evaluation")
    logger.info("=" * 60)

    store       = ConversationStore(persist_dir=args.chroma_dir,
                                    collection_name='synthetic_conversations')
    adapter     = load_gnn_gcl_adapter()
    retriever   = RAGRetriever(store, k=args.top_k)

    all_results = {}

    # 1. GNN+GCL Evaluation
    logger.info("\n[1] GNN+GCL Evaluation...")
    # Tạo test set giả (trong thực tế: load holdout set)
    import csv
    test_set = defaultdict(set)
    ratings_path = 'ml-latest-small/ratings.csv'
    if Path(ratings_path).exists():
        all_data = defaultdict(list)
        with open(ratings_path) as f:
            for row in csv.DictReader(f):
                if float(row['rating']) >= 4.0:
                    all_data[int(row['userId'])].append(int(row['movieId']))
        # 80/20 split
        for uid, items in list(all_data.items())[:100]:
            random.shuffle(items)
            split = max(1, int(len(items) * 0.8))
            test_set[uid] = set(items[split:])

    gnn_results = evaluate_gnn_gcl(adapter, test_set, args.top_k)
    all_results.update(gnn_results)
    logger.info(f"  GNN+GCL: {gnn_results}")

    # 2. RAG Evaluation (dùng câu hỏi mẫu)
    if store.count() > 0:
        logger.info("\n[2] RAG Evaluation...")
        test_queries = [
            {'query': 'phim hành động', 'relevant_turn_ids': []},
            {'query': 'phim hài hước', 'relevant_turn_ids': []},
            {'query': 'phim gia đình', 'relevant_turn_ids': []},
            {'query': 'nhạc indie', 'relevant_turn_ids': []},
            {'query': 'đàn guitar cho người mới', 'relevant_turn_ids': []},
        ]
        rag_results = evaluate_rag(retriever, test_queries, args.top_k)
        all_results.update(rag_results)
        logger.info(f"  RAG: {rag_results}")

    # 3. Pipeline Latency
    if args.latency_test:
        logger.info("\n[3] Pipeline Latency Test...")
        recommender = ConversationalRecommender(
            store=store,
            gnn_adapter=adapter,
            response_generator=ResponseGenerator(),
            prompt_builder=PromptBuilder(),
        )
        latency_results = evaluate_pipeline_latency(
            recommender, n_queries=args.n_queries
        )
        all_results.update(latency_results)
        logger.info(f"  Latency: {latency_results}")

    # Hiển thị kết quả tổng hợp
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in all_results.items():
        if isinstance(value, float):
            print(f"  {metric:30s}: {value:.4f}")
        else:
            print(f"  {metric:30s}: {value}")
    print("=" * 60)


if __name__ == '__main__':
    main()
