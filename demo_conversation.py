"""
demo_conversation.py
====================
Demo CLI Chatbot cho GNN+GCL+RAG Conversational Recommender.

Cách dùng:
    # Bước 1: Cài dependencies
    pip install chromadb sentence-transformers google-generativeai

    # Bước 2: Sinh dữ liệu hội thoại
    python -m src.rag_conversational.generate_synthetic_conversations --dataset all

    # Bước 3: Chạy demo
    python demo_conversation.py --user-id 1
    python demo_conversation.py --user-id 1 --api-key YOUR_GEMINI_KEY
    python demo_conversation.py --test-mode   # Chạy test tự động
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,      # Chỉ hiện WARNING+ để UI gọn
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Bật DEBUG cho rag_conversational nếu cần trace
# logging.getLogger('src.rag_conversational').setLevel(logging.DEBUG)


WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🎬 GNN+GCL+RAG Personalized Conversational Recommender   ║
║                                                              ║
║   Kiến trúc: GNN+GCL (LightGCN) + RAG (ChromaDB) + Gemini  ║
║   Dataset: MovieLens | LastFM | Amazon Musical Instruments  ║
╚══════════════════════════════════════════════════════════════╝

Gõ 'quit' để thoát | 'stats' để xem thống kê | 'accept <id>' để feedback
"""

HELP_TEXT = """
Lệnh:
  <câu hỏi>       : Nhập câu hỏi để nhận gợi ý
  accept <item_id>: Chấp nhận một gợi ý (feedback)
  stats           : Xem thống kê phiên hội thoại
  debug           : Hiển thị prompt đã gửi cho LLM
  clear           : Xóa màn hình
  help            : Hiển thị trợ giúp này
  quit / exit     : Thoát
"""

SAMPLE_QUERIES = [
    "Tôi muốn xem phim hành động hay, có gợi ý gì không?",
    "Tôi thích phim khoa học viễn tưởng như Interstellar, có gì tương tự không?",
    "Gợi ý phim hài để xem cuối tuần?",
    "Có phim nào về chiến tranh được đánh giá cao không?",
    "Tôi muốn khám phá phim kinh dị, bắt đầu từ đâu?",
]


def build_recommender(api_key: str = None, chroma_dir: str = './chroma_db'):
    """Khởi tạo ConversationalRecommender."""
    from src.rag_conversational.conversation_store import ConversationStore
    from src.rag_conversational.gnn_gcl_adapter import GNNGCLAdapter, load_gnn_gcl_adapter
    from src.rag_conversational.prompt_builder import PromptBuilder, PromptConfig
    from src.rag_conversational.response_generator import ResponseGenerator
    from src.rag_conversational.conversational_recommender import ConversationalRecommender

    print("⏳ Đang khởi tạo hệ thống...")

    # 1. Conversation Store
    print("  [1/4] Khởi tạo ChromaDB conversation store...")
    store = ConversationStore(
        persist_dir     = chroma_dir,
        collection_name = 'synthetic_conversations',
    )
    count = store.count()
    if count == 0:
        print(f"  ⚠️  Store trống! Hãy chạy trước:")
        print(f"      python -m src.rag_conversational.generate_synthetic_conversations")
        print(f"  Tiếp tục với store trống (không có RAG context)...")
    else:
        print(f"  ✓ Store: {count} conversations")

    # 2. GNN+GCL Adapter
    print("  [2/4] Khởi tạo GNN+GCL adapter...")
    gnn_adapter = load_gnn_gcl_adapter(
        model_path=None,   # Dùng dummy mode (random scores) nếu không có checkpoint
    )
    print(f"  ✓ GNN+GCL: {gnn_adapter}")

    # 3. Prompt Builder
    print("  [3/4] Khởi tạo Prompt Builder...")
    prompt_builder = PromptBuilder(
        config=PromptConfig(language='vi', style='friendly', top_n_display=8)
    )
    print(f"  ✓ PromptBuilder: {prompt_builder}")

    # 4. Response Generator (Gemini)
    print("  [4/4] Khởi tạo Gemini API...")
    api_key_final = api_key or os.environ.get('GOOGLE_API_KEY', '')
    response_gen = ResponseGenerator(
        api_key    = api_key_final or None,
        model_name = 'gemini-flash-latest',
        temperature= 0.7,
    )
    print(f"  ✓ ResponseGenerator: {response_gen}")

    # 5. Assembling pipeline
    recommender = ConversationalRecommender(
        store              = store,
        gnn_adapter        = gnn_adapter,
        prompt_builder     = prompt_builder,
        response_generator = response_gen,
        top_n              = 10,
        top_k_rag          = 5,
        save_turns         = True,
        personalized_rag   = False,  # Dùng toàn bộ store (hữu ích cho synthetic data)
    )

    print("\n✅ Hệ thống sẵn sàng!\n")
    return recommender


def run_interactive_demo(recommender, user_id: int):
    """Chạy chatbot CLI tương tác."""
    from src.rag_conversational.conversational_recommender import ConversationalRecommender

    print(WELCOME_BANNER)
    print(f"👤 Người dùng: ID = {user_id}")
    print(f"\nCâu hỏi mẫu:")
    for i, q in enumerate(SAMPLE_QUERIES[:3], 1):
        print(f"  {i}. {q}")
    print()

    recommender.start_session(user_id)
    last_result = None

    while True:
        try:
            user_input = input("💬 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nTạm biệt! 👋")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # --- Lệnh đặc biệt ---
        if lower in ('quit', 'exit', 'q'):
            print("\nTạm biệt! 👋")
            break

        elif lower == 'help':
            print(HELP_TEXT)
            continue

        elif lower == 'stats':
            stats = recommender.get_session_stats(user_id)
            print("\n📊 THỐNG KÊ PHIÊN:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue

        elif lower == 'debug' and last_result:
            print("\n🔍 PROMPT ĐÃ GỬI:")
            print(last_result.prompt[:2000])
            if len(last_result.prompt) > 2000:
                print(f"... (truncated, total {len(last_result.prompt)} chars)")
            print()
            continue

        elif lower == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        elif lower.startswith('accept '):
            try:
                item_id = int(user_input.split()[1])
                recommender.accept_item(user_id, item_id)
                print(f"✅ Đã ghi nhận: bạn thích item {item_id}")
            except (ValueError, IndexError):
                print("❌ Cú pháp: accept <item_id>")
            continue

        # --- Xử lý câu hỏi thực ---
        print("\n⏳ Đang xử lý...")
        try:
            result = recommender.chat(user_id=user_id, query=user_input)
            last_result = result
            print(result.to_display())

            # Gợi ý top items được đề xuất
            if result.gnn_recommendations:
                print("🎯 Top gợi ý (nhập 'accept <số>' để xác nhận thích):")
                for i, item in enumerate(result.gnn_recommendations[:5], 1):
                    print(f"   [{item.get('item_id', i)}] {item.get('title', f'Item {i}')}")
                print()

        except Exception as e:
            logger.exception("Lỗi khi xử lý câu hỏi")
            print(f"❌ Lỗi: {e}")

    # Kết thúc phiên
    session = recommender.end_session(user_id)
    if session:
        print(f"\n📈 Tổng kết: {session.turn_count} lượt hội thoại, "
              f"{len(session.accepted_items)} items được chấp nhận")


def run_test_mode(recommender):
    """Chạy test tự động với các câu hỏi mẫu."""
    print("\n🧪 TEST MODE — Chạy 3 câu hỏi mẫu...\n")
    user_id = 999
    recommender.start_session(user_id)

    for i, query in enumerate(SAMPLE_QUERIES[:3], 1):
        print(f"\n--- Test {i}/3 ---")
        print(f"Query: {query}")
        result = recommender.chat(user_id=user_id, query=query)
        print(f"Response ({result.total_latency_ms:.0f}ms): {result.response.text[:200]}...")
        print(f"RAG context: {len(result.retrieved_context)} turns")
        print(f"GNN recs: {len(result.gnn_recommendations)} items")

    recommender.end_session(user_id)
    print("\n✅ Test hoàn thành!")

    # System stats
    stats = recommender.get_system_stats()
    print("\n📊 SYSTEM STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='GNN+GCL+RAG Conversational Recommender Demo'
    )
    parser.add_argument('--user-id', type=int, default=1,
                        help='ID người dùng (default: 1)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Google Gemini API Key (hoặc set GOOGLE_API_KEY env)')
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db',
                        help='Thư mục ChromaDB (default: ./chroma_db)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Chạy test tự động thay vì interactive')
    args = parser.parse_args()

    # Khởi tạo
    recommender = build_recommender(
        api_key    = args.api_key,
        chroma_dir = args.chroma_dir,
    )

    if args.test_mode:
        run_test_mode(recommender)
    else:
        run_interactive_demo(recommender, args.user_id)


if __name__ == '__main__':
    main()
