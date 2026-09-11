"""
src/rag_conversational/generate_synthetic_conversations.py
==========================================================
Sinh dữ liệu hội thoại tổng hợp (Synthetic Conversations) từ 3 nguồn:
  1. MovieLens ml-latest-small: Phim, ratings, tags
  2. LastFM hetrec2011-lastfm-2k: Nghệ sĩ, tương tác
  3. Amazon Musical Instruments reviews

Mỗi conversation được sinh theo template:
  - Query : Mẫu câu hỏi tự nhiên dựa trên sở thích của user
  - Response: Mẫu phản hồi hệ thống có chứa tên item + lý do

Sử dụng:
    python generate_synthetic_conversations.py --dataset all --output ./chroma_db
"""

import os
import sys
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_conversational.conversation_store import ConversationStore, ConversationTurn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ============================================================

def _find_repo_root():
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / 'ml-latest-small').exists() or (current / 'hetrec2011-lastfm-2k').exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent

BASE_DIR = _find_repo_root()

MOVIELENS_DIR = BASE_DIR / 'ml-latest-small'
LASTFM_DIR    = BASE_DIR / 'hetrec2011-lastfm-2k'
AMAZON_FILE   = BASE_DIR / 'reviews_Musical_Instruments_5.json' / 'Musical_Instruments_5.json'

# ============================================================
# TEMPLATE HỎI & ĐÁP
# ============================================================

MOVIE_QUERY_TEMPLATES = [
    "Tôi muốn xem phim {genre} hay, bạn có thể gợi ý không?",
    "Có phim nào hay giống {movie_title} không?",
    "Tôi thích phim về {tag}, hãy gợi ý cho tôi vài bộ.",
    "Gợi ý cho tôi phim {genre} được ra mắt gần đây.",
    "Tôi vừa xem {movie_title} và thích lắm. Có phim gì tương tự không?",
    "Bạn nghĩ phim nào trong thể loại {genre} đáng xem nhất?",
    "Tôi đang tìm phim {genre} để xem cuối tuần này.",
    "Có bộ phim hay nào về chủ đề {tag} không?",
]

MOVIE_RESPONSE_TEMPLATES = [
    "Dựa trên sở thích của bạn, tôi gợi ý: {recommendations}. "
    "Đây đều là những bộ phim {genre} được đánh giá cao!",

    "Vì bạn yêu thích {movie_title}, tôi nghĩ bạn sẽ thích: {recommendations}. "
    "Chúng có phong cách tương tự và được nhiều người khen ngợi.",

    "Với chủ đề {tag} mà bạn quan tâm, đây là những bộ phim phù hợp: {recommendations}.",

    "Tôi đề xuất {recommendations}. "
    "Tất cả đều thuộc thể loại {genre} và có điểm đánh giá cao từ cộng đồng.",
]

MUSIC_QUERY_TEMPLATES = [
    "Tôi thích nghe nhạc của {artist}, có nghệ sĩ nào tương tự không?",
    "Gợi ý cho tôi một số nghệ sĩ {genre} hay.",
    "Tôi muốn khám phá thêm nhạc {genre}, bạn có đề xuất nào không?",
    "Ngoài {artist} ra, bạn nghĩ tôi nên nghe ai?",
    "Tôi đang tìm nhạc có phong cách giống {artist}.",
]

MUSIC_RESPONSE_TEMPLATES = [
    "Nếu bạn thích {artist}, tôi nghĩ bạn sẽ yêu thích: {recommendations}. "
    "Họ có phong cách âm nhạc tương tự và được nhiều fan đánh giá cao.",

    "Dựa trên sở thích của bạn, đây là những nghệ sĩ bạn nên khám phá: {recommendations}.",

    "Với gu âm nhạc của bạn, tôi đề xuất: {recommendations}. "
    "Đây đều là những nghệ sĩ nổi bật trong thể loại {genre}.",
]

AMAZON_QUERY_TEMPLATES = [
    "Tôi cần {product_type} phù hợp cho người mới học, bạn có thể gợi ý không?",
    "Sản phẩm nào trong danh mục {product_type} tốt nhất hiện tại?",
    "Tôi đang xem xét mua {product_name}, có ý kiến gì không?",
    "Gợi ý cho tôi {product_type} trong tầm giá hợp lý.",
    "Có {product_type} nào tốt cho người chuyên nghiệp không?",
]

AMAZON_RESPONSE_TEMPLATES = [
    "Dựa trên các đánh giá tôi tổng hợp được, đây là những gợi ý tốt: {recommendations}. "
    "Chúng được người dùng đánh giá rất cao!",

    "Cho người mới học, tôi đề xuất: {recommendations}. "
    "Đây đều là những lựa chọn phổ biến với chất lượng tốt ở tầm giá vừa phải.",
]


# ============================================================
# GENERATOR CHO MOVIELENS
# ============================================================

def load_movielens(data_dir: Path) -> dict:
    """Load toàn bộ dữ liệu MovieLens."""
    logger.info("Loading MovieLens data...")

    # Movies
    movies = {}
    with open(data_dir / 'movies.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            movie_id = int(row['movieId'])
            genres = row['genres'].split('|') if row['genres'] != '(no genres listed)' else []
            movies[movie_id] = {
                'title' : row['title'],
                'genres': genres,
            }

    # Tags theo movie
    movie_tags = defaultdict(list)
    with open(data_dir / 'tags.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            movie_tags[int(row['movieId'])].append(row['tag'].lower())

    # Ratings theo user
    user_ratings = defaultdict(list)
    with open(data_dir / 'ratings.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rating = float(row['rating'])
            if rating >= 4.0:   # Chỉ lấy ratings cao (positive)
                user_ratings[int(row['userId'])].append({
                    'movie_id' : int(row['movieId']),
                    'rating'   : rating,
                    'timestamp': int(row['timestamp']),
                })

    logger.info(
        f"  Movies: {len(movies)}, "
        f"Users with ratings >= 4.0: {len(user_ratings)}, "
        f"Movies with tags: {len(movie_tags)}"
    )
    return {'movies': movies, 'movie_tags': movie_tags, 'user_ratings': user_ratings}


def generate_movielens_conversations(
    data: dict,
    max_users: int = 100,
    turns_per_user: int = 5,
    seed: int = 42,
) -> list[ConversationTurn]:
    """
    Sinh synthetic conversations từ MovieLens.

    Với mỗi user có ratings cao:
      - Chọn ngẫu nhiên 1 phim họ thích làm "seed"
      - Sinh query dựa trên genre/tag của phim đó
      - Sinh response gợi ý các phim khác cùng genre
    """
    random.seed(seed)
    movies = data['movies']
    movie_tags = data['movie_tags']
    user_ratings = data['user_ratings']

    # Index phim theo genre
    genre_movies = defaultdict(list)
    for movie_id, info in movies.items():
        for genre in info['genres']:
            genre_movies[genre].append(movie_id)

    turns = []
    users = list(user_ratings.keys())[:max_users]

    for user_id in users:
        liked_movies = sorted(user_ratings[user_id], key=lambda x: -x['rating'])
        if not liked_movies:
            continue

        for turn_idx in range(turns_per_user):
            # Chọn seed movie
            seed_entry = random.choice(liked_movies[:10])  # Trong top 10 liked
            seed_movie = movies.get(seed_entry['movie_id'])
            if not seed_movie or not seed_movie['genres']:
                continue

            genre = random.choice(seed_movie['genres'])
            tags  = movie_tags.get(seed_entry['movie_id'], [])
            tag   = random.choice(tags) if tags else genre.lower()

            # Chọn query template
            template = random.choice(MOVIE_QUERY_TEMPLATES)
            query = template.format(
                genre       = genre,
                movie_title = seed_movie['title'],
                tag         = tag,
            )

            # Chọn 3-5 items cùng genre để gợi ý
            candidate_ids = [
                mid for mid in genre_movies.get(genre, [])
                if mid != seed_entry['movie_id']
            ]
            rec_ids = random.sample(candidate_ids, min(4, len(candidate_ids)))
            rec_titles = [movies[mid]['title'] for mid in rec_ids if mid in movies]

            if not rec_titles:
                continue

            rec_str = ', '.join(f'"{t}"' for t in rec_titles[:3])
            resp_template = random.choice(MOVIE_RESPONSE_TEMPLATES)
            response = resp_template.format(
                recommendations = rec_str,
                genre           = genre,
                movie_title     = seed_movie['title'],
                tag             = tag,
            )

            turn = ConversationTurn(
                user_id     = user_id,
                turn_index  = turn_idx,
                query       = query,
                response    = response,
                items_shown = rec_ids,
                timestamp   = seed_entry['timestamp'] + turn_idx * 3600,
                dataset_src = 'movielens',
            )
            turns.append(turn)

    logger.info(f"Generated {len(turns)} MovieLens conversations for {len(users)} users")
    return turns


# ============================================================
# GENERATOR CHO LASTFM
# ============================================================

def load_lastfm(data_dir: Path) -> dict:
    """Load dữ liệu LastFM."""
    logger.info("Loading LastFM data...")

    # Artists
    artists = {}
    with open(data_dir / 'artists.dat', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                artists[int(parts[0])] = {'name': parts[1]}

    # User-Artist interactions (listen counts)
    user_artists = defaultdict(list)
    with open(data_dir / 'user_artists.dat', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                user_id   = int(parts[0])
                artist_id = int(parts[1])
                weight    = int(parts[2])
                user_artists[user_id].append({
                    'artist_id': artist_id,
                    'weight'   : weight,
                })

    logger.info(f"  Artists: {len(artists)}, Users: {len(user_artists)}")
    return {'artists': artists, 'user_artists': user_artists}


def generate_lastfm_conversations(
    data: dict,
    max_users: int = 100,
    turns_per_user: int = 3,
    seed: int = 42,
) -> list[ConversationTurn]:
    """Sinh conversations từ LastFM listening history."""
    random.seed(seed)
    artists = data['artists']
    user_artists = data['user_artists']

    turns = []
    users = list(user_artists.keys())[:max_users]

    for user_id in users:
        # Sắp xếp nghệ sĩ theo tần suất nghe
        liked = sorted(user_artists[user_id], key=lambda x: -x['weight'])
        if not liked:
            continue

        for turn_idx in range(turns_per_user):
            seed_artist_entry = random.choice(liked[:5])
            seed_artist = artists.get(seed_artist_entry['artist_id'])
            if not seed_artist:
                continue

            template = random.choice(MUSIC_QUERY_TEMPLATES)
            query = template.format(
                artist = seed_artist['name'],
                genre  = 'nhạc indie',   # LastFM không có genre field đơn giản
            )

            # Gợi ý nghệ sĩ khác (random sampling từ toàn bộ)
            other_artists = [
                aid for aid, info in artists.items()
                if aid != seed_artist_entry['artist_id']
            ]
            rec_ids = random.sample(other_artists, min(4, len(other_artists)))
            rec_names = [artists[aid]['name'] for aid in rec_ids if aid in artists]

            if not rec_names:
                continue

            rec_str = ', '.join(f'"{n}"' for n in rec_names[:3])
            resp_template = random.choice(MUSIC_RESPONSE_TEMPLATES)
            response = resp_template.format(
                recommendations = rec_str,
                artist          = seed_artist['name'],
                genre           = 'nhạc indie',
            )

            turn = ConversationTurn(
                user_id     = user_id + 100000,   # Tránh trùng với MovieLens user_id
                turn_index  = turn_idx,
                query       = query,
                response    = response,
                items_shown = rec_ids,
                timestamp   = time.time() - (len(users) - user_id) * 86400,
                dataset_src = 'lastfm',
            )
            turns.append(turn)

    logger.info(f"Generated {len(turns)} LastFM conversations for {len(users)} users")
    return turns


# ============================================================
# GENERATOR CHO AMAZON
# ============================================================

def load_amazon(amazon_file: Path) -> list[dict]:
    """Load Amazon reviews."""
    logger.info("Loading Amazon Musical Instruments reviews...")
    reviews = []
    with open(amazon_file, encoding='utf-8') as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    logger.info(f"  Reviews: {len(reviews)}")
    return reviews


def generate_amazon_conversations(
    reviews: list[dict],
    max_reviews: int = 200,
    seed: int = 42,
) -> list[ConversationTurn]:
    """
    Sinh conversations từ Amazon reviews.
    Mỗi review → 1 synthetic conversation (dùng reviewText làm context).
    """
    random.seed(seed)
    turns = []
    sampled = random.sample(reviews, min(max_reviews, len(reviews)))

    # Nhóm sản phẩm theo ASIN
    products = defaultdict(list)
    for r in reviews:
        products[r.get('asin', '')].append(r)

    for idx, review in enumerate(sampled):
        product_id = review.get('asin', '')
        reviewer_id = review.get('reviewerID', f'amazon_user_{idx}')
        review_text = review.get('reviewText', '')
        summary     = review.get('summary', '')

        if len(review_text) < 30:
            continue

        # Tạo user_id từ reviewerID
        user_id = abs(hash(reviewer_id)) % 90000 + 200000  # Tránh overlap

        # Lấy context từ summary và review text
        product_type = 'nhạc cụ'
        if 'guitar' in review_text.lower() or 'guitar' in summary.lower():
            product_type = 'guitar'
        elif 'string' in review_text.lower():
            product_type = 'dây đàn'
        elif 'pick' in review_text.lower():
            product_type = 'pick guitar'

        template = random.choice(AMAZON_QUERY_TEMPLATES)
        query = template.format(
            product_type = product_type,
            product_name = summary[:50] if summary else product_type,
        )

        # Gợi ý từ các sản phẩm có rating cao
        good_products = [
            r.get('asin', '') for r in reviews
            if float(r.get('overall', 0)) >= 4.0
            and r.get('asin', '') != product_id
        ]
        rec_asins = random.sample(good_products, min(3, len(good_products)))

        resp_template = random.choice(AMAZON_RESPONSE_TEMPLATES)
        rec_names = [f"ASIN:{a[:8]}" for a in rec_asins]
        response = resp_template.format(
            recommendations = ', '.join(f'"{n}"' for n in rec_names),
        )

        # Dùng review text như một phần của response để RAG có thể truy xuất
        response = f"{response} Theo một đánh giá: \"{review_text[:200]}...\""

        # Map ASIN thành integer ID
        item_ids = [abs(hash(a)) % 50000 + 300000 for a in rec_asins]

        turn = ConversationTurn(
            user_id     = user_id,
            turn_index  = 0,
            query       = query,
            response    = response,
            items_shown = item_ids,
            timestamp   = float(review.get('unixReviewTime', time.time())),
            dataset_src = 'amazon',
        )
        turns.append(turn)

    logger.info(f"Generated {len(turns)} Amazon conversations")
    return turns


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Sinh synthetic conversations cho RAG')
    parser.add_argument('--dataset', choices=['all', 'movielens', 'lastfm', 'amazon'],
                        default='all', help='Dataset cần sinh')
    parser.add_argument('--output', default='./chroma_db',
                        help='Thư mục lưu ChromaDB')
    parser.add_argument('--max-users-ml', type=int, default=100,
                        help='Số user tối đa từ MovieLens')
    parser.add_argument('--max-users-lfm', type=int, default=100,
                        help='Số user tối đa từ LastFM')
    parser.add_argument('--max-reviews', type=int, default=300,
                        help='Số reviews tối đa từ Amazon')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Synthetic Conversation Generator for GNN+GCL+RAG")
    logger.info("=" * 60)

    # Khởi tạo store
    store = ConversationStore(
        persist_dir     = args.output,
        collection_name = 'synthetic_conversations',
    )
    logger.info(f"Store initialized: {store}")

    all_turns = []

    # MovieLens
    if args.dataset in ['all', 'movielens']:
        if MOVIELENS_DIR.exists():
            ml_data = load_movielens(MOVIELENS_DIR)
            ml_turns = generate_movielens_conversations(
                ml_data,
                max_users      = args.max_users_ml,
                turns_per_user = 5,
                seed           = args.seed,
            )
            all_turns.extend(ml_turns)
        else:
            logger.warning(f"MovieLens dir không tồn tại: {MOVIELENS_DIR}")

    # LastFM
    if args.dataset in ['all', 'lastfm']:
        if LASTFM_DIR.exists():
            lfm_data = load_lastfm(LASTFM_DIR)
            lfm_turns = generate_lastfm_conversations(
                lfm_data,
                max_users      = args.max_users_lfm,
                turns_per_user = 3,
                seed           = args.seed,
            )
            all_turns.extend(lfm_turns)
        else:
            logger.warning(f"LastFM dir không tồn tại: {LASTFM_DIR}")

    # Amazon
    if args.dataset in ['all', 'amazon']:
        if AMAZON_FILE.exists():
            amazon_reviews = load_amazon(AMAZON_FILE)
            amazon_turns = generate_amazon_conversations(
                amazon_reviews,
                max_reviews = args.max_reviews,
                seed        = args.seed,
            )
            all_turns.extend(amazon_turns)
        else:
            logger.warning(f"Amazon file không tồn tại: {AMAZON_FILE}")

    if not all_turns:
        logger.error("Không có conversations nào được sinh ra!")
        return

    # Xáo trộn và thêm vào store
    random.shuffle(all_turns)
    logger.info(f"\nTổng số conversations: {len(all_turns)}")
    logger.info("Adding to ChromaDB (có thể mất vài phút)...")

    store.add_turns_batch(all_turns)

    logger.info(f"\n✓ Hoàn thành! Store hiện có {store.count()} conversations")
    logger.info(f"  Lưu tại: {args.output}")

    # Test tìm kiếm với 10 prompt phù hợp với dữ liệu MovieLens
    TEST_PROMPTS = [
        "Tôi muốn xem phim hành động hay, bạn có thể gợi ý không?",
        "Gợi ý cho tôi phim Comedy để xem cuối tuần này.",
        "Có phim nào hay giống The Dark Knight không?",
        "Tôi thích phim về tình yêu lãng mạn, hãy gợi ý cho tôi vài bộ.",
        "Tôi đang tìm phim Sci-Fi hấp dẫn để xem tối nay.",
        "Bạn nghĩ phim nào trong thể loại Drama đáng xem nhất?",
        "Tôi vừa xem Forrest Gump và thích lắm. Có phim gì tương tự không?",
        "Có bộ phim hay nào về chủ đề friendship không?",
        "Gợi ý cho tôi phim Horror kinh điển được đánh giá cao.",
        "Tôi muốn xem phim Animation phù hợp cho cả gia đình.",
    ]


    logger.info("\n--- Test Search (10 Prompt - MovieLens) ---")
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        logger.info(f"\n[Prompt {i:02d}] {prompt}")
        results = store.search(prompt, k=3)
        if results:
            for r in results:
                logger.info(f"  Score {r['score']:.3f}: {r['query'][:80]}")
        else:
            logger.info("  (Không tìm thấy kết quả)")

    # Test gợi ý dựa trên lịch sử
    _test_history_based_search(store, all_turns)


def _test_history_based_search(store, all_turns: list) -> None:
    """
    Test gợi ý dựa trên lịch sử hội thoại của user.

    Chiến lược:
      1. Lấy user thực (có nhiều turns nhất) từ dữ liệu đã sinh.
      2. Hiển thị lịch sử hội thoại của user đó.
      3. Tìm kiếm Personalized (user_id filter) vs Global (không filter).
      4. So sánh kết quả để thấy hiệu quả của cá nhân hóa.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Gợi Ý Dựa Trên Lịch Sử Người Dùng")
    logger.info("=" * 60)

    if not all_turns:
        logger.warning("Không có turns để test lịch sử.")
        return

    # --- Bước 1: Chọn user có nhiều turns nhất ---
    from collections import Counter
    user_turn_counts = Counter(t.user_id for t in all_turns)
    # Lấy top-3 user để test đa dạng
    top_users = [uid for uid, _ in user_turn_counts.most_common(3)]

    for test_user_id in top_users:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"👤 USER ID: {test_user_id}  "
                    f"(tổng {user_turn_counts[test_user_id]} turns trong dataset)")

        # --- Bước 2: Hiển thị lịch sử của user ---
        history = store.get_user_history(test_user_id)
        if not history:
            logger.info("  (Chưa có lịch sử trong store)")
            continue

        logger.info(f"\n📜 Lịch sử hội thoại ({len(history)} lượt):")
        for idx, turn in enumerate(history[:5], 1):   # Hiển thị tối đa 5
            q_text = turn['query'] if isinstance(turn, dict) else turn.query
            r_text = turn['response'] if isinstance(turn, dict) else turn.response
            logger.info(f"  [{idx}] Q: {q_text[:70]}")
            logger.info(f"       R: {r_text[:70]}...")

        # --- Bước 3: Sinh query mới dựa trên lịch sử ---
        # Lấy genre/tag từ lịch sử để tạo query tiếp theo tự nhiên
        last_turn  = history[-1]
        last_query = (last_turn['query'] if isinstance(last_turn, dict) else last_turn.query) if history else ""
        # Trích genre từ query cuối (ví dụ: "phim Action", "phim Drama")
        follow_up_queries = [
            "Bạn có thể gợi ý thêm phim tương tự những gì tôi đã xem không?",
            "Dựa trên sở thích của tôi, phim nào tôi chưa xem mà nên thử?",
            "Tôi muốn xem gì tiếp theo phù hợp với gu của mình?",
        ]

        logger.info(f"\n🔍 Personalized Search (có lọc theo user_id={test_user_id}):")
        for q in follow_up_queries:
            logger.info(f"\n  Query: \"{q}\"")

            # Personalized: chỉ tìm trong lịch sử của user này
            personal_results = store.search(q, user_id=test_user_id, k=3)
            if personal_results:
                for r in personal_results:
                    logger.info(f"    [Personal] Score {r['score']:.3f}: {r['query'][:65]}")
            else:
                logger.info("    [Personal] Không tìm thấy (user chưa có lịch sử đủ)")

            # Global: tìm toàn bộ store (không filter user)
            global_results = store.search(q, k=3)
            if global_results:
                for r in global_results:
                    logger.info(f"    [Global]   Score {r['score']:.3f}: {r['query'][:65]}")

        logger.info(f"\n📊 Phân tích lịch sử user {test_user_id}:")
        dataset_sources = Counter(
            t.get('dataset_src', 'unknown') if isinstance(t, dict) else getattr(t, 'dataset_src', 'unknown')
            for t in history
        )
        for src, cnt in dataset_sources.items():
            logger.info(f"  - {src}: {cnt} turns")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Hoàn thành test lịch sử người dùng")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
