# GNN + GCL + LLM: Personalized Conversational Recommender System (PCRS)

> **Báo cáo và mã nguồn nghiên cứu bài toán Hệ thống Gợi ý Hội thoại Cá nhân hóa (PCRS)** kết hợp **Mạng Nơ-ron Đồ thị (LightGCN)**, **Học Tương phản (SimGCL)** và **Truy xuất Tăng cường Ngữ cảnh Đa phiên (RAG)**.  
> **Giảng viên hướng dẫn:** TS. Đỗ Thị Liên — Khoa Công nghệ Thông tin, Học viện Công nghệ Bưu chính Viễn thông.

---

## 📌 Tổng Quan Kiến Trúc Hệ Thống

Hệ thống được thiết kế theo kiến trúc **2 nhánh song song** (Dual-branch Architecture) và quy trình **3 bước chuẩn mực**:

```
                              [USER QUERY q_t & USER ID u]
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
[NHÁNH 1A: RAG CONTEXT RETRIEVAL]                   [NHÁNH 1B: GNN+GCL DUAL RECOMMENDATION]
• Bi-Encoder all-MiniLM-L6-v2 (384-dim)              • LightGCN K-layer Message Passing
• ChromaDB Cosine Indexing (HNSW)                   • SimGCL Contrastive Learning (InfoNCE)
• MMR Reranking khử trùng lặp                       • Dot Product Ranking: y_ui = e_u^T · e_i
• Trả về: Top-K lượt đàm thoại tiền lệ              • Trả về: Top-N Recommended Items
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           ▼
                           [BƯỚC 3: STRUCTURED PROMPT FUSION]
                           • Khối 1: Ngữ cảnh hội thoại liên quan từ RAG
                           • Khối 2: Danh sách Top-N gợi ý từ GNN+GCL
                           • Khối 3: Tóm tắt truy vấn hiện tại q_t
                           • Khóa ảo giác (Anti-Hallucination Guardrail)
                                           │
                                           ▼
                           [BƯỚC 4: LLM RESPONSE GENERATION]
                           • Google Gemini Flash API (hoặc Local Rule-based Mock)
                           • Sinh phản hồi tự nhiên, giải thích rõ ràng lý do gợi ý
                                           │
                                           ▼
                           [BƯỚC 5: PHẢN HỒI & CẬP NHẬT VECTOR DB]
```

---

## 📂 Cấu Trúc Thư Mục

```
.
├── architech-report-pcrs.md      # Đặc tả kiến trúc kỹ thuật chi tiết toàn hệ thống PCRS
├── requirements_rag.txt          # Danh sách dependencies thư viện
├── README.md                     # Tài liệu hướng dẫn sử dụng
├── .gitignore                    # Cấu hình bỏ qua các file tạm, docx và script sinh tài liệu
│
├── src/                          # Mã nguồn chính
│   ├── model_lightgcn_gcl.py     # Mô hình LightGCN kết hợp SimGCL / SGL
│   ├── model_lightgcn.py         # Baseline LightGCN
│   ├── model_mf.py               # Baseline Matrix Factorization
│   ├── model_cb.py               # Baseline Content-Based
│   ├── model_hybrid.py           # Baseline Late-Fusion Hybrid
│   ├── data_loader.py            # Nạp và tiền xử lý dữ liệu tương tác
│   ├── train_eval.py             # Huấn luyện và đánh giá mô hình gợi ý
│   ├── metrics.py                # Đo lường Recall@K, NDCG@K, Precision@K
│   │
│   └── rag_conversational/       # Gói module RAG & Hội thoại
│       ├── __init__.py
│       ├── conversational_recommender.py  # Điều phối toàn bộ pipeline đầu cuối
│       ├── conversation_store.py          # Quản lý CSDL Vector ChromaDB
│       ├── rag_retriever.py               # Tìm kiếm ngữ nghĩa & MMR Reranking
│       ├── gnn_gcl_adapter.py             # Adapter trích xuất gợi ý từ GNN
│       ├── prompt_builder.py              # Xây dựng prompt 3 thành phần chuẩn hóa
│       ├── response_generator.py          # Sinh phản hồi qua Gemini Flash / Mock
│       ├── session_manager.py             # Quản lý phiên đàm thoại người dùng
│       └── generate_synthetic_conversations.py # Sinh dữ liệu hội thoại mẫu & benchmark
│
├── ml-latest-small/              # Bộ dữ liệu MovieLens (movies, ratings, tags, links)
├── checkpoints/                  # Trọng số mô hình đã huấn luyện (simgcl_best.pth)
├── drawio_diagrams/              # Sơ đồ thiết kế kiến trúc Draw.io
│
├── demo_conversation.py          # Giao diện chạy thử nghiệm đàm thoại (CLI / Chatbot)
├── evaluate_pcrs.py              # Đo lường độ trễ và đánh giá chất lượng gợi ý PCRS
├── train_and_save_simgcl.py      # Huấn luyện và lưu checkpoint SimGCL
└── run_all_experiments.py        # Chạy toàn bộ các bài thực nghiệm so sánh
```

---

## 🚀 Hướng Dẫn Cài Đặt và Khởi Chạy

### 1. Cài đặt Môi trường
```powershell
pip install -r requirements_rag.txt
```

### 2. Sinh Dữ liệu Hội thoại Mẫu và Lập Chỉ mục Vector
```powershell
python -m src.rag_conversational.generate_synthetic_conversations --output ./chroma_db --num-users 50
```
Lệnh này sẽ:
1. Trích xuất tương tác tích cực của người dùng từ `ml-latest-small/`.
2. Sinh các phiên hội thoại đa dạng (250+ turns).
3. Nhúng vector qua `all-MiniLM-L6-v2` và lưu bền vững vào ChromaDB.
4. Tự động chạy benchmark bộ **10 Test Prompts** và **Thực nghiệm Gợi ý Theo Lịch sử Người dùng** (Personalized Search vs Global Search).

### 3. Chạy Thử Nghiệm Tương Tác Hội Thoại (Interactive Demo)
```powershell
# Chế độ kiểm thử nhanh với bộ sinh nội bộ (Offline Mock - độ trễ < 25ms)
python demo_conversation.py --test-mode

# Chế độ trò chuyện trực tiếp
python demo_conversation.py
```

### 4. Đo Lường Hiệu Năng & Độ Trễ Đầu Cuối
```powershell
python evaluate_pcrs.py
```

---

## 📊 Kết Quả Thực Nghiệm Nổi Bật

1. **Thời gian đáp ứng thời gian thực (Latency Benchmark):**
   - Mã hóa vector (MiniLM): $P_{50} = 9.2\text{ ms}$.
   - Truy vấn ChromaDB (HNSW Cosine): $P_{50} = 14.9\text{ ms}$.
   - Suy luận gợi ý GNN+GCL: $P_{50} = 3.5\text{ ms}$.
   - Tổng độ trễ pipeline nội bộ (Mock): $17.0 \sim 24.0\text{ ms}$ (đạt chuẩn thời gian thực $< 100\text{ ms}$).
   - Gọi Cloud Gemini Flash API: $650 \sim 1100\text{ ms}$.

2. **Thực nghiệm Tìm kiếm Cá nhân hóa (Personalized vs Global Search):**
   - Với User có gu phim Crime rõ rệt (User 5), khi đưa ra các câu hỏi mở không chứa từ khóa thể loại (*"Dựa trên sở thích của tôi, phim nào nên thử?"*):
     - **Personalized Search (lọc theo `user_id = 5`):** Truy xuất chính xác các ngữ cảnh thể loại Crime (điểm tương đồng $0.664, 0.529, 0.540$).
     - **Global Search (không lọc user):** Bị thiên lệch bởi độ phổ biến chung của hệ thống (Popularity Bias), trả về các thể loại ngẫu nhiên như Drama ($0.710$) hoặc Mystery ($0.613$).
   - Minh chứng cho năng lực thấu cảm ngữ cảnh dài hạn vượt trội của PCRS.
