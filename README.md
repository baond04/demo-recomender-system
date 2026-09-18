# Nghiên Cứu Xây Dựng Trợ Lý Hội Thoại AI Cá Nhân Hóa Người Dùng Trong Quản Lý Nhà Hàng

> **Đề tài Đồ án Tốt nghiệp Đại học — Chuyên ngành Công nghệ Phần mềm (PTIT)**  
> **Mã số đề tài đăng ký trên hệ thống:** *"Nghiên cứu xây dựng trợ lý hội thoại AI cá nhân hóa người dùng trong quản lý nhà hàng"*  
> **Giảng viên hướng dẫn:** TS. Đỗ Thị Liên — Khoa Công nghệ Thông tin, Học viện Công nghệ Bưu chính Viễn thông (PTIT).  
> **Sinh viên thực hiện:**  
> - Nguyễn Đức Bảo — Mã SV: B22DCCN064 (Lớp D22CNPM02)  
> - Nguyễn Mạnh Cường — Mã SV: B22DCCN100 (Lớp D22CNPM02)  

---

## 📌 1. Tổng Quan Ý Tưởng & Kiến Trúc Hệ Thống

Hệ thống được thiết kế theo mô hình **Trợ lý Hội thoại Cá nhân hóa (Personalized Conversational Recommender System - PCRS)** tích hợp 3 trụ cột công nghệ chính: **Mạng Nơ-ron Đồ thị (GNN - LightGCN)**, **Học Tương phản Đồ thị (GCL - SimGCL)** và **Truy xuất Tăng cường Ngữ cảnh Đa phiên (RAG - Retrieval-Augmented Generation)**.

```
                                [USER QUERY q_t & USER ID u]
                                             │
          ┌──────────────────────────────────┴──────────────────────────────────┐
          ▼                                                                     ▼
 [NHÁNH 1A: RAG CONTEXT RETRIEVAL]                    [NHÁNH 1B: GNN+GCL DUAL RECOMMENDATION]
 • Bi-Encoder all-MiniLM-L6-v2 (384-dim)              • LightGCN K-layer Message Passing
 • ChromaDB Cosine Indexing (HNSW)                    • SimGCL Contrastive Learning (InfoNCE)
 • MMR Reranking khử trùng lặp                        • Dot Product Ranking: y_ui = e_u^T · e_i
 • Trả về: Top-K lượt đàm thoại tiền lệ               • Trả về: Top-N Recommended Items
          │                                                                     │
          └──────────────────────────────────┬──────────────────────────────────┘
                                             ▼
                             [BƯỚC 3: STRUCTURED PROMPT FUSION]
                             • Khối 1: Ngữ cảnh hội thoại liên quan từ RAG
                             • Khối 2: Danh sách Top-N gợi ý từ GNN+GCL
                             • Khối 3: Tóm tắt truy vấn hiện tại q_t
                             • Khóa chống ảo giác (Anti-Hallucination Guardrail)
                                             │
                                             ▼
                             [BƯỚC 4: LLM RESPONSE GENERATION]
                             • Google Gemini Flash API (hoặc Local Rule-based Mock)
                             • Sinh phản hồi tự nhiên, giải thích rõ ràng lý do gợi ý
                                             │
                                             ▼
                             [BƯỚC 5: PHẢN HỒI & CẬP NHẬT VECTOR DB]
```

### Quy trình xử lý 3 bước chuẩn mực:
1. **Bước 1 (Truy xuất ngữ cảnh RAG):** Khi người dùng gửi câu hỏi $q_t$, hệ thống thực hiện tìm kiếm ngữ nghĩa qua CSDL Vector (ChromaDB) để tìm các đoạn đàm thoại tiền lệ liên quan nhất, giúp hệ thống nắm bắt ngữ cảnh hội thoại dài hạn.
2. **Bước 2 (Gợi ý cá nhân hóa GNN+GCL):** Song song với Bước 1, hệ thống khai thác đồ thị tương tác lưỡng phân User-Item thông qua **LightGCN** và bổ sung tín hiệu tương phản **SimGCL** để trích xuất danh sách Top-N sản phẩm gợi ý phù hợp nhất với sở thích của người dùng $u$.
3. **Bước 3 (Dung hợp Prompt & Sinh phản hồi):** Tổng hợp Ngữ cảnh đàm thoại (RAG) + Danh sách Top-N gợi ý (GNN+GCL) + Câu hỏi hiện tại ($q_t$) vào một **Structured Prompt** chuẩn hóa để LLM (Gemini Flash) sinh câu trả lời tự nhiên, có lý giải rõ ràng và không bị ảo giác.

---

## 📂 2. Cấu Trúc Thư Mục Dự Án

```
.
├── teacher_request.txt             # Lịch sử yêu cầu và chỉ đạo chi tiết từ GVHD TS. Đỗ Thị Liên
├── requirements_rag.txt         # Danh sách dependencies thư viện Python
├── README.md                    # Tài liệu hướng dẫn và báo cáo kỹ thuật toàn bộ dự án
├── .gitignore                   # Cấu hình bỏ qua các file tạm, bytecode và dữ liệu nặng
│
├── src/                         # Mã nguồn chính của các thuật toán & Data Loaders
│   ├── data_loader.py           # Module nạp tổng hợp dữ liệu
│   ├── data_loader_amazon.py    # Loader cho Amazon Review Datasets
│   ├── data_loader_lastfm.py    # Loader cho Last.fm Music Dataset
│   ├── data_loader_tripadvisor.py # Loader cho TripAdvisor Hotel Reviews
│   ├── data_loader_yelp.py       # Loader cho Yelp2018 (SimGCL Benchmark)
│   ├── data_loader_taobao.py     # Loader cho Taobao User Behavior
│   ├── data_loader_netflix.py    # Loader cho Netflix Shows
│   ├── metrics.py               # Thư viện tính toán 7 độ đo: Precision, Recall, NDCG, MRR, HitRate, Coverage, Improvement
│   ├── model_cb.py              # Baseline Content-Based Filtering
│   ├── model_mf.py              # Baseline Matrix Factorization (BPR Loss)
│   ├── model_hybrid.py          # Baseline Late-Fusion Hybrid
│   ├── model_lightgcn.py        # Baseline LightGCN (GNN)
│   ├── model_lightgcn_gcl.py    # LightGCN + GCL (SGL & SimGCL Noise Perturbation)
│   ├── train_eval.py            # Vòng lặp huấn luyện và đánh giá mô hình
│   │
│   └── rag_conversational/      # Gói module RAG & Trợ lý Hội thoại
│       ├── __init__.py
│       ├── conversational_recommender.py  # Điều phối pipeline đầu cuối
│       ├── conversation_store.py          # Quản lý CSDL Vector ChromaDB
│       ├── rag_retriever.py               # Tìm kiếm ngữ nghĩa & MMR Reranking
│       ├── gnn_gcl_adapter.py             # Adapter trích xuất gợi ý từ GNN
│       ├── prompt_builder.py              # Xây dựng prompt 3 thành phần chuẩn hóa
│       ├── response_generator.py          # Sinh phản hồi qua Gemini Flash / Mock
│       └── session_manager.py             # Quản lý phiên đàm thoại người dùng
│
├── ml-latest-small/             # Bộ dữ liệu MovieLens 100K
├── hetrec2011-lastfm-2k/        # Bộ dữ liệu Last.fm HetRec 2011
├── reviews_Musical_Instruments_5.json/ # Bộ dữ liệu Amazon Musical Instruments
├── TripAdvisor_Dataset/         # Bộ dữ liệu TripAdvisor Hotel Reviews (.parquet)
├── Yelp_Dataset/                # Bộ dữ liệu Yelp2018 (.txt)
├── Taobao_Dataset/              # Bộ dữ liệu Taobao User Behavior (.parquet)
├── Netflix_Dataset/             # Bộ dữ liệu Netflix Shows (.parquet)
├── checkpoints/                 # Trọng số mô hình đã huấn luyện (simgcl_best.pth)
│
├── run_all_experiments.py       # Chạy thực nghiệm so sánh 6 mô hình trên TOÀN BỘ 7 DỮ LIỆU
├── evaluate_pcrs.py             # Đo lường độ trễ và đánh giá hiệu năng PCRS
├── train_and_save_simgcl.py     # Huấn luyện & lưu checkpoint SimGCL
├── demo_conversation.py         # Giao diện chạy thử nghiệm đàm thoại PCRS
└── create_pcrs_architecture_diagram.py # Script tự động sinh sơ đồ kiến trúc
```

---

## 📊 3. Danh Sách 7 Bộ Dữ Liệu Thực Nghiệm (Multi-Dataset Benchmark)

Nhằm đáp ứng yêu cầu của giảng viên hướng dẫn về việc thực nghiệm đa dạng trên nhiều miền ứng dụng khác nhau:

| STT | Bộ Dữ Liệu | Miền Ứng Dụng | Định Dạng | Số Users | Số Items | Số Tương Tác |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | **MovieLens 100K** (`ml-latest-small`) | Phim ảnh | CSV / Text | 609 | 9,742 | 81,763 |
| **2** | **Last.fm** (`hetrec2011-lastfm-2k`) | Âm nhạc / Nghệ sĩ | DAT / Text | 1,892 | 17,632 | 92,834 |
| **3** | **Amazon Reviews** (`Musical_Instruments_5`) | Thương mại điện tử | JSON | 1,429 | 900 | 9,794 |
| **4** | **TripAdvisor Dataset** | Đánh giá Khách sạn / Du lịch | Parquet | 500 | 14,206 | 13,624 |
| **5** | **Yelp Dataset** (`yelp2018`) | Đánh giá Nhà hàng / Dịch vụ | Text (train/test) | 31,668 | 38,048 | 1,561,406 |
| **6** | **Ecommerce Taobao** (`UserBehavior`) | Mua sắm Thương mại điện tử | Parquet | 1,300 | 17 | 68,900 |
| **7** | **Netflix Prize Dataset** | Phim ảnh & Chương trình | Parquet | 400 | 8,807 | 8,800 |

---

## 🤖 4. Các Phương Pháp Thực Nghiệm & Giải Pháp Kết Hợp (Combination Solutions)

Theo chỉ đạo của Giảng viên hướng dẫn, đồ án tập trung phát triển các **giải pháp kết hợp đa mô hình (Hybrid & Ensemble Solutions)** từ mức độ cơ bản đến nâng cao:

### 🔹 Các Giải Pháp Kết Hợp (Combination / Hybrid Models):
1. **Cấp độ 1 — Late-Fusion Hybrid Baseline (MF + CB):** Kết hợp ma trận phân rã nhân tố ẩn $S_{\text{MF}}$ (Collaborative Filtering) và ma trận tương đồng đặc trưng văn bản $S_{\text{CB}}$ (Content-Based) với trọng số Late-Fusion $\alpha = 0.6$:
   $$S_{\text{Hybrid}} = \alpha \cdot \tilde{S}_{\text{MF}} + (1 - \alpha) \cdot \tilde{S}_{\text{CB}}$$
2. **Cấp độ 2 — Dual Graph Convolution & Contrastive Learning (LightGCN + SimGCL):** Kết hợp truyền thông điệp đồ thị lưỡng phân User-Item $K=2$ tầng (LightGCN) và hàm mất mát tương phản InfoNCE với nhiễu đồng nhất $\epsilon=0.1$ (SimGCL) để giải quyết triệt để vấn đề thưa dữ liệu.
3. **Cấp độ 3 — Advanced Hybrid (SimGCL + CB):** Giải pháp kết hợp nâng cao dung hợp vector biểu diễn đồ thị tương phản (SimGCL) và ma trận đặc trưng nhận xét sản phẩm (CB TF-IDF) giúp hệ thống gợi ý chính xác ngay cả với các item mới phát sinh (Cold-Start Items).
4. **Cấp độ 4 — Toàn vẹn PCRS (GNN + GCL + RAG + LLM):** Giải pháp cao nhất của đồ án kết hợp 2 nhánh song song:
   - **Nhánh 1:** Truy xuất ngữ cảnh đàm thoại qua ChromaDB Vector Database (RAG).
   - **Nhánh 2:** Gợi ý sản phẩm cá nhân hóa từ mô hình SimGCL+CB (GNN+GCL+Content).
   - **Dung hợp (Prompt Fusion):** Ghép ngữ cảnh đàm thoại + Danh sách gợi ý Top-$N$ + Câu hỏi hiện tại vào Structured Prompt cho Gemini Flash LLM sinh phản hồi tư nhiên và chính xác.

---

### 🔹 Danh Sách Các Mô Hình Đối Chứng & Kết Hợp Thực Nghiệm:
1. **Content-Based Filtering (CB)**
2. **Matrix Factorization (MF - BPR)**
3. **Hybrid Baseline (MF + CB)**
4. **LightGCN Baseline (GNN)**
5. **LightGCN + GCL (SGL - Structure Augmentation)**
6. **SimGCL (Simple Graph Contrastive Learning)**
7. **Advanced Hybrid (SimGCL + CB)**


---

## 📈 5. Các Độ Đo Đánh Giá Tiêu Chuẩn (7 Metrics)

Độ đo thực nghiệm được xây dựng chuẩn mực theo quy chuẩn đánh giá hệ thống học máy Top-$K$ ($K=10, 20$):

1. **Precision@K:** Tỷ lệ sản phẩm gợi ý đúng trong số $K$ sản phẩm đề xuất.
2. **Recall@K:** Tỷ lệ sản phẩm đúng được truy xuất trên tổng số sản phẩm người dùng thực sự thích ở tập Test.
3. **NDCG@K (Normalized Discounted Cumulative Gain):** Đánh giá chất lượng thứ hạng xếp hạng (sản phẩm đúng xếp vị trí càng cao điểm càng lớn).
4. **MRR@K (Mean Reciprocal Rank):** Nghịch đảo vị trí của sản phẩm đúng đầu tiên trong danh sách gợi ý.
5. **HitRate@K:** Tỷ lệ phần trăm người dùng nhận được *ít nhất 1 sản phẩm đúng* trong Top-$K$.
6. **Coverage@K:** Tỷ lệ phủ sản phẩm được mang đi gợi ý trên toàn bộ danh mục sản phẩm của hệ thống.
7. **Improvement (%):** Tỷ lệ phần trăm cải thiện hiệu năng của **SimGCL** so với Baseline **LightGCN**:
   $$\text{Improvement (\%)} = \frac{\text{Metric}_{\text{SimGCL}} - \text{Metric}_{\text{LightGCN}}}{\text{Metric}_{\text{LightGCN}}} \times 100\%$$

---

## 🚀 6. Hướng Dẫn Cài Đặt và Khởi Chạy

### 1. Cài đặt Môi trường Python
```powershell
pip install -r requirements_rag.txt
```

### 2. Chạy Thực Nghiệm So Sánh Toàn Bộ 6 Mô Hình Trên 7 Bộ Dữ Liệu
```powershell
python run_all_experiments.py
```
*Lệnh này sẽ tự động nạp 7 datasets, huấn luyện và đánh giá 6 mô hình, xuất bảng so sánh chi tiết và lưu kết quả ra file `experiment_results.json`.*

### 3. Huấn Luyện và Lưu Checkpoint SimGCL
```powershell
python train_and_save_simgcl.py
```

### 4. Khởi Chạy Thử Nghiệm Trợ Lý Hội Thoại (Interactive Demo)
```powershell
# Chế độ kiểm thử nhanh với Offline Mock (< 25ms latency)
python demo_conversation.py --test-mode

# Chế độ tương tác trực tiếp
python demo_conversation.py
```

### 5. Đánh Giá Độ Trễ Đầu Cuối Pipeline PCRS
```powershell
python evaluate_pcrs.py
```

---

## 📄 7. Trích Dẫn Yêu Cầu Của Giảng Viên ( trích `teacher_request.txt` )

> *"Các em đăng ký tên đề tài trên hệ thống tốt nghiệp với tên: 'Nghiên cứu xây dựng trợ lý hội thoại AI cá nhân hóa người dùng trong quản lý nhà hàng'"*  
> *"Kết hợp GNN+GCL+RAG cho bài toán Personalized conversational recommendations... Khi người dùng gửi một câu hỏi, hệ thống thực hiện tìm kiếm ngữ nghĩa qua CSDL vector... Đồng thời bước 1 khai thác thông tin tư vấn sản phẩm theo mô hình GNN+GCL... Cuối cùng dung hợp vào Prompt có cấu trúc cho LLM sinh phản hồi."* — **TS. Đỗ Thị Liên**
