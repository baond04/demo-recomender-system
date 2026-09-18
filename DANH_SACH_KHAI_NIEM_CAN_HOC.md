# BÁO CÁO GIẢI THÍCH CHI TIẾT TẤT CẢ CÁC KHÁI NIỆM & QUY TRÌNH HỆ GỢI Ý (PCRS: GNN + GCL + RAG + LLM)

> **Tài liệu hướng dẫn chuyên sâu 100% dành cho người mới bắt đầu từ số 0.**  
> **Phong cách trình bày:** Có định nghĩa trực quan, công thức toán học, sơ đồ ASCII, ví dụ số cụ thể và luồng xử lý từng bước (Step-by-Step Execution Flow).

---

## 🗺️ MỤC LỤC TỔNG QUAN

1. **CHƯƠNG 1: LLM LÀ GÌ & TẠI SAO BỊ ẢO GIÁC?**
2. **CHƯƠNG 2: RAG LÀ GÌ & QUY TRÌNH 4 BƯỚC TRUY XUẤT NGỮ CẢNH**
3. **CHƯƠNG 3: HỆ GỢI Ý TRUYỀN THỐNG (CB, CF, MF, BPR LOSS, HYBRID)**
4. **CHƯƠNG 4: MẠNG NƠ-RON ĐỒ THỊ (GNN & LIGHTGCN) — PHÂN TÍCH TỪNG TẦNG**
5. **CHƯƠNG 5: HỌC TƯƠNG PHẢN ĐỒ THỊ (GCL, SGL, SIMGCL & INFONCE LOSS)**
6. **CHƯƠNG 6: CẤU TRÚC PROMPT 5 KHỐI & CƠ CHẾ CHỐNG ẢO GIÁC**
7. **CHƯƠNG 7: BỘ CHỈ SỐ ĐÁNH GIÁ (PRECISION, RECALL, NDCG, MRR, COVERAGE)**
8. **CHƯƠNG 8: QUY TRÌNH XỬ LÝ TOÀN CỤC ĐẦU CUỐI (END-TO-END WALKTHROUGH TESTCASE)**
9. **CHƯƠNG 9: BẢNG TRA CỨU THUẬT NGỮ & BỘ CÂU HỎI BẢO VỆ ĐỒ ÁN**

---

## CHƯƠNG 1: MÔ HÌNH NGÔN NGỮ LỚN (LLM) LÀ GÌ & TẠI SAO BỊ ẢO GIÁC?

### 1.1. LLM (Large Language Model) là gì?
* **Khái niệm:** Mô hình Ngôn ngữ Lớn (Large Language Model - LLM) là một mạng nơ-ron sâu được huấn luyện trên hàng tỷ văn bản từ Internet. LLM có khả năng hiểu, tóm tắt, dịch thuật và sinh văn bản tự nhiên giống hệt như con người.
* **Các ví dụ LLM phổ biến:** Google Gemini, OpenAI ChatGPT (GPT-4), Meta LLaMA. Trong đề tài này sử dụng mô hình **Google Gemini Flash**.

```
[Văn bản đầu vào (Prompt)] ──► [Mô hình LLM (Gemini)] ──► [Văn bản phản hồi tự nhiên]
```

### 1.2. Nguyên lý sinh từ của LLM (Autoregressive Generation)
LLM không "suy nghĩ" hay "tra cứu cơ sở dữ liệu" như con người. Bản chất toán học của LLM là **dự đoán xác suất của từ tiếp theo (Token)** dựa trên các từ đứng trước:
$$P(y_1, y_2, \dots, y_L | \text{Prompt}) = \prod_{t=1}^{L} P(y_t | y_1, y_2, \dots, y_{t-1}, \text{Prompt})$$

### 1.3. Tại sao LLM lại bị hiện tượng ẢO GIÁC (Hallucination)?
* **Ảo giác (Hallucination) là gì?** Là hiện tượng LLM tự bịa ra thông tin sai sự thật, tự bịa tên bộ phim không có thực hoặc gán sai đạo diễn/diễn viên một cách cực kỳ tự tin.
* **Nguyên nhân:**
  1. LLM sinh từ dựa trên **tương quan xác suất thống kê văn bản**, không phải tra cứu cơ sở dữ liệu thực (Ground Truth).
  2. LLM không có quyền truy cập vào ma trận lịch sử xem phim thực tế của từng người dùng cụ thể.
  3. Tri thức của LLM bị đóng bế tại thời điểm hoàn tất huấn luyện (*Knowledge Cutoff*).
* **Nhiệm vụ của LLM trong đề tài này:**  
  👉 **LLM KHÔNG ĐƯỢC PHÉP tự chọn phim để gợi ý.**  
  👉 **LLM chỉ đóng vai trò là "Giao diện đàm thoại" (Conversational Interface)**: Nhận danh sách phim do GNN/SimGCL chọn sẵn và diễn đạt lại thành lời khuyên thân thiện bằng tiếng Việt.

---

## CHƯƠNG 2: RAG (RETRIEVAL-AUGMENTED GENERATION) LÀ GÌ & QUY TRÌNH 4 BƯỚC?

### 2.1. RAG là gì và Tại sao phải dùng RAG?
* **Khái niệm:** RAG (*Retrieval-Augmented Generation* — Truy xuất tăng cường ngữ cảnh) là kỹ thuật kết hợp giữa **Bộ tìm kiếm thông tin (Retriever)** và **Mô hình sinh (Generator/LLM)**.
* **Tại sao cần RAG?**  
  Khi người dùng hỏi: *"Hôm nay tôi mệt quá, có phim nào giống bộ hôm trước bạn tư vấn cho tôi không?"*, LLM thuần túy sẽ không biết "bộ hôm trước" là bộ nào. RAG sẽ đóng vai trò như một "trợ lý tra cứu" bay vào Cơ sở dữ liệu Vector để tìm lại đúng đoạn hội thoại quá khứ đó và nạp vào cho LLM đọc.

```
                                 [CÂU HỎI NGƯỜI DÙNG q_t]
                                            │
                                            ▼
                           [BƯỚC 1: NHỦNG VECTOR (MINILM)]
                                            │
                                            ▼
                           [BƯỚC 2: TRA CỨU CSDL VECTOR (CHROMADB)]
                                            │
                                            ▼
                           [BƯỚC 3: TÁI XẾP HẠNG ĐA DẠNG (MMR)]
                                            │
                                            ▼
                           [BƯỚC 4: NHÓI NGỮ CẢNH VÀO PROMPT] ──► [GỬI LLM (GEMINI)]
```

---

### 2.2. Quy trình RAG 4 bước chi tiết kèm Ví dụ minh họa

#### 🔹 BƯỚC 1: Mã hóa Vector Ngữ nghĩa (Dense Embedding)
* **Thành phần:** Mô hình Bi-Encoder `all-MiniLM-L6-v2`.
* **Nhiệm vụ:** Biến câu văn tự nhiên $q_t$ thành một chuỗi gồm 384 số thực (Vector 384 chiều $\mathbb{R}^{384}$).
* **Ví dụ:**  
  * Câu hỏi: `"Tôi muốn xem phim hành động tội phạm"` $\rightarrow$ Vector $v_q = [0.12, -0.45, 0.89, \dots, 0.03]$.

#### 🔹 BƯỚC 2: Tìm kiếm Tương đồng trong CSDL Vector (ChromaDB & HNSW Index)
* **Thành phần:** Cơ sở dữ liệu Vector `ChromaDB` dùng chỉ mục **HNSW (Hierarchical Navigable Small World)**.
* **Nhiệm vụ:** Tính độ tương đồng Cosine giữa vector câu hỏi $v_q$ với tất cả các vector lượt đàm thoại mẫu lưu trong DB:
  $$\text{CosineSimilarity}(v_q, v_d) = \frac{v_q \cdot v_d}{\|v_q\|_2 \|v_d\|_2}$$
* **Ví dụ số:**
  * Lượt đàm thoại A: *"Phim tội phạm trí tuệ đấu trí?"* $\rightarrow$ Độ tương đồng = $0.78$ ($78\%$).
  * Lượt đàm thoại B: *"Phim hoạt hình thiếu nhi?"* $\rightarrow$ Độ tương đồng = $0.12$ ($12\%$).
  * ChromaDB chọn Lượt A vì độ tương đồng cao.

#### 🔹 BƯỚC 3: Tái xếp hạng đa dạng bằng Thuật toán MMR (Maximal Marginal Relevance)
* **Thách thức:** Nếu chỉ chọn theo độ tương đồng Cosine, ChromaDB có thể trả về 5 câu hội thoại giống hệt nhau về nội dung (bị trùng lặp ý).
* **Giải pháp MMR:** Cân bằng giữa **Độ khớp ngữ nghĩa** và **Độ đa dạng nội dung**:
  $$\text{MMR} = \arg\max_{d_i \in \mathcal{D} \setminus \mathcal{S}} \left[ \alpha \cdot \text{sim}(q_t, d_i) - (1 - \alpha) \max_{d_j \in \mathcal{S}} \text{sim}(d_i, d_j) \right]$$
  * $\alpha = 0.8$: Ưu tiên $80\%$ độ khớp và $20\%$ độ đa dạng nội dung.

#### 🔹 BƯỚC 4: Tiêm Ngữ cảnh vào Prompt (Context Injection)
Đóng gói các đoạn hội thoại tìm được thành văn bản chuẩn hóa để gửi kèm cho LLM.

---

## CHƯƠNG 3: CÁC KHÁI NIỆM HỆ GỢI Ý TRUYỀN THỐNG (BASELINES)

### 3.1. Content-Based Filtering (CB - Lọc theo nội dung)
* **Ý tưởng:** Gợi ý sản phẩm dựa trên thuộc tính (Thể loại, Đạo diễn, Tag).
* **Công thức TF-IDF (Term Frequency - Inverse Document Frequency):**
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
* **Tạo User Profile ($\mathbf{p}_u$):** Trung bình cộng vector thuộc tính các phim người dùng $u$ đã thích:
  $$\mathbf{p}_u = \frac{1}{|\mathcal{N}_u|} \sum_{i \in \mathcal{N}_u} \mathbf{v}_i$$

---

### 3.2. Matrix Factorization (MF - Phân rã ma trận)
* **Ý tưởng:** Nén ma trận tương tác $\mathbf{R}$ ($M \times N$) thành tích 2 ma trận ẩn $E_U$ ($M \times d$) và $E_I$ ($N \times d$).

```
  Ma trận thưa R (User x Item)  ≈  User Embedding E_U  ×  Item Embedding E_I^T
   [ 5  .  .  1 ]                   [ 0.8  0.2 ]
   [ .  4  .  . ]          ≈        [ 0.1  0.9 ]   ×   [ 0.9  0.3  0.1  0.2 ]
   [ 2  .  5  . ]                   [ 0.4  0.7 ]       [ 0.2  0.8  0.9  0.1 ]
```

* **Công thức điểm dự đoán:** $\hat{y}_{ui} = \mathbf{e}_u^T \cdot \mathbf{e}_i = \sum_{k=1}^d e_{u,k} \cdot e_{i,k}$.
* **Hàm BPR Loss (Bayesian Personalized Ranking):**
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda_{\text{reg}} \|\Theta\|_2^2$$
  * $i$: Item dương (User đã xem). $j$: Item âm (User chưa xem - Negative sampling).

---

### 3.3. Late-Fusion Hybrid System (Hệ tư vấn Lai)
* **Công thức chuẩn hóa Min-Max Scaling:**
  $$\tilde{y}_{ui} = \frac{\hat{y}_{ui} - \min(\hat{\mathbf{Y}})}{\max(\hat{\mathbf{Y}}) - \min(\hat{\mathbf{Y}})}$$
* **Công thức kết hợp có trọng số $\alpha$:**
  $$S_{\text{Hybrid}}(u, i) = \alpha \cdot \tilde{y}_{ui}^{(\text{MF})} + (1 - \alpha) \cdot \tilde{y}_{ui}^{(\text{CB})}$$

---

## CHƯƠNG 4: MẠNG NƠ-RON ĐỒ THỊ (GNN & LIGHTGCN) — PHÂN TÍCH TỪNG TẦNG

### 4.1. Đồ thị lưỡng phân User-Item (Bipartite Graph)
Đồ thị $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ gồm tập nút Người dùng $\mathcal{U}$ và tập nút Sản phẩm $\mathcal{I}$.

```
      USERS (U)                      ITEMS (I)
     [User u1] ────────────────────── [Phim i1]
         │    \                      /
         │     \                    /
         │      ────────── [Phim i2] ───
     [User u2] ────────────────────── [Phim i3]
```

---

### 4.2. Phân tích chi tiết từng tầng lan truyền trong LightGCN

#### 🔹 Tầng 0 ($k=0$): Khởi tạo Embedding Ban đầu
Khởi tạo ngẫu nhiên cho tất cả các nút $M+N$: $\mathbf{e}_u^{(0)}, \mathbf{e}_i^{(0)} \sim \mathcal{N}(0, 0.1)$. Tầng này chỉ chứa thông tin cá thể độc lập.

#### 🔹 Tầng 1 ($k=1$): Lan truyền 1-hop
Nút User gom tụ thông tin từ các Phim chính mình từng xem:
$$\mathbf{e}_{u1}^{(1)} = \frac{1}{\sqrt{|\mathcal{N}_{u1}||\mathcal{N}_{i1}|}} \mathbf{e}_{i1}^{(0)} + \frac{1}{\sqrt{|\mathcal{N}_{u1}||\mathcal{N}_{i2}|}} \mathbf{e}_{i2}^{(0)}$$

#### 🔹 Tầng 2 ($k=2$): Lan truyền 2-hop (Khai phóng Tín hiệu Lọc cộng tác)
Nút User $u_1$ nhận được thông tin từ User $u_2$ thông qua phim xem chung $i_2$ ($u_1 \rightarrow i_2 \rightarrow u_2$). Đây chính là **tín hiệu lọc cộng tác ngầm trên đồ thị**.

#### 🔹 Tầng $k \ge 4$: Hiện tượng Quá mượt (Over-smoothing)
Nếu lan truyền quá sâu, tất cả các vector trong đồ thị bị san bằng và giống hệt nhau. Do đó LightGCN chỉ dừng ở $K=2$ hoặc $K=3$.

---

### 4.3. Tại sao LightGCN loại bỏ ma trận trọng số $W$ và ReLU?
* Trong GNN truyền thống (GCN, NGCF):
  $$\mathbf{e}_u^{(k+1)} = \text{ReLU}\left( W^{(k)} \cdot \text{Aggregate}(\mathbf{e}_i^{(k)}) \right)$$
* Các tác giả LightGCN chứng minh: $W$ và $\text{ReLU}$ khiến mô hình bị quá khớp (overfitting) và khó huấn luyện trong đồ thị gợi ý. LightGCN chỉ giữ lại duy nhất **phép gom tụ chuẩn hóa đối xứng**:
  $$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u| |\mathcal{N}_i|}} \mathbf{e}_i^{(k)}$$

#### Phép tổng hợp đa tầng Mean Pooling:
$$\mathbf{e}_u^{(\text{final})} = \frac{1}{K+1} \sum_{k=0}^{K} \mathbf{e}_u^{(k)}, \quad \mathbf{e}_i^{(\text{final})} = \frac{1}{K+1} \sum_{k=0}^{K} \mathbf{e}_i^{(k)}$$

---

## CHƯƠNG 5: HỌC TƯƠNG PHẢN ĐỒ THỊ (GCL, SGL, SIMGCL & INFONCE LOSS)

### 5.1. Tại sao phải Học Tương Phản (GCL)?
Khi ma trận thưa thớt cực độ (như dữ liệu Last.fm thưa **$99.78\%$**), LightGCN không có đủ cạnh để truyền thông điệp. Học tương phản (Graph Contrastive Learning - GCL) áp dụng **Học tự giám sát (Self-Supervised Learning)** để tự tạo nhãn phụ giúp huấn luyện vector chắc chắn.

---

### 5.2. So sánh SGL vs SimGCL

```
[Mô hình SGL]    Đồ thị G ──► Edge/Node Dropout ──► 2 Views G1, G2 (Tốn chi phí tính toán)
[Mô hình SimGCL] Vector z ──► Cộng nhiễu ngẫu nhiên ──► 2 Views z1, z2 (Chạy siêu tốc)
```

* **Công thức SimGCL:**
  $$\mathbf{z}' = \mathbf{z} + \epsilon \cdot \text{sign}(\mathbf{\Delta}), \quad \mathbf{\Delta} \sim \mathcal{N}(0, \mathbf{I})$$
  (với $\epsilon = 0.1$ là cường độ nhiễu).

---

### 5.3. Hàm Mất Mát InfoNCE Loss ($\mathcal{L}_{\text{CL}}$)
* **Positive Pair (Cặp tích cực):** Cùng 1 nút $u$ qua 2 views $\mathbf{z}_u^{(1)}$ và $\mathbf{z}_u^{(2)}$ $\rightarrow$ **Kéo lại gần nhau**.
* **Negative Pairs (Cặp tiêu cực):** Nút $u$ với các nút $v \neq u$ khác $\rightarrow$ **Đẩy ra xa nhau**.

$$\mathcal{L}_{\text{CL}} = -\sum_{u \in \mathcal{U}} \log \frac{\exp\left(\frac{\text{sim}(\mathbf{z}_u^{(1)}, \mathbf{z}_u^{(2)})}{\tau}\right)}{\sum_{v \in \mathcal{U}} \exp\left(\frac{\text{sim}(\mathbf{z}_u^{(1)}, \mathbf{z}_v^{(2)})}{\tau}\right)}$$
* $\tau = 0.2$: Tham số nhiệt độ (Temperature parameter).

#### Hàm mục tiêu toàn cục Joint Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{cl}} \cdot \mathcal{L}_{\text{CL}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$

---

## CHƯƠNG 6: CẤU TRÚC PROMPT 5 KHỐI & CƠ CHẾ CHỐNG ẢO GIÁC

Để khóa hoàn toàn việc LLM bịa đặt tên phim, tệp `prompt_builder.py` đóng gói Prompt theo 5 khối kỹ thuật:

```text
================================================================================
STRUCTURED PROMPT GỬI GOOGLE GEMINI (BẢN NGUYÊN VĂN KHÓA ẢO GIÁC)
================================================================================
[KHỐI 1: SYSTEM PERSONA & KHÓA ẢO GIÁC (ANTI-HALLUCINATION GUARDRAIL)]
Bạn là trợ lý tư vấn phim ảnh thông minh. Nhiệm vụ của bạn là trả lời tự nhiên.
⚠️ QUY TẮC TỐI CAO: CHỈ ĐƯỢC PHÉP ĐỀ XUẤT CÁC PHIM NẰM TRONG KHỐI 4 DƯỚI ĐÂY.
NGHIÊM CẤM TỰ Ý BỊA TÊN PHIM KHÔNG CÓ TRONG DANH SÁCH.

[KHỐI 2: LỊCH SỬ GẦN ĐÂY CỦA NGƯỜI DÙNG]
  • Người dùng thích thể loại trinh thám kịch tính.

[KHỐI 3: NGỮ CẢNH HỘI THOẠI LIÊN QUAN TỪ RAG (CHROMADB)]
  • Lượt 1 (Khớp 78%): "Tìm phim đấu trí căng thẳng?" -> Trợ lý đề xuất Usual Suspects.

[KHỐI 4: DANH SÁCH GỢI Ý CÁ NHÂN HÓA TỪ SIM GCL (CLOSED KNOWLEDGE BOUNDARY)]
  1. • Usual Suspects, The (1995) [Score: 0.892] [Genre: Crime, Mystery, Thriller]
  2. • Pulp Fiction (1994) [Score: 0.865] [Genre: Comedy, Crime, Drama]
  3. • Fargo (1996) [Score: 0.841] [Genre: Comedy, Crime, Drama, Thriller]

[KHỐI 5: TRUY VẤN HIỆN TẠI & NHIỆM VỤ BẮT BUỘC]
  💬 "Hôm nay tôi muốn đổi gió xem phim hành động tội phạm kịch tính, giúp tôi với"
  📝 Nhiệm vụ: Chọn 3-5 phim từ KHỐI 4 và giải thích ngắn gọn lý do.
================================================================================
```

---

## CHƯƠNG 7: BỘ CHỈ SỐ ĐÁNH GIÁ TIÊU CHUẨN (EVALUATION METRICS)

Giả sử tập Test người dùng $u$ thích **5 bộ phim**: $\{A, B, C, D, E\}$.  
Hệ thống gợi ý **Top 10 phim**: $[A, X, B, Y, Z, C, M, N, O, P]$ (Đoán trúng 3 phim: $A, B, C$).

### 7.1. Precision@10 (Độ chính xác)
$$\text{Precision}@10 = \frac{\text{Số phim trúng}}{K} = \frac{3}{10} = 0.30 \quad (30\%)$$

### 7.2. Recall@10 (Độ bao phủ)
$$\text{Recall}@10 = \frac{\text{Số phim trúng}}{\text{Tổng phim Test thích}} = \frac{3}{5} = 0.60 \quad (60\%)$$

### 7.3. NDCG@10 (Normalized Discounted Cumulative Gain)
* **DCG@10:** Phạt vị trí đoán đúng bằng $\log_2(r+1)$:
  * Phim $A$ đúng ở vị trí #1 $\rightarrow 1 / \log_2(2) = 1.000$
  * Phim $B$ đúng ở vị trí #3 $\rightarrow 1 / \log_2(4) = 0.500$
  * Phim $C$ đúng ở vị trí #6 $\rightarrow 1 / \log_2(7) = 0.356$
  * $\text{DCG}@10 = 1.000 + 0.500 + 0.356 = 1.856$.
* **IDCG@10 (Lý tưởng nhất khi $A, B, C$ đứng ở vị trí #1, #2, #3):**
  * $\text{IDCG}@10 = 1/\log_2(2) + 1/\log_2(3) + 1/\log_2(4) = 1.000 + 0.631 + 0.500 = 2.131$.
* **$\text{NDCG}@10 = \frac{\text{DCG}}{\text{IDCG}} = \frac{1.856}{2.131} = 0.871$.**

### 7.4. MRR@10 (Mean Reciprocal Rank)
Nghịch đảo vị trí phim đúng đầu tiên ($A$ nằm ở vị trí #1): $\text{MRR} = \frac{1}{1} = 1.00$.

### 7.5. Coverage@10 (Độ bao phủ kho sản phẩm)
$$\text{Coverage}@10 = \frac{|\text{Tập các phim unique đem gợi ý}|}{N_{\text{tổng phim hệ thống}}}$$

---

## CHƯƠNG 8: QUY TRÌNH XỬ LÝ TOÀN CỤC ĐẦU CUỐI (END-TO-END WALKTHROUGH TESTCASE)

Minh họa luồng chạy thực tế 5 bước cho 1 câu hỏi cụ thể trong mã nguồn:

```
[BƯỚC 1: ĐẦU VÀO] User 5 hỏi: "Hôm nay tôi muốn xem phim hành động tội phạm kịch tính"
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[BƯỚC 2A: RAG RETRIEVAL (15.4ms)]        [BƯỚC 2B: SIMGCL INFERENCE (3.2ms)]
- MiniLM mã hóa vector 384-dim           - Trích xuất user embedding e_{u=5}
- ChromaDB Cosine Search 5 turns         - Nhân ma trận scores = e_u · E_I^T
- MMR Rerank lọc trùng lặp               - Train Masking: Gán -∞ cho phim đã xem
- Trả về: 2 lượt đàm thoại mẫu           - Trả về: Top-5 phim (Usual Suspects, Pulp Fiction...)
       │                                          │
       └────────────────────┬─────────────────────┘
                            ▼
               [BƯỚC 3: PROMPT FUSION (0.8ms)]
               - Ghép Context RAG + Top-5 SimGCL + System Persona
               - Tạo Prompt 5 khối có cấu trúc (450 tokens)
                            │
                            ▼
               [BƯỚC 4: GEMINI GENERATION (712ms)]
               - Gemini nhận prompt, trả lời tiếng Việt tự nhiên
               - Bộ Regex Hậu kiểm ảo giác: Tỷ lệ ảo giác = 0.0%
                            │
                            ▼
               [BƯỚC 5: LƯU TRỮ SESSION]
               - Ghi nhận ConversationTurn mới vào ChromaDB
```

---

## CHƯƠNG 9: BẢNG TRA CỨU THUẬT NGỮ & BỘ CÂU HỎI BẢO VỆ ĐỒ ÁN

### 📋 Bảng Thuật Ngữ Anh - Việt Cheatsheet

| Thuật ngữ Tiếng Anh | Viết tắt | Dịch nghĩa Tiếng Việt |
| :--- | :--- | :--- |
| **Large Language Model** | LLM | Mô hình Ngôn ngữ Lớn (như Gemini, GPT-4) |
| **Retrieval-Augmented Generation**| RAG | Truy xuất tăng cường ngữ cảnh cho LLM |
| **Graph Neural Network** | GNN | Mạng nơ-ron đồ thị |
| **LightGCN** | LightGCN | Mạng tích chập đồ thị nhẹ |
| **Graph Contrastive Learning** | GCL | Học tương phản trên đồ thị (Tự giám sát) |
| **SimGCL** | SimGCL | Học tương phản bằng cách thêm nhiễu vào nhúng |
| **Bipartite Graph** | — | Đồ thị lưỡng phân User-Item |
| **Message Passing** | — | Lan truyền thông điệp giữa các nút lân cận |
| **InfoNCE Loss** | $\mathcal{L}_{\text{CL}}$ | Hàm mất mát tương phản Positive/Negative Pairs |
| **Maximal Marginal Relevance** | MMR | Thuật toán tái xếp hạng cân bằng khớp & đa dạng |
| **Anti-Hallucination Guardrail** | — | Rào chắn chống hiện tượng LLM bịa đặt thông tin |
| **Precision / Recall / NDCG** | Metrics | Bộ chỉ số đánh giá chất lượng gợi ý |

---

### ❓ Bộ Câu Hỏi Bảo Vệ Đồ Án & Đáp Án Chuẩn

1. **Hỏi:** *Tại sao lại dùng LightGCN mà không dùng GCN truyền thống?*  
   **Đáp:** Vì GCN truyền thống dùng ma trận trọng số $W$ và hàm phi tuyến ReLU gây quá khớp và khó huấn luyện trên đồ thị gợi ý. LightGCN cắt bỏ $W$ và ReLU, chỉ giữ lại lan truyền đối xứng chuẩn hóa, đạt Recall/NDCG cao hơn rõ rệt.
2. **Hỏi:** *Học tương phản SimGCL giúp ích gì cho hệ thống?*  
   **Đáp:** Giúp giải quyết bài toán ma trận thưa cực độ ($99.78\%$) và nhiễu tương tác. SimGCL tự thêm nhiễu vào embedding và dùng InfoNCE Loss để tạo không gian biểu diễn nhúng phân bố đều (*Uniformity*), tăng NDCG@10 lên $+14.38\%$.
3. **Hỏi:** *Làm sao chống hiện tượng LLM bịa đặt tên phim (Ảo giác)?*  
   **Đáp:** Bằng kiến trúc 3 lớp: Tách rời nhiệm vụ chọn phim cho SimGCL, đóng gói Prompt 5 khối với "Biên giới tri thức đóng", và dùng bộ hậu kiểm Regex xác nhận $100\%$ tựa phim sinh ra nằm trong danh sách SimGCL.

---
*Tài liệu hướng dẫn hoàn chỉnh được cập nhật đầy đủ sơ đồ, ví dụ số và công thức toán học — Chúc bạn bảo vệ Đồ án Tốt nghiệp đạt kết quả xuất sắc!*
