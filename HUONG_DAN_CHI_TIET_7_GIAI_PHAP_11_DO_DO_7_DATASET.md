# HƯỚNG DẪN TOÀN DIỆN: 7 GIẢI PHÁP — 11 ĐỘ ĐO — 7 BỘ DỮ LIỆU VÀ KẾ HOẠCH NÂNG CẤP BÁO CÁO TỐT NGHIỆP

> **Đề tài đăng ký tốt nghiệp:** Nghiên cứu xây dựng trợ lý hội thoại AI cá nhân hóa người dùng trong quản lý nhà hàng  
> **Giảng viên hướng dẫn:** TS. Đỗ Thị Liên — Khoa CNTT, Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Sinh viên thực hiện:** Nguyễn Đức Bảo (B22DCCN064) — Nguyễn Mạnh Cường (B22DCCN100)  
> **Căn cứ yêu cầu chỉnh sửa:** Email chỉ đạo của TS. Đỗ Thị Liên (`teacher_request.txt`) và Báo cáo hiện tại (`Bao Cao GNN_GCL_RAG.pdf`).

---

## PHẦN 1: BẢN CHẤT KỸ THUẬT VÀ TOÁN HỌC CỦA 7 GIẢI PHÁP (GP1 ĐẾN GP7)

Đề tài xây dựng một hệ thống phân tầng gồm **7 giải pháp** đại diện cho 3 thế hệ công nghệ gợi ý, nhằm chứng minh sự tiến hóa vượt bậc của giải pháp đề xuất cốt lõi:

```
[GP1: Content-Based] ──┐
                       ├──► [GP3: Hybrid MF+CB] ──► [GP4: LightGCN] ──► [GP5: LightGCN + SimGCL] ──┐
[GP2: Matrix Fact.]  ──┘                                                                           │
                                                                                                   ▼
[GP6: Vanilla RAG (Truy xuất ngữ nghĩa văn bản thuần túy)] ──────────────────────────────► [GP7: GNN + GCL + RAG]
                                                                                              (Mô hình đề xuất)
```

---

### 1. GP1 — Content-Based Filtering (Lọc dựa trên nội dung)
* **Tệp mã nguồn:** [`src/GP1_model_cb.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP1_model_cb.py)
* **Bản chất toán học:** Biểu diễn sản phẩm (Item) bằng ma trận đặc trưng văn bản TF-IDF từ tiêu đề, thể loại, mô tả:
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
* **Hồ sơ sở thích người dùng ($\mathbf{p}_u$):** Là trung bình cộng có trọng số vector đặc trưng các món người dùng đã tương tác:
  $$\mathbf{p}_u = \frac{1}{|\mathcal{N}_u|} \sum_{i \in \mathcal{N}_u} \mathbf{v}_i$$
* **Dự đoán điểm xếp hạng:** Tính bằng độ tương đồng Cosine:
  $$\hat{y}_{ui} = \cos(\mathbf{p}_u, \mathbf{v}_i) = \frac{\mathbf{p}_u \cdot \mathbf{v}_i}{\|\mathbf{p}_u\|_2 \|\mathbf{v}_i\|_2}$$
* **Ưu điểm:** Khả năng giải quyết tốt hiện tượng khởi đầu lạnh cho sản phẩm mới (New Item Cold-Start) khi chưa có tương tác.
* **Nhược điểm:** Dễ bị thiên lệch (Over-specialization), chỉ gợi ý các món tương tự quá khứ, không khai thác được hành vi cộng đồng.

---

### 2. GP2 — Matrix Factorization (Lọc cộng tác phân rã ma trận với BPR Loss)
* **Tệp mã nguồn:** [`src/GP2_model_mf.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP2_model_mf.py)
* **Bản chất toán học:** Phân rã ma trận tương tác người dùng - sản phẩm $\mathbf{R} \in \mathbb{R}^{M \times N}$ thành tích vô hướng của 2 không gian ẩn:
  $$\hat{y}_{ui} = \mathbf{e}_u^T \cdot \mathbf{e}_i = \sum_{k=1}^d e_{u,k} \cdot e_{i,k}$$
* **Hàm mất mát Bayesian Personalized Ranking (BPR Loss):** Tối ưu hóa thứ tự ưu tiên cặp món đã tương tác ($i$) so với món chưa tương tác ($j$):
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|_2^2$$
* **Ưu điểm:** Khai thác quy luật ẩn của cộng đồng người dùng, tốc độ suy luận nhanh ($<1\text{ ms}$).
* **Nhược điểm:** Tuyến tính hóa tương tác, gặp khó khăn nghiêm trọng khi ma trận tương tác có độ thưa thớt cao ($> 99\%$).

---

### 3. GP3 — Hybrid Recommendation (Lọc kết hợp MF và CB)
* **Tệp mã nguồn:** [`src/GP3_model_hybrid.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP3_model_hybrid.py)
* **Bản chất toán học:** Kết hợp điểm số sau chuẩn hóa Min-Max (Late-Fusion) giữa lọc cộng tác (MF) và lọc nội dung (CB):
  $$S_{\text{Hybrid}}(u, i) = \alpha \cdot \tilde{S}_{\text{MF}}(u, i) + (1 - \alpha) \cdot \tilde{S}_{\text{CB}}(u, i), \quad (\alpha = 0.65)$$
* **Ưu điểm:** Bù trừ lẫn nhau: CB hỗ trợ giảm nhẹ Cold-start, MF gia tăng tính cá nhân hóa theo cộng đồng.
* **Nhược điểm:** Kết hợp bề mặt (Late-fusion), chưa học được biểu diễn không gian sâu chung giữa đồ thị và văn bản.

---

### 4. GP4 — LightGCN (Mạng nơ-ron đồ thị - Graph Neural Network)
* **Tệp mã nguồn:** [`src/GP4_model_lightgcn.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP4_model_lightgcn.py)
* **Bản chất toán học:** Xây dựng đồ thị lưỡng phân tương tác $\mathcal{G} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E})$. Cơ chế lan truyền thông điệp đồ thị tinh giản loại bỏ hoàn toàn ma trận trọng số phi tuyến tính $W$ và kích hoạt phi tuyến $\text{ReLU}$:
  $$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \mathbf{e}_i^{(k)}, \quad \mathbf{e}_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \mathbf{e}_u^{(k)}$$
* **Tổng hợp biểu diễn đa tầng:**
  $$\mathbf{e}_u^* = \frac{1}{K+1} \sum_{k=0}^K \mathbf{e}_u^{(k)}, \quad \mathbf{e}_i^* = \frac{1}{K+1} \sum_{k=0}^K \mathbf{e}_i^{(k)}$$
* **Ưu điểm:** Khai thác kết nối đa chặng (Higher-order collaborative signals), hiệu năng vượt trội so với MF truyền thống.
* **Nhược điểm:** Bị suy giảm hiệu năng khi đồ thị quá thưa thớt (ít cạnh liên kết).

---

### 5. GP5 — LightGCN + SimGCL (GNN kết hợp Học tương phản đồ thị)
* **Tệp mã nguồn:** [`src/GP5_model_lightgcn_gcl.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP5_model_lightgcn_gcl.py)
* **Bản chất toán học:** Khung học tương phản đơn giản (Simple Graph Contrastive Learning - SimGCL - Yu et al., SIGIR 2022). Thay vì biến đổi cấu trúc cạnh (DropEdge/DropNode) tốn kém, SimGCL cộng nhiễu trực tiếp vào không gian biểu diễn ẩn:
  $$\mathbf{e}' = \mathbf{e} + \epsilon \cdot \text{sign}(\mathbf{\Delta}), \quad \mathbf{\Delta} \sim \mathcal{N}(0, \mathbf{I})$$
* **Hàm mất mát liên hiệp (Joint Multi-Task Loss):**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{cl}} \cdot \mathcal{L}_{\text{CL}}$$
  với $\mathcal{L}_{\text{CL}}$ là InfoNCE loss:
  $$\mathcal{L}_{\text{CL}} = -\sum_{i \in \mathcal{B}} \log \frac{\exp(\mathbf{z}_i' \cdot \mathbf{z}_i'' / \tau)}{\sum_{j \in \mathcal{B}} \exp(\mathbf{z}_i' \cdot \mathbf{z}_j'' / \tau)}$$
* **Ưu điểm:** Phân bổ đồng đều vector nhúng (Uniformity), chống quá khớp, giải quyết triệt để vấn đề thưa thớt dữ liệu.

---

### 6. GP6 — Vanilla RAG (Truy xuất ngữ nghĩa CSDL Item văn bản thuần túy)
* **Tệp mã nguồn:** [`src/GP6_model_rag.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP6_model_rag.py)
* **Bản chất toán học:** Gợi ý sản phẩm chỉ dựa trên truy xuất tương đồng vector ngữ nghĩa giữa từ khóa/truy vấn của người dùng với CSDL Item qua không gian vector nhúng (Dense retrieval / TF-IDF). **Không sử dụng mạng đồ thị GNN cá nhân hóa**.
* **Ý nghĩa học thuật:** Đóng vai trò **mô hình đối chứng cốt lõi trong Ablation Study** nhằm chứng minh: Nếu chỉ áp dụng RAG dựa trên CSDL Item đơn thuần mà thiếu GNN+GCL thì hệ thống sẽ rơi vào bẫy thiên lệch độ phổ biến (Popularity Bias) và mất tính cá nhân hóa chiều sâu.

---

### 7. GP7 — GNN + GCL + RAG (Giải pháp Trợ lý Hội thoại Đề xuất Toàn diện của Đồ án)
* **Tệp mã nguồn:** [`src/GP7_model_gnn_gcl_rag.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP7_model_gnn_gcl_rag.py)
* **Bản chất kỹ thuật:** Hệ thống hợp nhất gồm 3 pha phối hợp song song:
  1. **Nhánh 1A (RAG Context Retrieval):** Bi-Encoder `all-MiniLM-L6-v2` mã hóa câu hỏi thành vector 384 chiều, ChromaDB tìm kiếm ngữ cảnh hội thoại tương tự bằng chỉ mục HNSW và lọc trùng lặp MMR ($\lambda = 0.6$).
  2. **Nhánh 1B (GNN+GCL SimGCL Recommender):** Khai thác biểu diễn đồ thị sở thích người dùng, lọc các món đã xem qua cơ chế **Train Masking**, tính điểm và trích xuất Top-N sản phẩm tối ưu.
  3. **Pha 2 (Structured Prompt Fusion):** Tích hợp kết quả 2 nhánh vào bản mẫu Prompt 5 khối có khóa chống ảo giác.
  4. **Pha 3 (LLM Response):** Google Gemini Flash phân tích, giải thích lý do gợi ý và sinh phản hồi tự nhiên, đảm bảo 100% món ăn nằm trong danh mục.

---

## PHẦN 2: HỆ THỐNG 11 ĐỘ ĐO ĐÁNH GIÁ (ÁP DỤNG CHUNG CHO TẤT CẢ GIẢI PHÁP)

Toàn bộ 11 độ đo được tính toán thống nhất trong [`src/metrics.py`](file:///d:/DoAnTotNghiep/recommender-system/src/metrics.py) trên danh sách xếp hạng Top-$K$ ($K \in [10, 20]$) sau khi áp dụng Train Masking:

### Nhóm A: Các độ đo theo yêu cầu của Cô (Ưu tiên hàng đầu)
1. **Precision@K:** Tỷ lệ sản phẩm gợi ý đúng trong số $K$ sản phẩm đề xuất:
   $$\text{Precision@K} = \frac{|\text{TopK} \cap \mathcal{I}_{\text{test}}|}{|K|}$$
2. **Recall@K:** Tỷ lệ tìm được các sản phẩm thực tế mà người dùng thích trên tập Test:
   $$\text{Recall@K} = \frac{|\text{TopK} \cap \mathcal{I}_{\text{test}}|}{|\mathcal{I}_{\text{test}}|}$$
3. **NDCG@K (Normalized Discounted Cumulative Gain):** Đánh giá chất lượng xếp hạng (item đúng xuất hiện ở vị trí càng cao điểm càng lớn):
   $$\text{DCG@K} = \sum_{r=1}^K \frac{\mathbb{I}(i_r \in \mathcal{I}_{\text{test}})}{\log_2(r+1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
4. **MRR@K (Mean Reciprocal Rank):** Nghịch đảo vị trí của sản phẩm đúng đầu tiên trong danh sách:
   $$\text{MRR@K} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_1(u)}$$
5. **Coverage@K:** Tỷ lệ bao phủ danh mục sản phẩm của toàn hệ thống (đánh giá mức độ bình đẳng gợi ý):
   $$\text{Coverage@K} = \frac{|\bigcup_{u \in \mathcal{U}} \text{TopK}(u)|}{N}$$
6. **Improvement (%):** Tỷ lệ phần trăm cải thiện hiệu năng của giải pháp mục tiêu so với baseline:
   $$\text{Improvement (\%)} = \frac{\text{Metric}_{\text{target}} - \text{Metric}_{\text{base}}}{\text{Metric}_{\text{base}}} \times 100\%$$

---

### Nhóm B: Các độ đo mở rộng bổ sung (Nâng cao tính thuyết phục học thuật)
7. **HitRate@K (HR@K):** Tỷ lệ người dùng nhận được ít nhất 1 sản phẩm đúng trong Top-$K$ ($1.0$ nếu có $\ge 1$ hit, ngược lại $0.0$).
8. **F1-Score@K (F1@K):** Trung bình điều hòa giữa Precision@K và Recall@K:
   $$\text{F1@K} = 2 \times \frac{\text{Precision@K} \times \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$
9. **MAP@K (Mean Average Precision):** Độ chính xác trung bình có trọng số theo từng vị trí trúng.
10. **Novelty@K:** Khả năng gợi ý các sản phẩm mới lạ/ngách dài hạn (Long-tail items) qua hàm tự thông tin $-\log_2 P(i)$.
11. **Diversity@K (Intra-List Diversity):** Độ đa dạng, khác biệt thuộc tính giữa các sản phẩm trong cùng một danh sách Top-$K$:
    $$\text{ILD} = \frac{2}{K(K-1)} \sum_{i \in \text{TopK}} \sum_{j \in \text{TopK}, j \ne i} (1 - \cos(\mathbf{v}_i, \mathbf{v}_j))$$

---

## PHẦN 3: BẢNG ĐẶC TẢ 7 BỘ DỮ LIỆU THỰC NGHIỆM ĐA MIỀN

Dữ liệu đã có sẵn 100% trong dự án, bao phủ đầy đủ các lĩnh vực theo đúng chỉ đạo của Cô:

| STT | Bộ Dữ Liệu | Miền Ứng Dụng | Đường dẫn / Tệp dữ liệu | Số Users ($M$) | Số Items ($N$) | Số Tương Tác ($|\mathcal{E}|$) | Độ Thưa (Sparsity) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **MovieLens 100K** | Phim ảnh | `ml-latest-small/` | 609 | 9,742 | 81,763 | 98.62% |
| **2** | **Last.fm** (`hetrec2011`) | Âm nhạc / Nghệ sĩ | `hetrec2011-lastfm-2k/` | 1,892 | 17,632 | 92,834 | 99.72% |
| **3** | **Amazon Reviews** | Thương mại điện tử (Nhạc cụ) | `reviews_Musical_Instruments_5.json/` | 1,429 | 900 | 9,794 | 99.24% |
| **4** | **TripAdvisor** | Khách sạn & Du lịch | `TripAdvisor_Dataset/` | 500 | 14,206 | 13,624 | 99.81% |
| **5** | **Yelp Dataset** (`yelp2018`) | Nhà hàng & Dịch vụ ăn uống | `Yelp_Dataset/` | 31,668 | 38,048 | 1,561,406 | 99.87% |
| **6** | **Ecommerce Taobao** | Bán lẻ trực tuyến | `Taobao_Dataset/` | 1,300 | 17 | 68,900 | 68.82% |
| **7** | **Netflix Prize Dataset** | Phim & Chương trình TV | `Netflix_Dataset/` | 400 | 8,807 | 8,800 | 99.75% |

---

## PHẦN 4: HƯỚNG DẪN CỤ THỂ CÁCH SỬA BÁO CÁO `Bao Cao GNN_GCL_RAG.pdf`

Đối chiếu trực tiếp với file PDF báo cáo hiện tại:

### 1. Giữ nguyên và phát huy:
* **Tên đề tài:** *"Nghiên cứu xây dựng trợ lý hội thoại AI cá nhân hóa người dùng trong quản lý nhà hàng"*.
* **Chương I & II:** Tổng quan bài toán, cơ sở đồ thị lưỡng phân, khung kiến trúc 2 nhánh song song của hệ thống PCRS (Hình 1, 2, 3).
* **Chương III:** Cơ chế Message Passing của LightGCN (Hình 5), nguyên lý SimGCL Joint Loss (Hình 6, 7), và Cấu trúc Prompt 5 khối (Hình 8).
* **Chương IV:** Thiết kế lớp (Hình 9) và Sơ đồ luồng thực thi điều phối End-to-End (Hình 10).
* **Mục 5.3 & 5.4:** Phân tích quy trình xử lý Testcase mẫu (User 5, MovieLens) và phân tích Ablation Study.

### 2. Sửa đổi và Bổ sung vào Chương V (Thực nghiệm):
1. **Thêm Mục 5.0: Môi trường và Dữ liệu thực nghiệm đa miền:**
   - Đưa Bảng đặc tả 7 bộ dữ liệu vào báo cáo để chứng minh tính tổng quát và năng lực mở rộng của giải pháp đề xuất.
2. **Thêm Mục 5.1: Hệ thống 11 độ đo đánh giá tiêu chuẩn:**
   - Trình bày công thức và ý nghĩa của 2 nhóm độ đo: *Nhóm theo yêu cầu của Cô* (Precision, Recall, NDCG, MRR, Coverage, Improvement) và *Nhóm mở rộng bổ sung* (HitRate, F1, MAP, Novelty, Diversity).
3. **Thêm Bảng kết quả thực nghiệm so sánh 7 giải pháp trên 7 dataset:**
   - Trích xuất bảng kết quả do `src/run_all_experiments.py` sinh ra đưa vào các bảng so sánh tương ứng.
4. **Chuẩn hóa Bảng 5.1 và Hình 11 (Phân rã độ trễ):**
   - Đổi tiêu đề Bảng 5.1 từ *"trung bình trên 50 truy vấn"* thành *"Đo lường trên 100 lượt tương tác thực tế"* để khớp 100% với lời văn ở trang 23 của PDF.
   - Bổ sung định nghĩa rõ ràng về mặt thống kê:
     - **$P_{50}$ (Phân vị 50 - Trung vị):** Thời gian đáp ứng trong điều kiện vận hành thông thường.
     - **$P_{95}$ (Phân vị 95):** Cận trên thời gian đáp ứng trong kịch bản tải bất lợi nhất (loại trừ các ngoại lệ).

---

## PHẦN 5: DANH MỤC CÁC TỆP MÃ NGUỒN CHẠY THỰC NGHIỆM TRONG `src/`

Mọi tệp tin đã được chuẩn hóa và gom gọn bên trong thư mục [`src/`](file:///d:/DoAnTotNghiep/recommender-system/src):

### 1. Nhóm các tệp mô hình thuật toán (GP1 -> GP7):
| Tệp Mô Hình | Tên Giải Pháp | Lệnh Chạy Độc Lập |
| :--- | :--- | :--- |
| [`src/GP1_model_cb.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP1_model_cb.py) | Giải pháp 1: Content-Based Filtering | `python src/GP1_model_cb.py` |
| [`src/GP2_model_mf.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP2_model_mf.py) | Giải pháp 2: Matrix Factorization (BPR Loss) | `python src/GP2_model_mf.py` |
| [`src/GP3_model_hybrid.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP3_model_hybrid.py) | Giải pháp 3: Hybrid (MF + CB) | `python src/GP3_model_hybrid.py` |
| [`src/GP4_model_lightgcn.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP4_model_lightgcn.py) | Giải pháp 4: LightGCN (GNN) | `python src/GP4_model_lightgcn.py` |
| [`src/GP5_model_lightgcn_gcl.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP5_model_lightgcn_gcl.py) | Giải pháp 5: LightGCN + SimGCL (GNN + GCL) | `python src/GP5_model_lightgcn_gcl.py` |
| [`src/GP6_model_rag.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP6_model_rag.py) | Giải pháp 6: Vanilla RAG (Truy xuất CSDL Item) | `python src/GP6_model_rag.py` |
| [`src/GP7_model_gnn_gcl_rag.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP7_model_gnn_gcl_rag.py) | Giải pháp 7: GNN + GCL + RAG (Đề tài đề xuất) | `python src/GP7_model_gnn_gcl_rag.py` |

### 2. Nhóm các tệp thực thi / công cụ (Scripts):
| Tệp Thực Thi | Chức Năng Chính | Lệnh Thực Thi |
| :--- | :--- | :--- |
| [`src/run_all_experiments.py`](file:///d:/DoAnTotNghiep/recommender-system/src/run_all_experiments.py) | **File chính chạy toàn bộ thực nghiệm:** Chạy 7 giải pháp trên 7 dataset và xuất bảng 11 độ đo | `python src/run_all_experiments.py` |
| [`src/train_and_save_simgcl.py`](file:///d:/DoAnTotNghiep/recommender-system/src/train_and_save_simgcl.py) | Huấn luyện và lưu trữ checkpoint mô hình SimGCL cho Nhánh 1B | `python src/train_and_save_simgcl.py` |
| [`src/demo_conversation.py`](file:///d:/DoAnTotNghiep/recommender-system/src/demo_conversation.py) | Giao diện Chatbot tương tác CLI trực tiếp với người dùng khi bảo vệ đồ án | `python src/demo_conversation.py --test-mode` |
| [`src/evaluate_pcrs.py`](file:///d:/DoAnTotNghiep/recommender-system/src/evaluate_pcrs.py) | Kiểm thử và đo lường độ trễ P50/P95 trên 100 lượt tương tác | `python src/evaluate_pcrs.py --latency-test --n-queries 100` |
