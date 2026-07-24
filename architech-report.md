# CẤU TRÚC VÀ CHI TIẾT BÁO CÁO ĐỒ ÁN TỐT NGHIỆP (BẢN CHUẨN HOÀN CHỈNH)

---

## CHƯƠNG I: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU
* **1.1. Đặt vấn đề:** Trình bày bối cảnh quá tải thông tin và lý do cần Hệ tư vấn (sử dụng thuật ngữ tổng quát $User$, $Item$, tránh nhắc tới miền dữ liệu đặc thù như món ăn, phim ảnh...).
* **1.2. Mục tiêu nghiên cứu:** Tìm hiểu các phương pháp tư vấn từ cơ bản đến nâng cao (GNN - LightGCN).
* **1.3. Phạm vi nghiên cứu:** Giới hạn đánh giá trên bộ dữ liệu chuẩn công bố quốc tế (MovieLens).

---

## CHƯƠNG II: TỔNG QUAN HỆ TƯ VẤN
* **2.1. Định nghĩa bài toán tổng quát:** Định nghĩa toán học của $User$, $Item$, ma trận tương tác $R$ và đầu ra $Top-K$.
* **2.2. Phân loại các hướng tiếp cận:** Trình bày 4 nhánh chính: Lọc nội dung, Lọc cộng tác, Lai và Mạng nơ-ron đồ thị.
* **[VISUALIZE 1]:** Cây phân loại các phương pháp Hệ tư vấn.

---

## CHƯƠNG III: CÁC PHƯƠNG PHÁP TƯ VẤN
*(Mỗi mục nhỏ từ 3.1 đến 3.4 tuân thủ đúng 5 ý: Ý tưởng $\rightarrow$ Input/Output $\rightarrow$ Sơ đồ luồng $\rightarrow$ Công thức $\rightarrow$ Ví dụ số)*

### 3.1. Phương pháp Lọc nội dung (Content-Based Filtering)
* **1. Ý tưởng bản chất:** Gợi ý $Item$ dựa trên độ tương đồng giữa thuộc tính/nội dung của $Item$ và lịch sử sở thích cá nhân của $User$.
* **2. Dữ liệu Đầu vào / Đầu ra:**
  * *Input:* Thuộc tính $Item$ (thể loại, mô tả) và danh sách $Item$ đã tương tác của $User$.
  * *Output:* Điểm số tương đồng $Cosine$ và danh sách xếp hạng $Top-K$.
* **3. [VISUALIZE 2]:** Sơ đồ luồng TF-IDF $\rightarrow$ Vector Item $\rightarrow$ User Profile $\rightarrow$ Cosine Similarity $\rightarrow$ Top-K.
* **4. Công thức toán học:** Trọng số TF-IDF, Vector User Profile (trung bình cộng), Độ đo Cosine Similarity.
* **5. Ví dụ bằng số cụ thể:** Cho 3 vector Item 2 chiều, tính User Profile và phép nhân Cosine để chọn Item gợi ý.

### 3.2. Collaborative Filtering — Phân rã ma trận (Matrix Factorization)
* **1. Ý tưởng bản chất:** Phân rã ma trận tương tác thưa thành các vector đặc trưng ẩn ($Latent\ Embeddings$) đại diện cho $User$ và $Item$ để học sở thích từ cộng đồng.
* **2. Dữ liệu Đầu vào / Đầu ra:**
  * *Input:* Ma trận tương tác thưa $R \in \mathbb{R}^{M \times N}$.
  * *Output:* Vector nhúng $p_u, q_i \in \mathbb{R}^d$ và điểm dự đoán tích vô hướng.
* **3. [VISUALIZE 3]:** Hình vẽ sơ đồ phân rã ma trận $R_{(M \times N)} \approx P_{(M \times d)} \times Q^T_{(d \times N)}$.
* **4. Công thức toán học:** Công thức Dot Product ($\hat{y}_{ui} = p_u^T q_i$) và Hàm mất mát BPR Loss.
* **5. Ví dụ bằng số cụ thể:** Cho 1 vector $User\ [0.5, 0.8]$ và 1 vector $Item\ [0.9, 0.1]$, tính tích vô hướng ra điểm số dự đoán.

### 3.3. Phương pháp Lai (Hybrid Systems — Late Fusion)
* **1. Ý tưởng bản chất:** Kết hợp điểm xếp hạng từ MF (hành vi cộng đồng) và CB (đặc trưng nội dung) theo cơ chế Late Fusion để giảm thiểu hiện tượng Cold-Start.
* **2. Dữ liệu Đầu vào / Đầu ra:**
  * *Input:* Ma trận điểm từ mô hình MF và mô hình CB.
  * *Output:* Ma trận điểm tổng hợp sau chuẩn hóa.
* **3. [VISUALIZE 4]:** Sơ đồ kết hợp chữ Y: MF Score $\rightarrow$ Min-Max $\searrow \oplus_{\alpha} \swarrow$ Min-Max $\leftarrow$ CB Score.
* **4. Công thức toán học:** Chuẩn hóa Min-Max theo hàng User, Công thức cộng trọng số Late Fusion ($\alpha \cdot S_{MF} + (1-\alpha) \cdot S_{CB}$).
* **5. Ví dụ bằng số cụ thể:** Lấy 1 điểm MF (3.2/5) và 1 điểm CB (0.8/1), chuẩn hóa về $[0, 1]$ và nhân với $\alpha = 0.6$.

### 3.4. Mạng nơ-ron đồ thị (LightGCN — Đồ thị lưỡng phân)
* **1. Ý tưởng bản chất:** Biểu diễn dữ liệu dưới dạng Đồ thị lưỡng phân ($Bipartite\ Graph$) chỉ gồm $User$ và $Item$. Áp dụng cơ chế $Message\ Passing$ qua các tầng để học quan hệ cấu trúc bậc cao mà không dùng phép biến đổi phi tuyến phức tạp.
* **2. Dữ liệu Đầu vào / Đầu ra:**
  * *Input:* Đồ thị lưỡng phân $G = (U \cup I, E)$.
  * *Output:* Vector nhúng tổng hợp đa tầng $e_u^{(final)}, e_i^{(final)}$ và điểm tích vô hướng.
* **3. Sơ đồ minh họa:**
  * **[VISUALIZE 5]:** Hình vẽ Đồ thị lưỡng phân User–Item (Bipartite Graph).
  * **[VISUALIZE 6]:** Sơ đồ quy trình Lan truyền thông điệp (Message Passing) qua $K$ tầng.
* **4. Công thức toán học:** Khởi tạo $e^{(0)}$, Lan truyền Message Passing với hệ số chuẩn hóa độ bậc $1/\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}$, Cộng trung bình đa tầng và Dot Product.
* **5. Ví dụ bằng số cụ thể:** Tính lan truyền 1 bước cho 1 Node User từ 2 Node Item láng giềng kèm hệ số chuẩn hóa độ bậc.

---

## CHƯƠNG IV: CÀI ĐẶT HỆ THỐNG (MÔ HÌNH HÓA VÀ LUỒNG THỰC THI)
*(Lưu ý: Viết theo luồng thuật toán của từng phương pháp và kiến trúc Class, không dán code hay giải thích dòng lệnh rời rạc)*

### 4.1. Kiến trúc hệ thống tổng thể
* Trình bày sơ đồ luồng tổng thể kết nối các khối: Dữ liệu thô $\rightarrow$ Tiền xử lý & Tạo đồ thị/ma trận $\rightarrow$ Khối 4 Mô hình tư vấn $\rightarrow$ Huấn luyện & Đánh giá Top-K.
* **[VISUALIZE 7]:** Sơ đồ luồng xử lý tổng thể từ dữ liệu thô đến kết quả.

### 4.2. Luồng thực thi phương pháp Lọc nội dung (Content-Based Pipeline)
* **Bước 1 (Vector hóa thuộc tính):** Trích xuất thuộc tính văn bản của Item, áp dụng `TfidfVectorizer` sinh ma trận đặc trưng $V_{Item}$.
* **Bước 2 (Xây dựng User Profile):** Lọc danh sách Item đã tương tác trong tập Train của từng User, tính trung bình cộng vector để tạo vector đại diện sở thích $U_u$.
* **Bước 3 (Tính tương đồng & Masking):** Tính điểm Cosine Similarity giữa $U_u$ với toàn bộ $V_{Item}$, che các Item đã tương tác và rút trích danh sách $Top-K$.

### 4.3. Luồng thực thi Phân rã ma trận (Matrix Factorization Pipeline)
* **Bước 1 (Khởi tạo Bảng nhúng):** Định nghĩa hai bảng `user_embedding` và `item_embedding` độc lập ($d=32$).
* **Bước 2 (Tính điểm & Lấy mẫu âm):** Tính điểm tích vô hướng $p_u^T q_i$. Với mỗi tương tác dương $(u, i)$, rút ngẫu nhiên mẫu âm $(u, j)$ chưa tương tác.
* **Bước 3 (Tối ưu BPR Loss):** Ép khoảng cách điểm giữa mẫu dương và mẫu âm qua hàm Sigmoid và cập nhật Gradient cho hai bảng nhúng.

### 4.4. Luồng thực thi Phương pháp Lai (Hybrid Pipeline)
* **Bước 1 (Thu thập điểm thành phần):** Lấy ma trận điểm dự đoán từ mô hình MF và CB.
* **Bước 2 (Chuẩn hóa Min-Max):** Tiến hành chuẩn hóa điểm về cùng thang $[0, 1]$ theo từng hàng User.
* **Bước 3 (Late Fusion):** Kết hợp hai ma trận điểm qua công thức cộng trọng số $\alpha$.

### 4.5. Luồng thực thi Mạng nơ-ron đồ thị (LightGCN Pipeline)
* **Bước 1 (Xây dựng Đồ thị Lưỡng phân):** Định danh ID (Node Mapping), chuyển ma trận tương tác thành ma trận cạnh `edge_index` 2 chiều (vô hướng) phục vụ Message Passing.
* **Bước 2 (Kiến trúc Class & Mối quan hệ Module):**
  * **[VISUALIZE 8]:** Sơ đồ mối quan hệ giữa Class `LightGCNConv` (Lớp lan truyền Message Passing mức thấp) và Class `LightGCNRecommender` (Module bọc mức cao quản lý Embedding đa tầng & Dot Product).
* **Bước 3 (Lan truyền & Gom tụ):** Gọi `LightGCNConv` lặp qua $K=2$ tầng, tính trung bình cộng embedding các tầng $e^{(final)} = \frac{1}{3}(e^{(0)} + e^{(1)} + e^{(2)})$.

### 4.6. Quy trình Huấn luyện & Đánh giá chung (Training & Evaluation Pipeline)
* **Phân tách dữ liệu:** Chia tập tương tác theo tỷ lệ 80% Train / 10% Validation / 10% Test (`random_seed = 42`).
* **Lấy mẫu âm (Negative Sampling):** Lấy mẫu âm ngẫu nhiên cho thuật toán huấn luyện BPR Loss.
* **Cơ chế Masking (Che món đã xem):** Gán điểm $-\infty$ cho toàn bộ các Item nằm trong tập Train của User trước khi sắp xếp lấy Top-K ở tập Test.

---

## CHƯƠNG V: THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Bộ dữ liệu thực nghiệm (Dataset Specification)
* Trình bày bảng đặc tả chi tiết bộ dữ liệu chuẩn **MovieLens 100K / 1M**: Nguồn (GroupLens), số lượng User, Item, Tương tác, Độ thưa ma trận ($Sparsity$).
* Quy trình tiền xử lý: Bỏ rating $< 3$, lọc User có $< 5$ tương tác, chuyển thể loại phim thành ma trận TF-IDF.

### 5.2. Môi trường và Cài đặt thực nghiệm (Setup & Hyperparameters)
* Cấu hình phần cứng (Colab GPU T4) và thư viện (PyTorch, PyTorch Geometric).
* Thiết lập Hyperparameters & Lý do lựa chọn:
  * `Embedding dim d = 32` (Tránh Overfitting trên dữ liệu thưa).
  * `Số tầng LightGCN K = 2` (Khai thác láng giềng 2 bước, tránh Over-smoothing).
  * `Learning rate = 0.01`, `Batch size = 2048`, `Trọng số Hybrid alpha = 0.6`.

### 5.3. Các độ đo đánh giá (Kèm ví dụ minh họa bằng số)
* **Precision@K:** Tỷ lệ đoán trúng trong K Item gợi ý.
  * *Ví dụ:* Gợi ý Top 10 Item, User thích 3 Item trong đó $\rightarrow \text{Precision}@10 = 3/10 = 0.30$ ($30\%$).
* **Recall@K:** Tỷ lệ bao phủ Item thực tế mà User thích.
  * *Ví dụ:* Thực tế User thích 5 Item trong tập Test. Top 10 gợi ý trúng 2 Item $\rightarrow \text{Recall}@10 = 2/5 = 0.40$ ($40\%$).
* **NDCG@K:** Đánh giá chất lượng thứ hạng xếp hạng (Item đúng ở vị trí cao được điểm tốt hơn).
  * *Ví dụ:* Trúng Item đúng ở vị trí #1 đạt $\text{NDCG}@3 = 1.0$, nếu trúng ở vị trí #3 đạt $\text{NDCG}@3 = \frac{1/ \log_2(4)}{1} \approx 0.50$.
* **MRR@K (Mean Reciprocal Rank):** Nghịch đảo vị trí trúng đầu tiên.
  * *Ví dụ:* Item đúng xuất hiện sớm nhất ở vị trí thứ 2 $\rightarrow \text{RR} = 1/2 = 0.50$.
* **HitRate@K:** Tỷ lệ User nhận được ít nhất 1 Item đúng.
  * *Ví dụ:* Trong 100 User, có 75 User có ít nhất 1 Item trúng trong Top-K $\rightarrow \text{HitRate} = 75/100 = 0.75$ ($75\%$).
* **Coverage (Độ bao phủ hệ thống):** Tỷ lệ Item trong toàn bộ danh mục được mang đi gợi ý.
  * *Ví dụ:* Kho có 1000 Item. Tổng hợp gợi ý cho toàn bộ User thấy xuất hiện 250 Item khác nhau $\rightarrow \text{Coverage} = 250/1000 = 0.25$ ($25\%$).

### 5.4. Kết quả thực nghiệm và Phân tích so sánh
* Bảng kết quả so sánh 4 mô hình (CB, MF, Hybrid, LightGCN) ở các ngưỡng Top-10, Top-20.
* **[VISUALIZE 9]:** Biểu đồ hình cột so sánh $Recall@20$ và $NDCG@20$ giữa 4 mô hình.
* Nhận xét chuyên sâu giải thích nguyên nhân: Vì sao LightGCN cao nhất (nhờ Message Passing 2 tầng), vì sao Hybrid vượt trội MF (khắc phục Cold-start), vì sao CB có Precision thấp nhưng Coverage cao.

---

## CHƯƠNG VI: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **6.1. Kết luận:** Tóm tắt kết quả đạt được, khẳng định tính hiệu quả của GNN trên dữ liệu chuẩn.
* **6.2. Hạn chế & Hướng phát triển:** Tích hợp yếu tố thời gian ($Temporal$), thử nghiệm UltraGCN hoặc kết hợp Mô hình ngôn ngữ lớn ($LLMs$).

---

## TỔNG HỢP DANH SÁCH 9 VỊ TRÍ BẮT BUỘC VISUALIZE (HÌNH VẼ)

| # | Vị trí | Tên Hình vẽ / Sơ đồ | Loại hình | Chuẩn nội dung |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Chương II.3 | Cây phân loại các phương pháp Hệ tư vấn | Tree Diagram | 4 nhánh chính (CB, CF, Hybrid, GNN) |
| **2** | Chương III.1 | Flowchart luồng xử lý Content-Based | Flowchart | TF-IDF $\rightarrow$ User Profile $\rightarrow$ Cosine |
| **3** | Chương III.2 | Sơ đồ Phân rã Ma trận $R \approx P \times Q^T$ | Matrix Diagram | 1 ma trận thưa lớn = 2 ma trận nhỏ |
| **4** | Chương III.3 | Sơ đồ kết hợp Late Fusion (MF + CB) | Block Diagram | Nhánh chữ Y đi qua Min-Max |
| **5** | Chương III.4 | Đồ thị lưỡng phân User–Item (Bipartite Graph) | Graph Diagram | Đỉnh User & Đỉnh Item nối bằng Cạnh tương tác |
| **6** | Chương III.4 | Sơ đồ Lan truyền thông điệp (Message Passing) | Propagation Diagram | Lan truyền 2 chiều qua $K$ tầng |
| **7** | Chương IV.1 | Kiến trúc mã nguồn & Luồng xử lý tổng thể hệ thống | Architecture Diagram | Khối Data $\rightarrow$ Khối Models $\rightarrow$ Khối Train/Eval |
| **8** | Chương IV.5 | Mối quan hệ giữa Class `LightGCNConv` & `LightGCNRecommender` | Class Diagram | Class bọc mức cao và Class lan truyền mức thấp |
| **9** | Chương V.4 | Biểu đồ cột so sánh $Recall@20$ và $NDCG@20$ | Bar Chart | So sánh trực quan hiệu năng 4 mô hình |