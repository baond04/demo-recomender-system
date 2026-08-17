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

### 5.1. Thiết lập thực nghiệm và Đặc tả các Bộ dữ liệu
* **Tổng quan các bộ dữ liệu thực nghiệm:** Giới thiệu 3 miền ứng dụng đại diện: Điện ảnh (MovieLens Latest Small), Địa điểm dịch vụ & Nhà hàng (Yelp Dataset 5-core) và Âm nhạc (Last.fm hetrec2011).
* **Bảng đặc tả thông số tổng hợp:** Bảng so sánh đối chiếu giữa 3 bộ dữ liệu về các chỉ số: Số User ($M$), Số Item ($N$), Số tương tác ($\vert E \vert$), Mật độ tương tác, Độ thưa ma trận ($Sparsity$) và Đặc trưng thuộc tính Metadata (Genres, Business Categories, Tags).
* **Quy trình tiền xử lý dữ liệu chung:** Chuẩn hóa tương tác ẩn ($r \ge 3.0$), Mã hóa định danh hai chiều (Index Mapping), Trích xuất ma trận đặc trưng TF-IDF, Phân tách tập dữ liệu ngẫu nhiên theo người dùng tỷ lệ 80% Train / 10% Validation / 10% Test (`seed = 42`).
* **Cấu hình Siêu tham số hệ thống ($Hyperparameters$):** Kích thước nhúng $d = 32$, Số tầng lan truyền LightGCN $K = 2$, Tốc độ học $lr = 0.01$, Batch size 2048, Hệ số phạt $L_2 = 10^{-4}$, Trọng số mô hình Lai $\alpha = 0.6$, Tối ưu BPR Loss qua 10-20 Epochs.

### 5.2. Hệ thống các Độ đo Đánh giá Tiêu chuẩn
*(Trình bày chi tiết ý nghĩa, công thức toán học và ví dụ tính toán bằng số cụ thể cho từng độ đo ở ngưỡng Top-K)*
* **Precision@K:** Độ chính xác gợi ý trong Top-K. $\text{Precision}@K = \frac{\vert \text{Top-}K(u) \cap \mathcal{I}_{test}(u) \vert}{K}$.
* **Recall@K:** Độ bao phủ sở thích thực tế trong Top-K. $\text{Recall}@K = \frac{\vert \text{Top-}K(u) \cap \mathcal{I}_{test}(u) \vert}{\vert \mathcal{I}_{test}(u) \vert}$.
* **NDCG@K:** Chất lượng và vị trí xếp hạng của sản phẩm gợi ý đúng. $\text{DCG}@K = \sum_{r=1}^{K} \frac{I(r \in \mathcal{I}_{test}(u))}{\log_2(r + 1)}, \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$.
* **MRR@K:** Nghịch đảo thứ hạng xuất hiện của sản phẩm gợi ý đúng đầu tiên. $\text{MRR}@K = \frac{1}{\text{rank}_{first}}$.
* **HitRate@K:** Tỷ lệ người dùng nhận được ít nhất 1 gợi ý đúng. $\text{HitRate}@K = \begin{cases} 1.0, & \text{nếu } \vert \text{Top-}K(u) \cap \mathcal{I}_{test}(u) \vert \ge 1 \\ 0.0, & \text{ngược lại} \end{cases}$.
* **Coverage@K:** Độ bao phủ danh mục sản phẩm toàn hệ thống. $\text{Coverage}@K = \frac{\vert \bigcup_{u \in U} \text{Top-}K(u) \vert}{N}$.

### 5.3. Kết quả thực nghiệm trên Miền dữ liệu Điện ảnh (MovieLens)
* **Đặc tả miền Phim ảnh:** 609 Users, 9,742 Items, 81,763 Tương tác tích cực, Độ thưa ma trận $98.62\%$. Đặc trưng thuộc tính: 19 thể loại phim.
* **Bảng kết quả đánh giá định lượng:** So sánh 4 mô hình (CB, MF, Hybrid, LightGCN) tại các ngưỡng Top-10 và Top-20 (kết hợp cơ chế Train Masking $S[u,i] = -\infty$).
* **Biểu đồ & Phân tích nhận xét chuyên sâu:** Giải thích lý do LightGCN đạt hiệu năng tối ưu vượt trội nhờ khả năng học biểu diễn đa tầng từ cấu trúc Đồ thị Lưỡng phân User–Item qua cơ chế Message Passing $2\text{-hop}$.

### 5.4. Kết quả thực nghiệm trên Miền dữ liệu Thiết bị & Sản phẩm Âm nhạc (Amazon Musical Instruments)
* **Đặc tả miền TMĐT Âm nhạc:** 1,429 Users, 900 Items (Thiết bị âm nhạc), 10,261 Tương tác tích cực, Dung lượng file siêu nhẹ **1.5MB** (`reviews_Musical_Instruments_5.json.gz`). Độ thưa $99.20\%$. Đặc trưng thuộc tính: Danh mục sản phẩm & Nhận xét.
* **Bảng kết quả đánh giá định lượng:** So sánh 4 mô hình (CB, MF, Hybrid, LightGCN) tại hai ngưỡng Top-10 và Top-20.
* **Biểu đồ & Phân tích nhận xét chuyên sâu:** Phân tích ảnh hưởng của độ thưa dữ liệu ($99.20\%$) khiến mô hình MF thuần túy bị suy giảm hiệu năng. Đánh giá vai trò của mô hình Lai (Hybrid) khi kết hợp 40% tín hiệu Lọc nội dung ($\alpha = 0.6$) giúp gia tăng Coverage và hỗ trợ gợi ý cho sản phẩm mới (Cold-start).

### 5.5. Kết quả thực nghiệm trên Miền dữ liệu Âm nhạc (Last.fm)
* **Đặc tả miền Âm nhạc:** 1,892 Users, 17,632 Items (Nghệ sĩ), 92,834 Tương tác, Độ thưa $99.72\%$. Đặc trưng thuộc tính: Thẻ gắn do người dùng định nghĩa (User Tags).
* **Bảng kết quả đánh giá định lượng:** So sánh 4 mô hình (CB, MF, Hybrid, LightGCN) tại hai ngưỡng Top-10 và Top-20.
* **Biểu đồ & Phân tích nhận xét chuyên sâu:** Đánh giá tác động của hành vi nghe nhạc lặp lại giúp cấu trúc đồ thị hình thành các cụm tương đồng liên kết mạnh (Clustering), tạo điều kiện cho LightGCN thu thập ngữ cảnh cộng đồng hiệu quả nhất.

### 5.6. Phân tích So sánh Tổng hợp và Đánh giá Tính Tổng quát
* **So sánh đối chiếu chéo giữa 3 miền dữ liệu:** Sử dụng biểu đồ cột nhóm tổng hợp so sánh sự chuyển giao hiệu năng (Recall@20 và NDCG@20) của LightGCN và Hybrid trên 3 tập dữ liệu (MovieLens, Yelp, Last.fm).
* **Phân tích hiện tượng Đánh đổi (Trade-off):** Phân tích sự đánh đổi giữa Độ chính xác xếp hạng (Precision/Recall) và Độ bao phủ danh mục (Coverage).
* **Tóm tắt kết quả Chương V:** Khẳng định tính đúng đắn, ổn định và khả năng mở rộng của giải pháp Mạng nơ-ron đồ thị (LightGCN) kết hợp cơ chế bù trừ từ Phương pháp Lai trên các ma trận dữ liệu thưa thực tế.

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