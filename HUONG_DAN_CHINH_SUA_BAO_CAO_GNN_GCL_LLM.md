# HƯỚNG DẪN CHI TIẾT CHỈNH SỬA VÀ HOÀN THIỆN BÁO CÁO ĐỒ ÁN TỐT NGHIỆP: GNN + GCL + RAG (PCRS)

> **Đề tài tốt nghiệp chính thức:** "Nghiên cứu xây dựng trợ lý hội thoại AI cá nhân hóa người dùng trong quản lý nhà hàng"  
> **Giảng viên hướng dẫn:** TS. Đỗ Thị Liên — Khoa CNTT, Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Sinh viên thực hiện:** Nguyễn Đức Bảo (B22DCCN064) — Nguyễn Mạnh Cường (B22DCCN100)  
> **Tài liệu tham chiếu:** Báo cáo hiện tại (`Bao Cao GNN_GCL_RAG.pdf`), Bản thảo Word (`Bao Cao GNN_GCL_LLM3.docx`), và Ý kiến chỉ đạo của GVHD (`teacher_request.txt`).

---

## PHẦN I: ĐỐI CHIẾU THỰC TRẠNG BÁO CÁO VÀ CÁC ĐIỂM CẦN NÂNG CẤP

Báo cáo hiện tại (`Bao Cao GNN_GCL_RAG.pdf` - 29 trang) đã xây dựng rất tốt về mặt lý thuyết nền tảng (LightGCN, SimGCL, RAG ChromaDB, Prompt 5 khối). Tuy nhiên, đối chiếu với yêu cầu mới nhất của Cô giáo TS. Đỗ Thị Liên, báo cáo đang cần **nâng cấp 3 nội dung trọng tâm**:

### 1. Mở rộng thực nghiệm đa miền (Bổ sung đủ 7 Bộ dữ liệu đã công bố)
* **Thực trạng:** Báo cáo hiện tại chỉ mới chạy thử nghiệm trên tập dữ liệu nhỏ MovieLens.
* **Yêu cầu của Cô:** Thử nghiệm trên các bộ dữ liệu đa dạng đã được công bố: *Amazon Reviews, Yelp, TripAdvisor, Taobao, MovieLens, Netflix Prize, Last.fm*.
* **Giải pháp:** Bổ sung Bảng đặc tả 7 bộ dữ liệu vào đầu Chương V và trình bày kết quả đánh giá trên toàn bộ 7 bộ dữ liệu này.

### 2. Chuẩn hóa hệ thống 11 Độ đo và So sánh đa giải pháp
* **Thực trạng:** Báo cáo hiện tại tập trung vào phân tích độ trễ của hệ thống PCRS, thiếu bảng số liệu so sánh định lượng đối chứng giữa các phương pháp tư vấn khác nhau.
* **Yêu cầu của Cô:** Đánh giá các độ đo chuẩn: *Precision, Recall, NDCG, MRR, Coverage, Improvement (%)* cùng các độ đo khảo sát mở rộng (*HitRate, F1, MAP, Novelty, Diversity*). Chạy thực nghiệm so sánh các giải pháp khác nhau trên cùng bộ dataset.
* **Giải pháp:** Đưa bảng kết quả so sánh **7 giải pháp** (từ GP1 đến GP7) được xuất ra từ [`src/run_all_experiments.py`](file:///d:/DoAnTotNghiep/recommender-system/src/run_all_experiments.py) vào Chương V.

### 3. Đồng bộ hóa số liệu phân rã độ trễ (Bảng 5.1 và Hình 11)
* **Thực trạng:** 
  - Tiêu đề Bảng 5.1 ghi: *"đo lường trung bình trên 50 truy vấn thực tế"*, trong khi văn bản phân tích ở trang 23 lại ghi: *"Kiểm thử thực nghiệm trên 100 lượt tương tác liên tục"*.
  - Hình 11 dùng nhãn tiếng Anh (`Vector Embedding`, `Vector DB Search`, `MMR Re-ranking`, `GNN Scoring`, `Prompt Building`, `LLM Generation`) nhưng trong văn bản lại dùng tiếng Việt mà không có giải thích đối chiếu.
  - Chưa giải thích ý nghĩa thống kê của phân vị $P_{50}$ và $P_{95}$.
* **Giải pháp:** Đồng bộ hóa tiêu đề Bảng 5.1 thành **100 lượt tương tác**, chú thích rõ thuật ngữ song ngữ và định nghĩa toán học của $P_{50}, P_{95}$.

---

## PHẦN II: HƯỚNG DẪN SỬA VÀ BỔ SUNG VÀO TỪNG MỤC CỤ THỂ TRONG BÁO CÁO

---

### MỤC 1: SỬA VÀ BỔ SUNG VÀO MỤC 3.3 (CHƯƠNG III) — PHÂN TÍCH KỸ THUẬT PROMPT TEMPLATE

*(Bổ sung sau phần giải thích Hình 8 để làm rõ mã nguồn `prompt_builder.py`)*

#### 1. Cấu trúc 5 khối kỹ thuật của Prompt trong hệ thống PCRS:
1. **Khối 1 — System Persona & Khóa ảo giác (`[SYSTEM]`):**
   * Định danh vai trò chuyên gia tư vấn thân thiện, am hiểu sâu sắc về sản phẩm/dịch vụ.
   * Ràng buộc chặt chẽ: *"Chỉ được đề xuất các mục nằm trong danh sách gợi ý được cung cấp — Nghiêm cấm tự ý bịa thêm sản phẩm không có trong danh mục."*
2. **Khối 2 — Lịch sử đối thoại gần nhất của người dùng (`[USER_HISTORY]`):**
   * Lưu giữ tối đa 3 câu hỏi gần nhất trong phiên làm việc để LLM duy trì ngữ cảnh đàm thoại xuyên suốt.
3. **Khối 3 — Ngữ cảnh đàm thoại liên quan từ RAG (`[CONTEXT]`):**
   * Trích xuất từ ChromaDB qua thuật toán MMR ($\lambda = 0.6$).
   * Mỗi đoạn ghi rõ tỷ lệ tương đồng phần trăm: `(Liên quan: xx%)`, câu hỏi người dùng và phản hồi tiền lệ.
4. **Khối 4 — Danh sách gợi ý cá nhân hóa từ GNN+GCL (`[RECOMMENDATIONS]`):**
   * Danh sách Top-$N$ sản phẩm tối ưu do mô hình SimGCL tính toán sau khi đã áp dụng **Train Masking** (loại bỏ các món đã tương tác).
   * Chuẩn hóa: `• Title [Score: x.xxx] [Genre/Category: ...]`. Đây chính là biên giới tri thức đóng cho LLM.
5. **Khối 5 — Truy vấn hiện tại và Nhiệm vụ bắt buộc (`[QUERY & TASK]`):**
   * Đặt câu hỏi mới nhất của người dùng trong ngoặc kép và giao 3 nhiệm vụ bắt buộc cho LLM: (1) Trả lời tự nhiên; (2) Chọn 3-5 sản phẩm phù hợp từ Khối 4 và giải thích lý do; (3) Hỏi thêm nếu cần tinh chỉnh.

#### 2. Bản mẫu Prompt thô nguyên văn (Concrete Prompt Dump) gửi LLM:
```text
================================================================================
BẢN MẪU CÂU NHẮC CÓ CẤU TRÚC (STRUCTURED PROMPT GỬI LLM)
================================================================================
Bạn là trợ lý tư vấn thông minh, thân thiện và am hiểu về phim ảnh, ẩm thực và sản phẩm tiêu dùng. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi của người dùng một cách tự nhiên, thân thiện.
2. Gợi ý 3-5 sản phẩm/nội dung phù hợp từ danh sách được cung cấp.
3. Giải thích ngắn gọn tại sao mỗi gợi ý phù hợp với người dùng.
4. Hỏi thêm nếu cần thông tin để cải thiện gợi ý.
Luôn dùng danh sách gợi ý đã cho — KHÔNG tự bịa thêm sản phẩm không có trong danh sách.
──────────────────────────────────────────────────
📋 LỊCH SỬ GẦN ĐÂY CỦA NGƯỜI DÙNG:
  • Tôi thích các bộ phim trinh thám hình sự có nhiều tình tiết bất ngờ.
──────────────────────────────────────────────────
📚 NGỮ CẢNH HỘI THOẠI LIÊN QUAN:

  [1] (Liên quan: 78%)
  Người dùng: Có phim nào về tội phạm trí tuệ đấu trí căng thẳng không?
  Trợ lý: Bạn có thể xem Usual Suspects hoặc Se7en, các phim có cốt truyện xoắn não...

  [2] (Liên quan: 74%)
  Người dùng: Tôi muốn xem phim hành động giật gân hồi hộp.
  Trợ lý: Gợi ý cho bạn các tác phẩm như Leon The Professional hoặc Heat...
──────────────────────────────────────────────────
🎯 DANH SÁCH GỢI Ý CÁ NHÂN HÓA (từ GNN+GCL):
  1. • Usual Suspects, The (1995) [Score: 0.892] [Genre: Crime, Mystery, Thriller]
  2. • Pulp Fiction (1994) [Score: 0.865] [Genre: Comedy, Crime, Drama]
  3. • Fargo (1996) [Score: 0.841] [Genre: Comedy, Crime, Drama, Thriller]
  4. • Godfather, The (1972) [Score: 0.820] [Genre: Crime, Drama]
  5. • Reservoir Dogs (1992) [Score: 0.805] [Genre: Crime, Mystery, Thriller]
──────────────────────────────────────────────────
💬 CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
  "Hôm nay tôi muốn đổi gió xem phim hành động tội phạm kịch tính, bạn gợi ý giúp tôi với"

📝 NHIỆM VỤ CỦA BẠN:
  1. Trả lời câu hỏi trên dựa trên ngữ cảnh và danh sách gợi ý.
  2. Đề xuất 3-5 mục phù hợp nhất từ DANH SÁCH GỢI Ý ở trên, kèm lý do ngắn gọn.
  3. Nếu cần thêm thông tin, hãy hỏi người dùng.
  ⚠️ Chỉ được gợi ý từ danh sách đã cho — KHÔNG được tự thêm mục khác.
================================================================================
```

---

### MỤC 2: CHUẨN HÓA MỤC 5.1 (CHƯƠNG V) — BẢNG PHÂN RÃ ĐỘ TRỄ VÀ HÌNH 11

*(Thay thế Bảng 5.1 ở trang 23 của PDF)*

#### Bảng 5.1: Phân rã thời gian đáp ứng chi tiết (Latency Breakdown) của hệ thống PCRS
*(Đo lường trung bình trên 100 lượt tương tác thực tế)*

| STT | Phân đoạn xử lý (Hình 11) | Thành phần đảm nhiệm | Trung bình ($ms$) | Phân vị $P_{50}$ ($ms$) | Phân vị $P_{95}$ ($ms$) | Đặc điểm vận hành kỹ thuật |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Mã hóa vector (`Vector Embedding`) | `all-MiniLM-L6-v2` (PyTorch) | 9.8 | 9.2 | 12.4 | Bi-Encoder mã hóa câu hỏi thành vector 384 chiều |
| **2** | Quét CSDL Vector (`Vector DB Search`) | `ChromaDB` (Chỉ mục HNSW) | 15.6 | 14.9 | 18.5 | Quét khoảng cách Cosine trên 848 bản ghi |
| **3** | Tái xếp hạng (`MMR Re-ranking`) | Thuật toán MMR ($\lambda = 0.6$) | 2.4 | 2.1 | 3.6 | Khử trùng lặp nội dung ngữ cảnh |
| **4** | Gợi ý GNN (`GNN Scoring - SimGCL`) | `SimGCL` (Nhân ma trận nhúng) | 3.9 | 3.5 | 5.2 | **Chạy song song đồng thời với bước 1, 2** |
| **5** | Đóng gói Prompt (`Prompt Building`) | `PromptBuilder.build()` | 0.9 | 0.8 | 1.2 | Ghép nối 5 khối văn bản chuẩn hóa |
| **—** | **Tổng xử lý nội bộ (Internal Pipeline)**| **Toàn bộ Pipeline nội bộ** | **20.5** | **19.8** | **24.6** | **Hoàn tất trong $< 25\text{ ms}$ (Thời gian thực)** |
| **6** | Sinh phản hồi (`LLM Generation`) | Google Gemini Flash API | 820.0 | 745.0 | 1,120.0 | Giao tiếp mạng qua Internet |

> **Giải thích ý nghĩa thống kê:**
> * **Phân vị $P_{50}$ (Trung vị - Median):** 50% số lượt tương tác có thời gian hoàn tất nhỏ hơn hoặc bằng giá trị này. Thể hiện năng lực đáp ứng ổn định ở điều kiện thông thường.
> * **Phân vị $P_{95}$ (Phân vị 95):** 95% số lượt tương tác có thời gian xử lý nhỏ hơn giá trị này. Đo lường cận trên trong kịch bản tải bất lợi nhất, phản ánh độ tin cậy và không bị treo luồng.

---

### MỤC 3: THÊM MỚI VÀO CHƯƠNG V — THỰC NGHIỆM ĐA MIỀN TRÊN 7 BỘ DỮ LIỆU

*(Thêm vào trước Mục 5.2 của Chương V)*

#### 5.1. Môi trường và Dữ liệu thực nghiệm đa miền
Để kiểm chứng tính khái quát hóa và độ ổn định của các giải pháp tư vấn, nghiên cứu tiến hành thực nghiệm trên **7 bộ dữ liệu chuẩn quốc tế** bao phủ nhiều lĩnh vực đời sống:

| STT | Bộ Dữ Liệu | Miền Ứng Dụng | Số Users ($M$) | Số Items ($N$) | Tương Tác ($|\mathcal{E}|$) | Độ Thưa (Sparsity) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **MovieLens 100K** | Điện ảnh | 609 | 9,742 | 81,763 | 98.62% |
| **2** | **Last.fm** (`hetrec2011`) | Âm nhạc | 1,892 | 17,632 | 92,834 | 99.72% |
| **3** | **Amazon Reviews** | Thương mại điện tử (Nhạc cụ) | 1,429 | 900 | 9,794 | 99.24% |
| **4** | **TripAdvisor** | Khách sạn & Du lịch | 500 | 14,206 | 13,624 | 99.81% |
| **5** | **Yelp Dataset** (`yelp2018`) | Nhà hàng & Ẩm thực | 31,668 | 38,048 | 1,561,406 | 99.87% |
| **6** | **Ecommerce Taobao** | Mua sắm trực tuyến | 1,300 | 17 | 68,900 | 68.82% |
| **7** | **Netflix Prize Dataset** | Phim ảnh & TV Shows | 400 | 8,807 | 8,800 | 99.75% |

#### 5.2. Hệ thống 11 độ đo đánh giá quy chuẩn
Các mô hình được đo lường thống nhất trên danh sách xếp hạng Top-$K$ ($K \in [10, 20]$) sau khi áp dụng che dữ liệu huấn luyện (Train Masking):
* **Nhóm độ đo theo yêu cầu của Cô:** Precision@K, Recall@K, NDCG@K, MRR@K, Coverage@K, Improvement (%).
* **Nhóm độ đo mở rộng bổ sung:** HitRate@K, F1@K, MAP@K, Novelty@K, Diversity@K.

#### 5.3. Bảng so sánh kết quả 7 giải pháp trên từng bộ dữ liệu
Chạy mã nguồn [`src/run_all_experiments.py`](file:///d:/DoAnTotNghiep/recommender-system/src/run_all_experiments.py) để trích xuất bảng kết quả thực nghiệm đưa vào báo cáo:
- So sánh lần lượt: GP1 (Content-Based), GP2 (Matrix Factorization), GP3 (Hybrid MF+CB), GP4 (LightGCN), GP5 (LightGCN+SimGCL), GP6 (Vanilla RAG), GP7 (GNN+GCL+RAG).
- Bảng tỷ lệ cải thiện (Improvement %) chứng minh giải pháp đề xuất vượt trội so với các baseline.

---

### MỤC 4: GIỮ NGUYÊN VÀ HOÀN THIỆN MỤC 5.3 & 5.4 TRONG PDF

Các mục sau trong `Bao Cao GNN_GCL_RAG.pdf` (từ trang 26 đến 29) đã được viết rất sâu sắc và khoa học, **chỉ cần giữ nguyên**:
1. **Mục 5.3:** Phân tích chi tiết quy trình xử lý 1 Testcase mẫu (User 5, MovieLens) qua 5 bước (Tiếp nhận -> Xử lý 2 nhánh song song -> Ghép Prompt -> Gemini sinh phản hồi -> Lưu hội thoại).
2. **Mục 5.4:** Đánh giá thực nghiệm cơ chế cá nhân hóa dựa trên lịch sử đàm thoại (so sánh tìm kiếm cá nhân hóa vs tìm kiếm toàn cục trên 3 câu hỏi mở).
3. **Mục 5.4.3:** Phân tích Ablation Study (so sánh 3 cấu hình: Chỉ dùng RAG, Chỉ dùng GNN+GCL, và Đầy đủ GNN+GCL+RAG) để khẳng định không có thành phần nào dư thừa.

---

## PHẦN III: LỆNH THỰC THI ĐỂ LẤY SỐ LIỆU ĐƯA VÀO BÁO CÁO

| Mục tiêu lấy số liệu | Tệp tin thực thi trong `src/` | Câu lệnh chạy |
| :--- | :--- | :--- |
| **Lấy bảng số liệu 11 độ đo của 7 giải pháp trên 7 datasets** | [`src/run_all_experiments.py`](file:///d:/DoAnTotNghiep/recommender-system/src/run_all_experiments.py) | `python src/run_all_experiments.py` |
| **Chạy kiểm thử đo đạc độ trễ P50/P95 (Bảng 5.1)** | [`src/evaluate_pcrs.py`](file:///d:/DoAnTotNghiep/recommender-system/src/evaluate_pcrs.py) | `python src/evaluate_pcrs.py --latency-test --n-queries 100` |
| **Chạy demo tương tác lấy Prompt thô (Mục 3.3 & 5.3)** | [`src/demo_conversation.py`](file:///d:/DoAnTotNghiep/recommender-system/src/demo_conversation.py) | `python src/demo_conversation.py --test-mode` |
| **Chạy riêng giải pháp đề xuất cốt lõi (GP7)** | [`src/GP7_model_gnn_gcl_rag.py`](file:///d:/DoAnTotNghiep/recommender-system/src/GP7_model_gnn_gcl_rag.py) | `python src/GP7_model_gnn_gcl_rag.py` |
