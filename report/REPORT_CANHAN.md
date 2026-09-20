# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Văn Sang  
**MSSV:** 2A202602391  
**Nhóm:** K4 — Biến thể L3B (Chính sách Thương mại Điện tử)  
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Khi hai đoạn văn bản có độ tương tự cosine cao (tiến gần về 1.0), nghĩa là các vector embedding của chúng hướng về cùng một phương trong không gian đa chiều, biểu thị sự tương đồng sâu sắc về mặt ngữ nghĩa (semantic similarity). Độ tương tự bằng 1.0 nghĩa là hướng giống hệt nhau, xấp xỉ 0 là không liên quan (vuông góc), và -1.0 là trái ngược hoàn toàn.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Khách hàng được quyền hoàn lại tiền nếu gửi yêu cầu trong vòng bảy ngày."
- Câu B: "Người mua có thể yêu cầu trả hàng và nhận lại tiền trong thời hạn 7 ngày."
- Tại sao tương đồng: Dù sử dụng từ vựng khác nhau ("khách hàng/người mua", "bảy ngày/7 ngày", "hoàn lại tiền/nhận lại tiền"), hai câu đều diễn đạt chính xác cùng một quy định pháp lý/chính sách.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Quy định bảo hành sản phẩm điện tử cho người bán."
- Câu B: "Thời tiết hôm nay tại Hà Nội trời nhiều mây và có mưa rào."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn tách biệt (thương mại điện tử vs. khí tượng học), không có bất kỳ liên kết ngữ nghĩa hay từ khóa chung nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc trực tiếp vào độ dài hình học (magnitude) của vector, vốn bị chi phối mạnh bởi độ dài câu văn hoặc số lượng từ lặp lại. Ngược lại, Cosine similarity chỉ đo góc giữa hai vector (chuẩn hóa về hướng), giúp so sánh nội dung ngữ nghĩa công bằng giữa các văn bản ngắn và dài mà không bị nhiễu bởi kích thước.

---

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:
> $$\text{số lượng chunk} = \text{ceil}\left(\frac{\text{độ\_dài} - \text{overlap}}{\text{chunk\_size} - \text{overlap}}\right) = \text{ceil}\left(\frac{10000 - 50}{500 - 50}\right) = \text{ceil}\left(\frac{9950}{450}\right) = \text{ceil}(22.11) = 23$$
> **Đáp án:** 23 chunks.  
> *(Kiểm chứng thực tế bằng FixedSizeChunker trong mã nguồn cho đúng 23 chunk)*.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Phép tính khi overlap = 100: $\text{ceil}((10000 - 100) / (500 - 100)) = \text{ceil}(9900 / 400) = \text{ceil}(24.75) = 25$ chunks (tăng thêm 2 chunks).  
> Người ta muốn tăng độ chồng chéo khi văn bản có nhiều thông tin ngữ cảnh liên tục, nhằm ngăn chặn việc các câu văn hoặc từ khóa định danh quan trọng bị cắt đôi ngay ranh giới giữa 2 chunk, giúp duy trì tính liên tục của ngữ cảnh khi đưa vào mô hình tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng biểu thức chính quy lookbehind `re.compile(r'(?<=[.!?])\s+')` để tách ranh giới câu mà không bị nuốt mất dấu câu ở cuối. Gom nhóm các câu thành danh sách các chuỗi con theo `max_sentences_per_chunk` và `strip()` sạch khoảng trắng.  
> *Edge case còn tồn tại:* Các từ viết tắt có dấu chấm (như "TS.", "v.v.", "NĐ-CP") hoặc các số thập phân ("3.5") vẫn có thể bị regex xem là kết thúc câu và cắt sai.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Áp dụng thuật toán chia đệ quy hai chiều: Ưu tiên phân tách theo các ranh giới ngữ nghĩa lớn trước (`["\n\n", "\n", ". ", " ", ""]`). Nếu các mảnh con quá nhỏ, thuật toán tiến hành gom các mảnh liền kề lại cho tới sát ngưỡng `chunk_size` để tránh sinh ra các chunk vụn. Base case dừng khi văn bản nhỏ hơn hoặc bằng `chunk_size`, hoặc khi danh sách separator rỗng thì cắt thuần túy theo độ dài ký tự.

---

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ hoàn toàn trên bộ nhớ trong (in-memory) bằng danh sách các bản ghi chuẩn hóa `dict`. Trong `_make_record`, sử dụng `metadata.setdefault("doc_id", doc.id)` để bảo toàn `doc_id` của file gốc nếu bên ngoài đã định nghĩa (tránh ghi đè cứng bằng `doc.id` dạng "file#0", "file#1"...). Khi tìm kiếm (`search`), truy vấn được nhúng và tính tích vô hướng (dot product) trực tiếp với các vector embedding trong kho (do vector đã được chuẩn hóa L2 nên dot product chính là cosine similarity), sau đó sắp xếp giảm dần lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc thực hiện **lọc siêu dữ liệu trước (pre-filtering)** trên toàn bộ kho rồi mới xếp hạng tìm kiếm; nếu xếp hạng trước rồi mới lọc thì các slot top-k có thể bị chiếm hết bởi tài liệu sai đối tượng. `delete_document` lọc bỏ toàn bộ các chunk có `metadata['doc_id'] == doc_id` và cập nhật lại danh sách nội bộ.

---

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Trích xuất top-k chunk liên quan nhất từ `store`, định dạng ngữ cảnh rõ ràng có đánh số thứ tự kèm trích dẫn nguồn `[1]`, `[2]`. Tạo prompt yêu cầu mô hình chỉ trả lời dựa trên ngữ cảnh được cung cấp, nếu không có thông tin thì nói rõ "không tìm thấy", đảm bảo tính truy vết nguồn gốc (Source Traceability).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Vin\LAB7 chiều\K4-DAY07-LeVanSang-2A202602391
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.13s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (Mock) | Đúng với thực tế? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | Chính sách đổi trả hàng của người mua | Quy trình hoàn tiền cho khách hàng | cao | -0.1193 | ❌ |
| 2 | Người bán phải cung cấp bảo hành sản phẩm | Seller must provide product warranty | cao | 0.0244 | ❌ |
| 3 | Con mèo ngồi trên tấm thảm | Thời tiết hôm nay rất đẹp | thấp | -0.1914 | ✅ |
| 4 | Hướng dẫn đăng ký tài khoản người mua | Cách tạo tài khoản mua sắm mới | cao | 0.1843 | ✅ |
| 5 | Quy định vận chuyển và giao hàng | Điều khoản bảo mật thông tin cá nhân | thấp | -0.0633 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Điểm số thực tế của Cặp 1 và Cặp 2 gây bất ngờ vì hai câu rõ ràng đồng nghĩa nhưng điểm cosine lại mang giá trị âm hoặc xấp xỉ 0. Nguyên nhân là do `MockEmbedder` sử dụng hàm băm MD5 để sinh vector giả ngẫu nhiên nên hoàn toàn không học được ngữ nghĩa. Điều này khẳng định rằng trong hệ thống RAG thực tế, việc sử dụng các mô hình Transformer đã qua tiền huấn luyện (như multilingual MiniLM hay OpenAI) là bắt buộc để phản ánh đúng trường ngữ nghĩa của ngôn ngữ.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá chính thức của nhóm** bằng chiến lược **`RecursiveChunker(chunk_size=500)`** (kết quả trích xuất trực tiếp từ file `ket_qua_benchmark.txt` sinh ra 40 chunks):

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (doc_id & tóm tắt) | Điểm Score | Có liên quan? | Ghi chú & Đối chiếu Gold Answer |
|---|:---|:---|:---:|:---:|:---|
| 1 | Người mua cần gửi yêu cầu trả hàng trong bao lâu nếu sản phẩm là thực phẩm tươi sống hoặc đông lạnh? | `tiki-return-policy-buyer#2`: Sản phẩm mua theo hình thức giao hàng từ nước ngoài không được áp dụng... | 0.267 | ❌ Chưa | Gold doc (`return-refund-policy#1`) nằm ở top-3. Mock embedding làm lệch thứ hạng truy xuất. |
| 2 | Trên Tiki, thời hạn đổi trả chung là bao nhiêu ngày kể từ khi giao hàng thành công? | `shopee-marketplace-operating-rules#2`: Sàn khuyến cáo người mua cần kiểm tra chính sách bảo hành... | 0.275 | ❌ Chưa | Top-1 bị lệch sang quy chế Shopee do MockEmbedder không hiểu ngữ nghĩa từ khóa "Tiki". |
| 3 | Điều kiện để một yêu cầu bảo hành sản phẩm được chấp nhận là gì? *(Có filter `audience: seller`)* | `seller-warranty-policy#3`: Sàn không phải là bên trực tiếp thực hiện nghĩa vụ bảo hành... | 0.160 | ✅ Có | **Thành công nhờ filter:** Toàn bộ top-1, 2, 3 đều thuộc đúng tài liệu `seller-warranty-policy` (chunk #0 chứa điều kiện bảo hành lọt vào top-3). Khi tắt filter, top-1 bị nhầm sang chính sách Lazada của người mua. |
| 4 | Người bán có quyền gì khi sàn quyết định hoàn tiền cho người mua mà không yêu cầu gửi trả sản phẩm? | `seller-warranty-policy#2`: Yêu cầu bảo hành sẽ bị từ chối hoặc phát sinh phí khi vi phạm... | 0.247 | ❌ Chưa | Gold doc (`shopee-warranty-electronics-general-rules`) chưa lọt top-3 trong lượt chạy Mock. |
| 5 | Người bán phải cung cấp những thông tin định danh nào khi đăng ký sử dụng dịch vụ sàn giao dịch thương mại điện tử? | `return-refund-policy#2`: Khi trả hàng vì lý do không còn nhu cầu, sản phẩm gửi trả phải giữ nguyên... | 0.230 | ⚠️ Một phần | Gold doc (`seller-legal-obligations-nd52-2013#1`) xếp ở vị trí **top-2** với score=0.214, chứa thông tin định danh. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5 (Câu 3 và Câu 5 đạt top-2 và top-3 chứa đúng tài liệu chuẩn).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Nhóm trưởng dùng `HeadingChunker` sinh ra 48 chunks bám sát từng đề mục của chính sách, trong khi `RecursiveChunker` của tôi sinh ra 40 chunks do gộp các đoạn văn lại cho đủ 500 ký tự. Đặc biệt, thí nghiệm A/B ở Câu 3 chứng minh rõ ràng: nếu không có `metadata_filter={'audience': 'seller'}`, hệ thống sẽ bị lẫn lộn giữa quy định của người bán và người mua.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---|:---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — 42/42 tests pass) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results & Phân tích A/B) | 9 / 10 |
| **Tổng phần cá nhân** | **59 / 60** |
