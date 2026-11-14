# Báo cáo cá nhân — Day 06: Data Ingestion

- Họ tên: **Hồ Nguyễn Minh Sang**.
- MSSV: **2A202603905**; lớp: **L3B**.
- Chủ đề: truy xuất kiến thức Python từ Python 3.13 Tutorial.
- Công cụ hỗ trợ: Codex hỗ trợ viết code, soạn benchmark, chạy thí nghiệm, đọc nguồn, review và dự thảo báo cáo. Nhãn định tính ban đầu do AI hỗ trợ chấm; học viên đã xác nhận xem xét audit trong phiên làm việc ngày 14/11/2026, không yêu cầu sửa nhãn. Đây không phải chấm mù độc lập.

## 1. Dữ liệu và pipeline

Corpus gồm sáu chương `controlflow`, `datastructures`, `modules`, `inputoutput`, `errors`, `classes` của Python 3.13 Tutorial. Snapshot HTML gốc và trang giấy phép nằm trong `data/raw/`. Tài liệu dùng PSF License Version 2; ví dụ code còn có lựa chọn 0BSD. Giấy phép, ghi công, thay đổi và cách tái tạo được lưu trong `data/ATTRIBUTION.md`, `data/LICENSE_PYTHON.txt`, `data/manifest.json`.

HTML không có phân trang vật lý. Mỗi chương được trích phần nội dung chính, chuẩn hóa Unicode NFC và khoảng trắng văn xuôi, giữ thụt lề của code; sau đó chia tuần tự tại ranh giới đoạn sau mốc 3.500 ký tự, gộp phần cuối nếu ngắn hơn 1.500. Kết quả **42 đơn vị nguồn hợp lệ**, tổng **155.072 ký tự**, mỗi đơn vị ≥1.500 ký tự. `page` là ID đơn vị trong chương, không phải số trang PDF. Manifest ghi start/end, số ký tự, URL, thời điểm tải, SHA-256 HTML và nội dung chuẩn hóa. Không nhân bản để đạt số lượng.

Pipeline: đọc snapshot → chuẩn hóa/đơn vị nguồn → chunk → embedding local → PostgreSQL/pgvector → truy xuất top-3 → chấm evidence → xuất trace và mẫu audit. Hàm đọc PDF cũng được hiện thực để hỗ trợ corpus khác, nhưng lần chạy này dùng HTML, không dùng OCR. Các lỗi PDF/rỗng được ghi qua logging; corpus hiện tại không có đơn vị rỗng hợp lệ bị bỏ qua âm thầm.

Một lỗi chuẩn hóa làm mất thụt lề code đã được phát hiện và sửa **trước khi chạy retrieval**; đã thêm regression test và tái tạo corpus/gold trước khi chốt thí nghiệm. Docker ban đầu lỗi route tới VM; khởi động lại Docker Desktop giúp database hoạt động. Log lỗi kết nối thật được khôi phục nguyên bản từ lịch sử Git trong `results/errors.json`. Pipeline không fallback sang vector giả hay tính similarity thuần Python.

### Model và database

- Python **3.13.5**, FastEmbed **0.8.0**, ONNX Runtime **1.27.0**, psycopg **3.3.9**.
- Model **BAAI/bge-small-en-v1.5**, **384 chiều**, chạy CPU local, 4 threads; bản ONNX quantized do FastEmbed phân phối từ `qdrant/bge-small-en-v1.5-onnx-q`, revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`.
- Query thêm prefix `Represent this sentence for searching relevant passages: `; document không thêm prefix. Giữ nguyên giữa A/B/C. Model/tokenizer có hash trong `results/environment.json`.
- PostgreSQL **16.16**, pgvector **0.8.7**, image `pgvector/pgvector:0.8.7-pg16`; table `lab6_chunks`, phân biệt bằng `strategy`.
- Cosine distance `<=>`, tìm kiếm **exact**; không có approximate vector index. Metadata JSONB có GIN index.
- Lưu bền vững bằng volume Docker `day6_student_starter_pg_data`, cổng `127.0.0.1:55433`.
- **Chi phí API: 0 USD**; không dùng Groq/API. Không quy đổi điện và khấu hao máy thành chi phí. Lần đầu cần tải model khoảng 67 MB; cache không đưa vào repo.

Lệnh setup/run đầy đủ nằm trong RUNNING.md. Từ root: `.venv/Scripts/python.exe -m starter.src.run validate`, sau đó `... -m starter.src.run experiment`. Từ `starter/` có thể chạy `../.venv/Scripts/python.exe -m src.run`.

### Metadata

Mỗi chunk có `document_id` (string), `source` (URL string), `page` (integer), `start/end` (integer, offset trong đơn vị), `unit_start/unit_end` (integer, offset trong chương), `chapter`, `language=en`, `version=3.13` (string). Các trường được lấy từ nguồn/ranh giới đã tính, không đoán ngày hiệu lực. `chapter` có sáu giá trị phân biệt và giúp giới hạn phạm vi theo chương được người dùng cung cấp. `language` và `version` hiện hằng số nên không đủ để chứng minh ích lợi lọc trên corpus này.

## 2. Ba cấu hình và benchmark

| Cấu hình | Chunk | Filter |
|---|---|---|
| A | Cửa sổ 500 ký tự, overlap 50 | Không |
| B | Gom theo đoạn, giới hạn 1.000 ký tự; đoạn quá dài vẫn phải cắt | Không |
| C | Cùng chunk và vector của B | Lọc `chapter` khi câu hỏi có filter |

Corpus/model/store/cosine/query/benchmark giống nhau. A–B so sánh cả chính sách chunking, bao gồm kích thước và overlap; không tách riêng ảnh hưởng của từng tham số. B–C chỉ đổi metadata filter. C tái dùng embedding của B để loại biến số và tiết kiệm tính toán; thời gian C không đại diện cho một lần ingest lạnh độc lập.

Benchmark được soạn trước truy xuất: **20 câu tiếng Anh, 15 dev / 5 test**. Test: q04, q08, q12, q16, q20. Có 4 câu filter (q06, q11, q16, q20), gồm 2 test. Tên chương có ngay trong câu hỏi; filter là input trước truy xuất, không lấy gold source/answer để tạo filter. Evidence nguyên đoạn được đối chiếu với nguồn chuẩn hóa. Chia theo ID để phủ các chương nên vẫn có nguy cơ tương đồng chủ đề giữa dev và test.

Quy tắc chọn chốt trước: **dev Evidence Hit@3 → dev Hit@1 → ít chunk hơn → thứ tự tên cấu hình**. Lần chạy lại của học viên bắt đầu `2026-11-14T04:29:20Z`; `results/selection.json` ghi chọn **B** lúc `2026-11-14T04:30:50Z`, trước bước test, kết thúc `04:30:51Z`. B và C bằng điểm và số chunk nên B thắng tie-break theo tên. Không điều chỉnh cấu hình theo kết quả test; lần chạy lại dùng `--allow-rerun`, không coi bộ test đã xem là holdout độc lập mới.

**Hiệu chỉnh gold công khai:** sau lần chạy đầu, review phát hiện q12 có evidence hợp lệ khác ở `errors/5`. Đã thêm alternate và chấm lại **cùng trace**, giữ mọi query/hit/ranking/config. A test Hit@3 đổi 0/5 → 1/5; B/C và dev không đổi. Gold v1, trace/metrics gốc và hash trước/sau đều được giữ ở `benchmark/questions_v1.json`, `results/original_scoring/`, `results/scoring_amendment.json`. Kết quả hiệu chỉnh **không được coi là một holdout hoàn toàn chưa bị tác động**. Chi tiết: `benchmark/AMENDMENTS.md`.

## 3. Kết quả truy xuất và tài nguyên

Hit tự động cần đúng nguồn/đơn vị và chứa đủ quote trong **một chunk**, chấp nhận alternate đã ghi. So khớp chuẩn hóa Unicode/khoảng trắng, không phân biệt hoa/thường. Không gộp evidence nhiều chunk để thay định nghĩa Hit@3 bắt buộc. Đây là proxy bảo thủ: mất vài ký tự đầu hoặc có bằng chứng tương đương chưa ghi có thể bị tính miss dù nội dung còn hữu ích; cần đọc cùng grounding.

| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunk count | Ingest giây | API USD |
|---|---|---|---|---:|---:|---:|
| A | dev | 7/15 (46,7%) | 9/15 (60,0%) | 359 | 40,46 | 0 |
| A | test | 0/5 (0%) | 1/5 (20,0%) | 359 | — | 0 |
| B | dev | 13/15 (86,7%) | 14/15 (93,3%) | 198 | 44,46 | 0 |
| B | test | 5/5 (100%) | 5/5 (100%) | 198 | — | 0 |
| C | dev | 13/15 (86,7%) | 14/15 (93,3%) | 198 | 0,29* | 0 |
| C | test | 5/5 (100%) | 5/5 (100%) | 198 | — | 0 |

Ingest trong bảng lấy từ lần chạy lại ngày 14/11/2026, đo một lần cho mỗi cấu hình, không lặp lại theo split; `*` C tái sử dụng embedding B. Tổng ingest gồm chunk/token thống kê, embedding và lần ghi đầu, không gồm tải model/khởi tạo runtime, kiểm tra nạp lại hoặc thời gian xử lý sự cố Docker. Thời gian embedding A **39,77 s**, B **44,04 s**, C tái dùng; thời gian ghi đầu tương ứng **0,63 / 0,37 / 0,20 s**. B ít chunk hơn nhưng chunk dài hơn và lần đo CPU vẫn chậm hơn A; chưa đủ phép đo có kiểm soát để kết luận tốc độ tổng quát. Lần chạy lại giữ nguyên điểm, thứ hạng và nội dung hits so với kết quả đã hiệu chỉnh trước đó.

| Strategy | Độ dài ký tự min / median / max | Tổng ký tự chunk | Tổng token | Token max | Chunk >512 token | Số bản ghi trước/sau nạp lại |
|---|---|---:|---:|---:|---:|---|
| A | 59 / 500 / 500 | 170.922 | 46.261 | 254 | 0 | 359 / 359 |
| B | 44 / 839 / 1.000 | 154.772 | 41.343 | 483 | 0 | 198 / 198 |
| C | 44 / 839 / 1.000 | 154.772 | 41.343 | 483 | 0 | 198 / 198 |

Overlap làm tổng ký tự A lớn hơn corpus; B/C bỏ một số dấu phân cách đoạn giữa chunk nên tổng ký tự hơi nhỏ hơn văn bản nguồn. Không mất nội dung vì truncation model trong lần chạy này. IDs dựa trên nội dung/vị trí; mỗi strategy được thay snapshot trong một transaction, xóa chunk cũ và ghi mới, có advisory lock. Kiểm thử database riêng xác minh thay đổi nguồn làm chunk cũ biến mất và dữ liệu còn đọc được qua client mới.

## 4. Coherence, metadata và grounding

Coherence dùng 10 vị trí nguồn chốt trước, offset giữa đơn vị, chọn chunk bao phủ có start nhỏ nhất. Mẫu là 9 đơn vị Classes và 1 Control Flow nên **thiên lệch về Classes**. Điểm 0 = không hiểu vì mất ý/chủ thể; 1 = hiểu một phần, thiếu ngữ cảnh; 2 = tự hiểu và giữ trọn ý.

Grounding đọc toàn bộ top-3 cho từng câu test: 0 = không evidence, 1 = thiếu điều kiện/ngữ cảnh cần thiết, 2 = đủ trả lời và truy nguồn. Đây là đánh giá evidence retrieval, không phải chất lượng câu trả lời do LLM sinh.

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ/đúng | Câu filter có top-3 tuân thủ | Grounding số câu 0 / 1 / 2 | Grounding trung bình /2 |
|---|---:|---|---|---|---:|
| A | 0,8 | 15/15 | 2/2 | 0 / 1 / 4 | 1,8 |
| B | 1,4 | 15/15 | 2/2 | 0 / 0 / 5 | 2,0 |
| C | 1,4 | 15/15 | 2/2 | 0 / 0 / 5 | 2,0 |

**Phương pháp và người chấm:** Codex đọc các mẫu đã xuất, đối chiếu nguồn và gán nhãn có lý do; metadata còn có kiểm tra tự động text span/hash hỗ trợ. Học viên Hồ Nguyễn Minh Sang đã xác nhận trong hội thoại rằng đã xem xét audit, không yêu cầu sửa nhãn. `results/review_labels.json` giữ reviewer/phương pháp AI-assisted ban đầu và bổ sung `human_review_status=confirmed_by_student`, người xác nhận, căn cứ và thời điểm ghi nhận. Đây là xác nhận do học viên cung cấp, không phải một lượt chấm mù hay đo độ đồng thuận độc lập. Người hỗ trợ triển khai cũng hỗ trợ chấm nên vẫn có thiên lệch.

Metadata được kiểm tra trên 5 test × top-3 mỗi cấu hình, tổng **45 bản ghi**. `source/page/start/end` giúp quay lại đoạn nguồn; `chapter` giới hạn phạm vi ở C. A/B không bật filter nhưng top-3 của hai câu filter test vẫn tự nhiên nằm trong đúng chương, vì vậy 2/2 **không chứng minh A/B có thực thi filter**, cũng không chứng minh C đã cải thiện relevance. Bộ test này chưa thể hiện lợi ích tăng điểm của filter; filter vẫn có ích để bảo đảm phạm vi khi được yêu cầu. Chương quá rộng chưa thay thế được metadata theo section hoặc nhóm người đọc.

Nhãn và ngữ cảnh của 90 mẫu chấm: `results/review_labels.json`. Tổng hợp: `results/audit_summary.csv`; chạy lại bảng tổng hợp bằng `python -m starter.src.audit` sau khi cập nhật nhãn có chủ đích.

## 5. Phân tích lỗi, đánh đổi và kết luận

1. **A, q08, `modules/2`:** đoạn giải thích script directory đứng trước standard library bị tách. Top-3 chỉ còn phần kết luận rằng module thư viện bị thay thế; thiếu lý do về thứ tự tìm kiếm. Hit@3 miss và grounding 1. B/C giữ nguyên đoạn, truy xuất top-1 đúng. Đây là lỗi thiếu ngữ cảnh thực sự.
2. **B/C, q09 dev, `modules/1`:** câu hỏi hỏi module initialization chạy một lần khi import và khi chạy như script, nhưng top-3 nghiêng về `import *`, namespace và phần giới thiệu module. Cũng đúng chương nhưng thiếu đoạn chứa tần suất thực thi; Hit@3 miss. Filter theo chương không sửa được lỗi phân biệt ý trong cùng chương.
3. **A, q04, `controlflow/6`:** top-1 mất chữ đầu của “Important”, nên không chứa nguyên quote đã ghi dù vẫn đủ nội dung về mutable default và ví dụ tích lũy. Hit@1/Hit@3 theo proxy vẫn miss, nhưng grounding 2. Điều này cho thấy không nên diễn giải Hit thấp thành toàn bộ truy xuất không hữu ích; checker nguyên đoạn và ranh giới chunk cùng ảnh hưởng số đo.

**Lựa chọn:** B theo quy tắc dev đã chốt. B/C đạt retrieval cao hơn A trên benchmark này và coherence cao hơn trong mẫu, song B tốn thời gian embedding CPU lớn hơn A ở lần đo. C không tăng điểm so với B trên bộ câu hỏi hiện tại; khi người dùng cung cấp phạm vi thì C vẫn cho cơ chế ràng buộc phạm vi rõ ràng. Chỉ kết luận trong corpus/model/bộ câu hỏi này, không tuyên bố chiến lược tốt nhất tổng quát.

**Giới hạn:** 5 test quá nhỏ; corpus và câu hỏi tiếng Anh; dev/test trùng chủ đề; quote nguyên đoạn có thể tạo false negative do cắt ranh giới; chưa ghi được mọi evidence tương đương; q12 có hiệu chỉnh gold sau lần chạy; lần tái lập dùng test đã xem; nhãn AI-assisted đã được học viên xác nhận nhưng không chấm mù; mẫu coherence thiên lệch; bảng tốc độ lấy một lần chạy lại, không có thiết kế đo lặp cold/warm có kiểm soát, có tái sử dụng vector C và cache query giữa cấu hình. `elapsed_seconds` của query gồm embedding khi chưa có cache; chỉ `search_seconds` đo riêng truy xuất database, không so trực tiếp elapsed giữa A/B/C. Không có đánh giá câu trả lời sinh, OCR hay hybrid retrieval.

Nếu mở rộng: chuẩn bị test mới độc lập và gold nhiều nguồn đã kiểm chứng trước chạy; chọn mẫu coherence phủ nhiều chương; chấm bằng hai người; thử chunk theo heading/sentence với ngân sách token được kiểm soát; đo lặp lại cold/warm; thử hybrid trên dev và giữ test mới cố định. Không triển khai các cải tiến này bằng cách chỉnh theo bộ test đã xem.

## 6. Tái lập và demo

Mã nguồn ở `starter/src/`; toàn bộ setup/run, kiểm thử và lệnh truy xuất thử ở RUNNING.md. Corpus đã có snapshot, manifest, giấy phép và benchmark. Trace đủ **45 dev + 15 test**, mỗi trace top-3; kết quả và nhãn đều lưu để kiểm tra. Chi tiết môi trường và hash nằm trong `results/environment.json`, `frozen_inputs.json`, `scoring_amendment.json`; cấu hình ở `artifacts/strategies.json`, nhật ký ở `artifacts/experiment_log.csv`.

Web demo chạy bằng `python -m starter.src.demo`, mở `http://127.0.0.1:8000`. Tab truy vấn hiển thị input embedding thật, vector mẫu, metadata filter và top-3 trực tiếp từ pgvector. Tab kết quả hiển thị metrics, trace/gold, thống kê ingest và audit từ file đã lưu. Web không gọi Groq, không sinh câu trả lời LLM, không ghi đè kết quả. Đây là phần trình bày bổ sung, không phải hybrid retrieval.

Thông tin họ tên, MSSV, lớp và việc xem xét audit đã được học viên xác nhận. Link repo đã điền trong `STUDENT.md`; commit chốt là HEAD nhánh main tại thời điểm nộp. Việc push, kiểm tra quyền truy cập của người chấm và nộp link theo kênh lớp cần được thực hiện ở bước nộp bài. Không commit `.env`, model cache hay dữ liệu nhạy cảm.
