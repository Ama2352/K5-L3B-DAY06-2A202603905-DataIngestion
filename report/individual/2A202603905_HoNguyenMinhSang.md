# Báo cáo cá nhân — Day 06: corpus PDF T5

- Họ tên: **Hồ Nguyễn Minh Sang**.
- MSSV: **2A202603905**; lớp: **L3B**.
- Chủ đề: truy xuất bằng chứng về dữ liệu, kiến trúc và thực nghiệm transfer learning cho NLP từ paper T5.
- Công cụ: Codex hỗ trợ triển khai, soạn câu hỏi, kiểm chứng nguồn, chạy thí nghiệm và đọc/chấm audit. Học viên đã kiểm tra kết quả thực nghiệm và xem xét audit của corpus PDF T5, xác nhận các nhãn hiện có. Nhãn ban đầu có AI hỗ trợ, kèm lý do và phương pháp.

## 1. Dữ liệu và pipeline

Nguồn duy nhất: Colin Raffel và cộng sự (2020), **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**, JMLR 21(140):1–67. Nguồn chính thức: https://www.jmlr.org/papers/v21/20-074.html. Snapshot `data/raw/t5_jmlr_2020.pdf` giữ nguyên PDF gốc; trang 1 ghi **CC BY 4.0**, cho phép chia sẻ/chuyển đổi khi ghi công, dẫn giấy phép và nêu thay đổi. Ghi công đủ tác giả, URL và cách xử lý ở `data/ATTRIBUTION.md`; quyền được kiểm tra cho chính bản JMLR này.

Chọn **trang vật lý 4–40, tổng 37 trang**, thuộc Setup/Experiments. Có nội dung hữu ích trên từng trang, gồm văn bản, bảng/hình và chú thích liên quan; không đếm bìa/mục lục/trang trắng/trang chỉ có tham khảo. Trang 1–3 và 41–67 ngoài phạm vi được liệt kê trong manifest; tham khảo bắt đầu ở trang 58. `page` là số trang thật, không đổi thành đơn vị HTML hay chia nhỏ để tăng số trang.

Pypdf trích layout một cột; chuẩn hóa Unicode NFC, mở rộng ligature, bỏ running header/số trang, nối dòng văn xuôi, giữ đoạn và bullet. Không OCR hoặc tự sinh nội dung. Kết quả **118.237 ký tự**, các trang có từ **1.991 đến 3.860 ký tự**. Manifest ghi hash PDF, hash text từng trang, phiên bản, ngày tải, trang chọn/loại và provenance của section. Khi validate, pipeline tái trích xuất PDF và so khớp text/metadata, không chỉ tin file JSON.

Giới hạn trích xuất: bảng/công thức được chuyển thành văn bản gần đúng; số mũ có thể mất định dạng, một số từ bị dính do kerning, line-end hyphen được giữ. Benchmark ưu tiên bằng chứng văn xuôi; PDF là bản chuẩn để đối chiếu. Không có trang scan/rỗng bị âm thầm bỏ qua; lỗi đọc trang gây dừng rõ ràng. `results/errors.json` là log lỗi kết nối setup thật từ lịch sử backend trước khi chuyển PDF; lần PDF này không gặp lỗi trích xuất.

Pipeline: PDF → trang chuẩn hóa → chunk A/B/C → embedding local → PostgreSQL/pgvector → top-3 → evidence metrics → audit và báo cáo. T5 là tài liệu nguồn, **không phải model embedding đang dùng**.

### Model, backend và metadata

- Embedding: **BAAI/bge-small-en-v1.5**, 384 chiều, FastEmbed/ONNX CPU local, 4 threads; query có prefix `Represent this sentence for searching relevant passages: `, document không prefix. Model quantized và tokenizer được hash trong `results/environment.json`.
- Store: PostgreSQL **16.16**, pgvector **0.8.7**, table `lab6_chunks`, volume Docker `day6_student_starter_pg_data`; cosine distance `<=>`, **exact search**, không approximate index. Runtime/library versions trong environment và requirements lock.
- Chi phí API **0 USD**, không gọi Groq/API; không quy đổi điện và phần cứng thành chi phí. Model cache/venv/.env không nộp.
- Metadata: `document_id`, `source` (PDF URL), `page` (int vật lý), `start/end` (offset trong text trang), `section` (array string), `language=en`, `year=2020`, `version=JMLR 21(140), 2020`, `document_type=research_paper`, `page_kind=pdf`.
- `section` lấy từ heading thật: Setup bắt đầu trang 4, Experiments bắt đầu trang 10. Trang 10 thuộc **cả hai**, không gán một nhãn giả. Filter input là một string được chuyển thành array containment trong JSONB; metadata đúng cho trang chuyển tiếp. Các trường hằng số như year/language giúp provenance nhưng không chứng minh lợi ích lọc trong corpus một tài liệu.

## 2. Cấu hình và benchmark

| Cấu hình | Chunking | Filter |
|---|---|---|
| A | Fixed 500 ký tự, overlap 50 | Không |
| B | Theo đoạn, tối đa 1.000 ký tự, không overlap; đoạn dài vẫn bị cắt | Không |
| C | Cùng chunk/vector của B | `section` khi câu hỏi cung cấp phạm vi |

Cùng corpus, model, metric, store, benchmark và query processing. A–B so sánh chính sách chunking, bao gồm size/overlap; không tách riêng từng tham số. B–C chỉ thay filter; C tái dùng vector B. Không fallback sang mock khi model/backend lỗi.

Benchmark mới gồm **20 câu tiếng Anh**, có câu trực tiếp, diễn đạt lại và điều kiện/ngoại lệ. Chốt **15 dev / 5 test** trước lần truy xuất PDF đầu tiên; test q04, q08, q12, q16, q20. Bốn câu filter: q06/q11 dev nêu Setup, q16/q20 test nêu Experiments. Filter lấy từ câu hỏi/ngữ cảnh trước truy xuất, không lấy đáp án hay gold page. Evidence được kiểm chứng nguyên văn trong text chuẩn hóa; vị trí quote trùng chính xác được tìm trước chạy để bổ sung alternative nếu có.

Thời điểm chốt và hash benchmark được ghi trong `artifacts/experiment_log.csv` trước chạy. Hash `questions.json` là `09317ff5009d1a9635d3d3c1dee52b2ee3a05cd5fbd2504d2355e858731adfac`. `frozen_inputs.json` ghi hash corpus/manifest/benchmark/config tại đầu chạy. Không sửa câu hỏi, gold hoặc cấu hình sau khi xem PDF test.

Quy tắc chọn: dev Evidence Hit@3 → dev Hit@1 → ít chunk hơn → tên cấu hình. Run bắt đầu **2026-11-14 05:28:34 UTC**, chọn **B lúc 05:29:33 UTC trước test**, kết thúc **05:29:34 UTC**. B/C hòa điểm và số chunk, B thắng tie-break theo tên. Không chọn lại theo test. Đây là corpus và câu hỏi mới; các thực nghiệm HTML cũ không được trộn vào số đo PDF và chỉ giữ trong lịch sử Git.

## 3. Kết quả và tài nguyên

Hit tự động cần đúng nguồn/trang và chứa **đủ quote trong một chunk**, không chỉ đúng chủ đề hoặc cosine cao. So khớp bỏ khác biệt case/khoảng trắng/Unicode. Không cộng hai chunk để thay định nghĩa Hit@3 chính. Grounding xét cả top-3 riêng.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Chunk count | Ingest (s) | API USD |
|---|---|---|---|---:|---:|---:|
| A | dev | 6/15 (40%) | 6/15 (40%) | 277 | 26,44 | 0 |
| A | test | 2/5 (40%) | 2/5 (40%) | 277 | — | 0 |
| B | dev | 11/15 (73,3%) | 12/15 (80%) | 160 | 28,58 | 0 |
| B | test | 3/5 (60%) | 5/5 (100%) | 160 | — | 0 |
| C | dev | 11/15 (73,3%) | 12/15 (80%) | 160 | 0,20* | 0 |
| C | test | 3/5 (60%) | 5/5 (100%) | 160 | — | 0 |

`*` C dùng lại embedding B, không đại diện cho ingest lạnh độc lập. Ingest gồm chunk/token statistics, embedding và lần ghi đầu; không gồm tải/khởi tạo model, nạp lại để kiểm chứng hoặc sự cố setup. Embedding A **26,05 s**, B **28,36 s**; ghi đầu lần lượt **0,33 / 0,17 / 0,14 s**. B ít chunk hơn nhưng đoạn dài hơn và embedding CPU vẫn chậm hơn A trong lần đo này; một lần đo không cho phép suy rộng tốc độ.

| Cấu hình | Ký tự min / median / max | Tổng ký tự chunk | Tổng token | Token max | Chunk >512 token | Bản ghi trước/sau nạp lại |
|---|---|---:|---:|---:|---:|---|
| A | 58 / 500 / 500 | 130.237 | 31.601 | 264 | 0 | 277 / 277 |
| B | 39 / 781,5 / 1.000 | 118.043 | 28.341 | 476 | 0 | 160 / 160 |
| C | 39 / 781,5 / 1.000 | 118.043 | 28.341 | 476 | 0 | 160 / 160 |

Token được đo bằng tokenizer thật, không có chunk vượt giới hạn 512 token. Overlap làm A có thêm 12.000 ký tự; B/C bỏ 194 ký tự phân cách đoạn giữa chunk. Nạp lại giữ số bản ghi đúng; client mới đọc lại từ database thật. Mỗi strategy thay toàn bộ snapshot trong transaction có advisory lock; khi đổi tài liệu, chunk cũ bị xóa và ghi chunk mới với ID theo nội dung/vị trí. Nhờ đó corpus HTML không còn lẫn trong A/B/C của database. Không đổi vector store hoặc cosine giữa cấu hình.

## 4. Audit: coherence, metadata, grounding

Mười vị trí coherence chốt trước: trang 4,8,12,16,20,24,28,32,36,40, offset giữa trang. Với nhiều chunk overlap bao phủ, chọn start nhỏ nhất. Đọc cùng vị trí cho A/B/C; mẫu trải đều phạm vi nhưng có thể rơi vào caption/công thức/bảng. Chấm 0 = không hiểu do mất ý/chủ thể; 1 = hiểu một phần/thiếu ngữ cảnh; 2 = tự hiểu, giữ trọn ý.

Metadata được đọc trên **5 test × top-3 × 3 = 45 bản ghi**; đối chiếu PDF URL, trang, section thật và text span. Grounding đọc **5 test cho mỗi cấu hình**, cả top-3: 0 = không có bằng chứng yêu cầu; 1 = thiếu chi tiết/điều kiện cần thiết; 2 = đủ để trả lời và truy nguồn. Không đánh giá câu trả lời LLM.

| Cấu hình | Coherence /2 (10 mẫu) | Metadata đủ/đúng | Câu filter có top-3 tuân thủ | Grounding 0 / 1 / 2 | Grounding /2 (5 câu) |
|---|---:|---|---|---|---:|
| A | 1,0 | 15/15 | 2/2 | 1 / 1 / 3 | 1,4 |
| B | 1,5 | 15/15 | 2/2 | 0 / 0 / 5 | 2,0 |
| C | 1,5 | 15/15 | 2/2 | 0 / 0 / 5 | 2,0 |

Phương pháp/người chấm: Codex hỗ trợ đọc text/context và gán nhãn ban đầu có lý do; metadata có thêm đối chiếu tự động nguồn/span. Học viên đã kiểm tra kết quả và xem xét audit PDF, xác nhận các nhãn hiện có mà không yêu cầu thay đổi điểm. `review_labels.json` giữ reviewer/phương pháp AI-assisted ban đầu và ghi `human_review_status=confirmed_by_student`, người xác nhận, căn cứ và thời điểm ghi nhận. B/C có cùng chunk mẫu và cùng test hits; tính giống nhau đã được kiểm chứng trước khi áp dụng cùng nhận xét. Đây không phải chấm mù hay đo đồng thuận độc lập; thiên lệch vẫn được công bố.

`source/page/start/end` giúp mở lại đoạn gốc, `section` bảo đảm phạm vi khi người dùng yêu cầu. Tuy nhiên cả B và C đã trả kết quả đúng scope ở các câu filter hiện tại, nên chưa thấy tăng Hit. A/B không thực thi filter dù top-3 hai câu filter test tình cờ đều thuộc Experiments; 2/2 không chứng minh A/B có cơ chế filter. Main section khá rộng, không xử lý nhầm lẫn giữa các ý trong cùng section.

## 5. Ví dụ lỗi, tác động và giới hạn

1. **A/q16, trang 29–30:** top-1 là bảng fine-tuning, top-3 mới có phần đầu mô tả adapter nhưng chưa tới câu nói layer normalization cũng được cập nhật. Hit@3=0, grounding=1. B/C giữ trọn đoạn trang 29 ở rank 2, đủ cả hai loại tham số; Hit@1=0 nhưng Hit@3=1.
2. **A/q20, trang 40/37:** top-3 nói về baseline-1T, scale và giao thức test, không có lý do English-only/backtranslation; grounding=0. B/C có đúng đoạn trang 40 ở rank 2, grounding=2. Cosine gần chủ đề chưa bảo đảm đúng chi tiết.
3. **A/q08, trang 12:** khoảng 5.000 bước và tiêu chí validation bị tách qua rank 1/rank 2. Hit@3 theo một-chunk là 0, nhưng đọc cả hai có đủ bằng chứng nên grounding=2. Không sửa gold để làm điểm đẹp hơn.
4. **B/C/q11 dev:** filter đúng Setup nhưng kết quả ở trang 9/10, thiếu đoạn quy tắc output classification không thuộc label tại trang 8. Lọc main section không sửa được lỗi phân biệt chi tiết trong cùng phạm vi.

Chọn **B** theo dev: tăng chất lượng evidence/coherence so với A, ít chunk hơn, nhưng thời gian embedding trong lần đo lớn hơn. C không tăng điểm so với B ở bộ câu hỏi này, chỉ bổ sung ràng buộc scope theo input người dùng. Không kết luận B là tốt nhất tổng quát từ 5 câu test.

Giới hạn: một paper tiếng Anh, dev/test có chủ đề chung; 5 test quá nhỏ; bảng/công thức và kerning PDF không được phục dựng hoàn hảo; các đoạn >1.000 ký tự vẫn có thể bị cắt giữa ý. q08 chưa nêu rõ baseline trong câu hỏi trong khi phần T5 cuối dùng checkpoint 1.000 bước thay vì 5.000; giữ nguyên benchmark sau test, ghi rõ điểm mơ hồ, cần thiết kế câu rõ hơn trong tập mới sau này. Exact quote checker có thể tạo false negative, chưa liệt kê mọi evidence tương đương về nghĩa. Mẫu coherence có caption, đánh giá do cùng AI hỗ trợ triển khai nên có thiên lệch. Chỉ một lần đo tốc độ, C tái dùng vector và query embedding được cache giữa cấu hình; không so trực tiếp elapsed tổng với search-only latency.

Hướng cải thiện: chốt test mới rõ scope trước chạy, mở rộng nguồn/câu hỏi, chấm đối chiếu bởi người khác, chunk theo sentence/heading với ngân sách token, đo lặp cold/warm. Hybrid là mở rộng tùy chọn chưa làm; không bổ sung bằng cách chỉnh theo test hiện tại.

## 6. Tái lập và nộp

Lệnh setup/run trong `README.md`, code ở `starter/src/`. Từ root: `python -m starter.src.corpus`, `python -m starter.src.run validate`, rồi `python -m starter.src.run experiment`. Bản đã chạy cần sao lưu kết quả và dùng `--allow-rerun` khi muốn tái lập. Truy xuất C bằng `--section "2. Setup"` hoặc `--section "3. Experiments"`. Web demo tạm bỏ theo yêu cầu.

Minh chứng: manifest/PDF/attribution, benchmark/questions, strategies/experiment log, 45 dev + 15 test traces đủ top-3, metrics/ingest stats, environment/frozen inputs/selection, 90 nhãn và audit summary, log setup thật. Không giữ file snapshot chunk/review inputs trùng lặp hoặc kết quả HTML cũ trong working tree.

Thông tin học viên và link repo ở `STUDENT.md`; commit chốt là HEAD nhánh main tại thời điểm nộp. Học viên đã xác nhận kiểm tra kết quả và audit PDF hiện tại; nguồn hỗ trợ AI được giữ rõ trong báo cáo và file nhãn. Bước push, bảo đảm người chấm truy cập được và nộp link theo kênh lớp là thao tác nộp bên ngoài repo. Không commit secret, .env, venv hay model cache.

## 7. Đối chiếu hồ sơ nộp theo SUBMISSION.md

| Yêu cầu | Minh chứng trong repo |
|---|---|
| Thông tin học viên, repo | `STUDENT.md` có họ tên, MSSV, lớp, link repo và cách lấy commit chốt |
| Corpus ≥30 trang, nguồn và quyền | PDF gốc 67 trang; manifest chọn 37 trang vật lý 4–40; `data/ATTRIBUTION.md` ghi CC BY 4.0 và thay đổi |
| Benchmark ≥20 câu, tách 15 dev/5 test | `benchmark/questions.json`, `benchmark/README.md`, hash và thời điểm chốt trước chạy trong experiment log |
| Embedding thật, store bền vững, metadata filter | `starter/src/`, cấu hình/requirements và `results/environment.json`; BGE 384 chiều, pgvector exact cosine, filter section |
| Ba cấu hình và so sánh có kiểm soát | `artifacts/strategies.json`, `artifacts/experiment_log.csv`; B–C chỉ thay filter |
| Top-3, dev/test metrics, thời gian và chi phí | Hai file trace (45 dev + 15 test), `metrics.csv`, `ingest_stats.csv`, `frozen_inputs.json`, `selection.json` |
| Audit và phân tích năm khía cạnh | 90 nhãn có lý do/phương pháp, học viên đã xem xét; `review_labels.json`, `audit_summary.csv` và mục 3–5 của báo cáo |
| Hướng dẫn tái lập, báo cáo và log | `README.md`, báo cáo cá nhân này, `results/errors.json` là log setup thật |
| Secret và quyền chia sẻ | `.env`, venv, model cache không đưa vào commit; nguồn PDF có giấy phép và attribution |
| Push, quyền truy cập người chấm, nộp link | Thực hiện theo kênh lớp bên ngoài repo; hồ sơ local không tự chứng minh đã nộp |
