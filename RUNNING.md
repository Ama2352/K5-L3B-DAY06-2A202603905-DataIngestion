# Hướng dẫn chạy bài làm Day 06

## Bài triển khai trong repo này

**Stack:** Python + PostgreSQL 16/pgvector theo starter; embedding thật chạy local bằng FastEmbed/ONNX với `BAAI/bge-small-en-v1.5` (384 chiều). Không cần API key. Groq không được dùng trong thí nghiệm bắt buộc vì bài này đánh giá truy xuất evidence, không yêu cầu chatbot.

**Corpus:** 6 chương Python 3.13 Tutorial, 42 đơn vị nguồn HTML hợp lệ, 155.072 ký tự. Mỗi đơn vị ít nhất 1.500 ký tự; `page` là ID đơn vị trong chương. Nguồn, ranh giới, hash và quyền sử dụng nằm ở [manifest](data/manifest.json), [ghi công](data/ATTRIBUTION.md) và [giấy phép](data/LICENSE_PYTHON.txt). Có sẵn snapshot nên không cần tải lại corpus.

### Setup và chạy trên Windows PowerShell

Chạy từ **root repo**, cần Python 3.11–3.14 và Docker Desktop đang chạy Linux containers:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r starter\requirements.lock.txt
if (!(Test-Path starter\.env)) { Copy-Item starter\.env.example starter\.env }
docker compose -f starter/docker-compose.yml -p day6_student_starter up -d --wait
.\.venv\Scripts\python.exe starter\scripts\check_setup.py
.\.venv\Scripts\python.exe -m starter.src.run validate
.\.venv\Scripts\python.exe -m starter.src.run experiment
```

Lần dùng đầu FastEmbed tải model ONNX khoảng 67 MB và tokenizer vào `.cache/models/`; sau đó chạy local. `.env` chỉ chứa cấu hình database local, không commit. `requirements.lock.txt` cố định phiên bản trực tiếp và runtime đã dùng; môi trường/thông tin hash model thực tế được ghi trong `results/environment.json`. Điện và phần cứng không được quy đổi thành chi phí API.

Từ thư mục `starter/`, lệnh tương đương là `..\.venv\Scripts\python.exe -m src.run`. Với Linux/macOS, dùng `.venv/bin/python` từ root và cú pháp copy/biến môi trường tương ứng.

### Thí nghiệm và kết quả

| Cấu hình | Chunking | Metadata filter |
|---|---|---|
| A | Fixed 500 ký tự, overlap 50 | Không |
| B | Gom theo đoạn, tối đa 1.000 ký tự | Không |
| C | Giữ nguyên chunk và vector của B | Lọc `chapter` nếu câu hỏi cung cấp |

Đo cosine **exact** trong PostgreSQL; không dùng approximate vector index. Các cấu hình giữ nguyên corpus/model/query/benchmark. B–C chỉ đổi filter; A–B đổi chính sách chunking (bao gồm kích thước và overlap), vì vậy không tách riêng tác động của từng tham số chunking. BGE dùng query prefix khuyến nghị, document không prefix; ONNX quantized do FastEmbed cung cấp, giới hạn 512 token. Số chunk vượt giới hạn được ghi trong ingest stats.

Benchmark có **15 dev / 5 test**, 4 câu filter (2 test), evidence đã kiểm chứng. Quy tắc chọn cấu hình: dev Hit@3 → dev Hit@1 → ít chunk hơn → thứ tự tên. Pipeline ghi `selection.json` trước khi đọc kết quả test. [Phương pháp benchmark](benchmark/README.md).

Sau lần chạy đầu có bổ sung nguồn gold hợp lệ cho q12 và chỉ chấm lại trace đã lưu. Kết quả gốc được giữ; đây không phải một holdout độc lập mới. Xem [nhật ký hiệu chỉnh](benchmark/AMENDMENTS.md).

- `results/metrics.csv`: Hit@1/Hit@3 theo split, có tử/mẫu.
- `results/traces_dev.json`, `traces_test.json`: toàn bộ top-3, text, score, metadata, filter và gold.
- `results/ingest_stats.csv`: số chunk, độ dài, token, thời gian, chi phí và kiểm tra nạp lại.
- `results/frozen_inputs.json`, `environment.json`: cấu hình, hash, phiên bản, model và thời điểm.
- `results/review_labels.json`, `audit_summary.csv`: 90 mẫu coherence, metadata, grounding kèm ngữ cảnh, nhãn, lý do và bảng tổng hợp.
- `results/selection.json`: lựa chọn cấu hình trên dev; `results/scoring_amendment.json` và `results/original_scoring/` lưu lịch sử hiệu chỉnh q12 cùng điểm/trace trước hiệu chỉnh.
- `artifacts/experiment_log.csv`: giả thuyết và kết quả ba cấu hình.
- `report/individual/`: báo cáo phân tích, lỗi và giới hạn.

Các nhãn do Codex hỗ trợ đọc/chấm được ghi rõ là **AI-assisted**. Học viên đã xác nhận xem xét audit; căn cứ xác nhận được ghi trong `review_metadata` của `results/review_labels.json`. Đây không phải chấm mù độc lập. Hit@3 không thay cho grounding/coherence và không phải độ chính xác câu trả lời LLM.

Lệnh `experiment` từ chối ghi đè khi test đã được đánh giá. Muốn chạy lại có chủ đích, sao lưu `results/` trước và dùng `--allow-rerun`; kết quả chạy lại không được mô tả là một holdout độc lập mới. Nếu chỉnh corpus/config/model sau khi đọc test, cần bộ test mới để có đánh giá độc lập.

Đầu ra audit được lưu gọn trong `review_labels.json`; không xuất thêm bản chunk snapshot và ba file review inputs trùng lặp. Khi chạy lại, chỉ giữ nhãn cho mẫu có ngữ cảnh nguồn/kết quả truy xuất không đổi. Mẫu thay đổi cần chấm lại và trạng thái review trở về `pending`; bảng tổng hợp audit cũ bị bỏ nếu còn nhãn chưa hoàn thành. Vì vậy luôn sao lưu trước khi chạy lại có thay đổi dữ liệu/cấu hình.

### Truy xuất thử và kiểm thử

#### Web demo local

Sau khi database có kết quả experiment, từ root repo và venv đã bật:

```powershell
python -m starter.src.demo
```

Mở **http://127.0.0.1:8000**. Có thể đổi cổng bằng `--port 8001`; dừng bằng Ctrl+C. Web dùng thư viện HTTP có sẵn của Python, không cần cài frontend hoặc thêm dependency. Demo đọc pgvector trực tiếp, chỉ embedding câu hỏi; không ingest lại hoặc ghi đè artifact trong `results/`. Nếu model/database lỗi, web báo lỗi rõ ràng.

Nhập câu hỏi, chọn A/B/C hoặc so sánh cả ba. Chương là input do người dùng chọn trước; chỉ C áp dụng filter, A/B luôn tìm toàn corpus. Mỗi cấu hình hiển thị top-3, cosine score, text, URL và metadata. `page` là đơn vị nguồn HTML. Cosine không phải xác suất đúng; demo không tự chấm câu hỏi mới bằng benchmark gold.

Sau truy vấn, mục **Luồng retrieval** hiển thị input thật có query prefix, model, số chiều, 8 giá trị vector đầu, chuẩn L2, thời gian embedding và filter/thời gian tìm vector của từng cấu hình. Các số đo được trả sau khi truy vấn hoàn tất, không phải mô phỏng tiến trình thời gian thực. Lần đầu có thêm thời gian nạp model và kiểm tra hash; tổng thời gian có thể gồm chờ khóa model.

Tab **Kết quả experiment đã lưu** đọc các file có sẵn trong `results/`: metrics thành bảng dev/test; ingest stats thành bảng tài nguyên; trace chọn theo câu hỏi để so sánh A/B/C với gold và thứ hạng hit; audit có nhãn, lý do, ngữ cảnh và phương pháp. Có thể xem selection, môi trường, frozen inputs và lịch sử hiệu chỉnh q12. Nhãn `pending` được giữ đúng như file, không tự xác nhận thay học viên. Các file `original_scoring/` được ghi rõ là trước hiệu chỉnh. Nút **Đọc lại file** tải lại nội dung từ đĩa; lỗi hoặc file thiếu được báo rõ, không điền số liệu giả.

Web chỉ thực hiện semantic retrieval bằng embedding local và pgvector; không gọi API sinh câu trả lời và không cần API key. Đây là phần trình bày bổ sung, không phải hybrid retrieval và không thay nhãn grounding/metrics của lab. Server chỉ bind localhost, dành cho demo trên máy cá nhân.

#### CLI và kiểm thử

```powershell
.\.venv\Scripts\python.exe -m starter.src.run query --strategy C --chapter errors --question "Does finally run without an exception?"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Kiểm thử tích hợp thật (sau setup, trong PowerShell):

```powershell
$env:RUN_LOCAL_MODEL = "1"
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -c "import os,pytest; from dotenv import load_dotenv; load_dotenv('starter/.env'); os.environ['TEST_DATABASE_URL']=os.environ['DATABASE_URL']; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider']))"
```

Tái tạo corpus từ snapshot: `python -m starter.src.corpus`. Chỉ dùng thêm `--download` khi muốn cập nhật nguồn có chủ đích; cập nhật sẽ thay hash và có thể làm gold cũ không còn hợp lệ. Tổng hợp lại các nhãn đã chấm: `python -m starter.src.audit`.

Docker Compose dùng cổng `127.0.0.1:55433` và volume riêng `day6_student_starter_pg_data`. Dừng mà giữ dữ liệu: `docker compose -f starter/docker-compose.yml -p day6_student_starter stop`. Không xóa volume để xử lý lỗi kết nối. Khi Docker engine lỗi 500, kiểm tra Docker Desktop/WSL. `results/errors.json` giữ log lỗi thật đã được loại thông tin kết nối nhạy cảm, theo yêu cầu ghi lỗi của GUIDE; không tự tạo lỗi hoặc log để bổ sung bài nộp.
