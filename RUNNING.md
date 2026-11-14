# Chạy bài lab với corpus PDF T5

## Corpus và cấu hình

Nguồn duy nhất là paper **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**, Raffel và cộng sự, JMLR 21(140), 2020. PDF có 67 trang; bài dùng **trang vật lý 4–40, tổng 37 trang**, 118.237 ký tự đã chuẩn hóa. Đây là phần Setup và Experiments, có nội dung liên quan trên từng trang, không tính bìa/mục lục/trang trắng/tài liệu tham khảo. Manifest liệt kê trang chọn/loại, lý do, số ký tự, hash và thời điểm tải. `page` luôn là số trang thật, không đánh số lại từ 1.

PDF gốc ở `data/raw/t5_jmlr_2020.pdf`; giấy phép **CC BY 4.0** in trên trang 1. Xem [ghi công và thay đổi](data/ATTRIBUTION.md), [manifest](data/manifest.json) và [benchmark](benchmark/README.md). PDF chỉ là tài liệu nguồn; embedding vẫn dùng **BAAI/bge-small-en-v1.5**, 384 chiều, FastEmbed/ONNX CPU local. Không cần API key.

| Cấu hình | Chunking | Filter |
|---|---|---|
| A | Cố định 500 ký tự, overlap 50 | Không |
| B | Gom theo đoạn, tối đa 1.000 ký tự, không overlap | Không |
| C | Giữ nguyên chunk và vector của B | `section` nếu câu hỏi cung cấp phạm vi |

PostgreSQL 16/pgvector lưu vector bền vững, cosine exact search, không dùng approximate index. A–B thay chính sách chunking; B–C chỉ thêm filter. `section` là danh sách tiêu đề thật: `2. Setup` hoặc `3. Experiments`. Trang 10 thuộc cả hai; backend chuyển filter chuỗi thành JSONB array containment nên trang chuyển tiếp vẫn được tìm trong cả hai phạm vi.

## Chạy từ đầu

Từ root repo, Windows PowerShell, cần Docker Desktop chạy Linux containers:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r starter/requirements.lock.txt
if (!(Test-Path starter/.env)) { Copy-Item starter/.env.example starter/.env }
docker compose -f starter/docker-compose.yml -p day6_student_starter up -d --wait
python starter/scripts/check_setup.py
python -m starter.src.corpus
python -m starter.src.run validate
python -m starter.src.run experiment
```

Snapshot PDF đã có sẵn. `corpus` trích xuất và tạo lại `data/pages.json`, `data/manifest.json` bằng pypdf; không OCR. `validate` kiểm tra hash, tái trích xuất PDF để đối chiếu toàn bộ text/metadata, rồi kiểm chứng evidence, split và filter. Có lỗi trích xuất/nguồn thiếu thì báo lỗi rõ ràng.

Lệnh `experiment` từ chối ghi đè khi `selection.json` đã tồn tại. Muốn chủ động chạy lại, sao lưu `results/` ra ngoài repo rồi dùng:

```powershell
python -m starter.src.run experiment --allow-rerun
```

Chạy lại trên bộ test đã xem là tái lập, không phải test độc lập mới. Không đổi corpus/model/config/gold để cải thiện điểm sau khi đã đọc test. Khi cần cập nhật PDF có chủ đích: `python -m starter.src.corpus --download`; việc này cập nhật thời điểm tải và có thể đổi hash, nên phải chốt lại benchmark trước thử nghiệm mới.

Lần đầu FastEmbed tải khoảng 67 MB model vào `.cache/models/`; các lần sau dùng cache. Cùng model, prefix query, tokenizer, metric, corpus và benchmark được giữ giữa A/B/C. Query thêm prefix BGE khuyến nghị, document không thêm prefix. Chi phí API là 0 USD; điện/phần cứng không được quy đổi. Không commit `.env`, model cache hoặc venv.

## Đọc kết quả và audit

- `results/metrics.csv`: Hit@1/Hit@3, tử/mẫu, riêng dev/test.
- `results/traces_dev.json`, `traces_test.json`: 45 dev + 15 test trace, mỗi trace top-3 gồm text, cosine score, metadata, gold, hit ranks và filter thực tế.
- `results/ingest_stats.csv`: chunk count, min/median/max độ dài, tổng ký tự/token, thời gian, chi phí và số bản ghi trước/sau nạp lại.
- `results/environment.json`: phiên bản runtime/backend, hash model, prefix và số chiều.
- `results/frozen_inputs.json`: hash các input và quy tắc chọn chốt trước khi truy xuất.
- `results/selection.json`: chọn trên dev trước khi đánh giá test.
- `results/review_labels.json`, `audit_summary.csv`: 30 coherence + 45 metadata + 15 grounding, nhãn/lý do/phương pháp và bảng tổng hợp.
- `results/errors.json`: log setup thật còn giữ từ lịch sử backend trước khi chuyển corpus; không phải lỗi giả hoặc lỗi trích xuất PDF mới.
- `artifacts/experiment_log.csv`: giả thuyết và thời điểm chốt trước truy xuất, hash benchmark và số đo sau chạy.

Không xuất file chunk snapshot hoặc review input trùng lặp. Lần chạy lại chỉ tái sử dụng nhãn có ngữ cảnh không đổi; mẫu thay đổi cần chấm lại. Audit PDF là **AI-assisted**, không dùng xác nhận audit HTML trước đây để giả định học viên đã xác nhận nhãn PDF. Sau khi xem xét/sửa nhãn có chủ đích, tổng hợp bằng:

```powershell
python -m starter.src.audit
```

Coherence/grounding là đánh giá định tính riêng, không lấy Hit@3 thay thế. Grounding xét cả top-3; một hit tự động yêu cầu đủ evidence trong một chunk. Không có chatbot hoặc điểm độ chính xác câu trả lời sinh.

## Truy xuất thử và kiểm thử

```powershell
python -m starter.src.run query --strategy C --section "2. Setup" --question "How does the text-to-text framework specify a task?"
python -m pytest -q -p no:cacheprovider
```

Kiểm thử đầy đủ với model local và database thật, sau setup:

```powershell
$env:RUN_LOCAL_MODEL = "1"
$env:HF_HUB_OFFLINE = "1"
python -c "import os,pytest; from dotenv import load_dotenv; load_dotenv('starter/.env'); os.environ['TEST_DATABASE_URL']=os.environ['DATABASE_URL']; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider']))"
```

Database dùng volume `day6_student_starter_pg_data`, cổng `127.0.0.1:55433`. Nạp lại thay snapshot A/B/C trong transaction, loại chunk cũ của corpus HTML và không nhân đôi. Dừng mà giữ dữ liệu: `docker compose -f starter/docker-compose.yml -p day6_student_starter stop`. Không xóa volume để chữa lỗi kết nối.

Phần web demo tạm bỏ theo yêu cầu; dùng CLI và các file kết quả để trình bày. Pipeline và số liệu HTML cũ chỉ còn trong commit trước, không trộn với kết quả PDF.
