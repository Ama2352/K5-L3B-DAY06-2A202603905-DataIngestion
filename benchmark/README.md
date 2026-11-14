# Benchmark cố định trước truy xuất

`questions.json` có 20 câu bằng tiếng Anh để khớp corpus và model tiếng Anh: 15 dev, 5 test. Test cố định: q04, q08, q12, q16, q20. Câu có filter: q06, q11, q16, q20; giá trị chương được nêu ngay trong câu hỏi, không suy ra từ đáp án sau truy xuất.

Mỗi câu có đáp án, nguồn, ID đơn vị và nguyên đoạn evidence đã đối chiếu trong `data/pages.json`. Bộ câu hỏi được AI hỗ trợ soạn và kiểm tra với nguồn. Tác giả cần đọc lại trước khi nộp.

Quy tắc chọn đã chốt: ưu tiên dev Evidence Hit@3, sau đó Hit@1, sau đó ít chunk hơn, cuối cùng thứ tự tên cấu hình. Lưu `results/selection.json` trước khi chạy test. Không chọn theo test và không dùng gold trong query/filter. Câu hỏi được chia theo ID để phủ nhiều chương; các chương xuất hiện ở cả dev và test nên còn nguy cơ tương đồng chủ đề. Tập 5 test nhỏ không cho phép suy rộng.

Hit cần đúng nguồn **và** đơn vị nguồn, đồng thời chứa đủ evidence trong một chunk. So khớp không phân biệt hoa/thường, chuẩn hóa Unicode và khoảng trắng. Đây là cách chấm tự động bảo thủ: bằng chứng tương đương chưa ghi trong gold có thể bị tính miss; xem trace để phân tích, không đổi gold sau khi xem test nhằm tăng điểm.

`coherence_locations.json` chọn trước 10 đơn vị đầu theo `(document_id, page)`, lấy vị trí giữa mỗi đơn vị. Với overlap, chọn chunk bao phủ có điểm bắt đầu nhỏ nhất. Mẫu thiên về chương Classes; phải ghi hạn chế này. Coherence và grounding dùng thang 0–2 theo GUIDE, không lấy Hit@3 thay nhãn đánh giá định tính.

Chạy `python -m starter.src.run validate` từ root để xác minh nguồn, hash, evidence và split mà không cần database. Không sửa corpus/config/benchmark giữa các chiến lược.
