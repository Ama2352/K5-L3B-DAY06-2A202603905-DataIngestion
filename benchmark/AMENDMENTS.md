# Hiệu chỉnh gold sau lần chạy đầu

Lượt review nguồn có AI hỗ trợ phát hiện q12 có thêm bằng chứng hợp lệ ở `errors`, đơn vị 5: câu nói file luôn đóng sau khối `with`, kể cả khi có lỗi xử lý. Gold ban đầu chỉ ghi `inputoutput`, đơn vị 4.

Đã bổ sung `alternative_gold` cho q12 và **chỉ chấm lại top-3 đã lưu**. Không đổi câu hỏi, split, query, filter, corpus, model, chunk, cấu hình hay thứ tự kết quả. Áp dụng cùng gold mới cho cả A/B/C. Cấu hình B đã chọn trên dev được giữ nguyên.

- Gold nguyên bản: `questions_v1.json`.
- Kết quả và trace trước hiệu chỉnh: `../results/original_scoring/`.
- Nhật ký, thời điểm, hash trước/sau và bảng chỉ số: `../results/scoring_amendment.json`.
- Sau hiệu chỉnh, A test Evidence Hit@3 đổi từ 0/5 thành 1/5; các chỉ số khác giữ nguyên.
- `results/frozen_inputs.json` hiện ghi hash của **lần chạy lại ngày 14/11/2026 lúc 04:29:20 UTC**, dùng gold đã hiệu chỉnh. Hash gold trước/sau lần hiệu chỉnh đầu vẫn ở `results/scoring_amendment.json`; gold cũ là `questions_v1.json`. `results/original_scoring/` giữ metrics và traces trước hiệu chỉnh, không đại diện cho lần chạy lại.

Học viên đã chạy lại `experiment --allow-rerun` từ 04:29:20 đến 04:30:51 UTC ngày 14/11/2026. Corpus, model, cấu hình, thứ hạng/text/metadata của hits và điểm sau hiệu chỉnh không đổi; thời gian thực thi được đo lại. B vẫn được chọn trên dev. Các file kết quả chính là lần chạy lại này; đây là tái lập trên bộ test đã xem, không phải holdout độc lập mới.

Lần chạy test đã hoàn tất trước khi chốt hiệu chỉnh. Đây là sửa thiếu sót annotation, không phải một lần đánh giá holdout độc lập mới; không mô tả kết quả đã hiệu chỉnh là test hoàn toàn chưa bị tác động. Muốn đánh giá độc lập nghiêm ngặt tiếp theo, cần soạn và chốt một tập test mới trước khi chạy. Không âm thầm xóa lịch sử để tăng điểm.
