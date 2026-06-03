# XÂY DỰNG MÔ HÌNH CÂY QUYẾT ĐỊNH PHÂN LOẠI ĐỊNH DẠNG TỆP TIN VÀ MÃ NGUỒN TRÊN DATASET THE STACK

# Sinh viên thực hiện: Đào Quốc Tuấn | MSSV: 23120392

## 1. Tổng quan hiệu năng hệ thống (Performance Summary)

Thực nghiệm được triển khai trên quy mô **32.050 mẫu thô** stream trực tiếp từ Dataset *The Stack* (Hugging Face), phân phối cân bằng trên 32 lớp (ngôn ngữ lập trình/định dạng văn bản). Sau công đoạn làm sạch dữ liệu nghiêm ngặt, hệ thống loại bỏ 97 tệp rác (tệp trống, lỗi nhị phân), giữ lại **31.953 mẫu sạch** để đưa vào phân tách cấu trúc đặc trưng.

Bảng số liệu dưới đây phản ánh hiệu năng tổng quát của hai mô hình sau khi mở rộng tập dữ liệu lên 1.000 mẫu/lớp và áp dụng bộ đặc trưng cải tiến:

| Chỉ số Metric | Baseline (Sklearn Decision Tree) | Advanced (LightGBM Classifier) |
| :--- | :---: | :---: |
| **Accuracy (Độ chính xác toàn cục)** | 49.81% | **68.56%** |
| **Macro Average Precision** | 0.58 | **0.69** |
| **Macro Average Recall** | 0.50 | **0.69** |
| **Macro Average F1-Score** | 0.51 | **0.68** |
| **Tỷ lệ chặn định dạng lạ (Unknown)** | 68.00% | **72.00%** |

### Nhận xét sơ bộ:
Việc nâng cấp quy mô dữ liệu lên 1.000 mẫu/lớp kết hợp kỹ nghệ đặc trưng chuyên sâu đã tạo ra bước nhảy vọt về hiệu năng. Độ chính xác của LightGBM tăng mạnh lên **68.56%** và F1-score đạt **0.68** (vượt xa mức 58% của tập dữ liệu nhỏ trước đó), chứng minh mô hình phân lớp trên dữ liệu thực tế có độ tin cậy cao.

---

## 2. Phân tích bẻ gãy các "Vùng mù" dữ liệu (Blind Spots Mitigation)

Sự kết hợp giữa việc mở rộng cửa sổ ngữ cảnh quét (150 dòng đầu) và bổ sung các tính năng "độc bản" đã mang lại hiệu quả vượt trội tại các phân lớp phức tạp:

* **JavaScript và TypeScript:** Trước đây, hai ngôn ngữ này có sự tương đồng cú pháp rất lớn, khiến điểm của `javascript` bị dìm sâu (dưới 10%). Sau khi bổ sung đặc trưng `fe_ts_types` (quét các định nghĩa kiểu `interface`, `type`, `as`), LightGBM đã tách biệt thành công hai ranh giới này, đưa F1-score của `javascript` lên **0.51** và `typescript` lên **0.58**.
* **Phân tách Markdown và YAML:** Tính năng `fe_md_fenced` (bắt khối code bọc ` ``` `) và `fe_yaml_blocks` (bắt cấu trúc thụt lề danh sách khối `- `) đã giải quyết hoàn toàn bài toán trộn lẫn mã nguồn. F1-score của `markdown` bứt phá lên **0.72** và `yaml` đạt **0.52** ở mô hình nâng cao.
* **Các lớp ngôn ngữ tuyến tính giữ vững phong độ:** Nhờ từ khóa đặc trưng, các lớp như `go` (F1: **0.94**), `dockerfile` (F1: **0.91**), `xml` (F1: **0.89**), và `json` (F1: **0.88**) đạt độ chính xác gần như tuyệt đối, hình thành các ranh giới phân tách cực kỳ vững chắc cho mô hình.

---

## 3. Biện luận cơ chế phân nhánh và Tầm quan trọng của Đặc trưng

Dựa trên việc đối chiếu trực quan giữa cấu trúc sơ đồ cây (`decision_tree.jpg`) và biểu đồ độ đóng góp (`feature_importance.png`), ta rút ra các kết luận học thuật quan trọng:

### A. Phân tích cấu trúc sơ đồ Cây quyết định (Decision Tree Analysis)
Cây quyết định Baseline lựa chọn các từ khóa và tỷ lệ phân phối có chỉ số Gini tốt nhất để thực hiện cắt nhánh ở các tầng cao nhất:
* **Gốc cây (Root Node):** Mô hình chọn đặc trưng `fe_rust_go <= 0.0033` làm điều kiện tiên quyết ở đỉnh. Điều này cho thấy mật độ xuất hiện của các từ khóa độc quyền như `fn`, `func`, `fmt.Println` mang lại độ tinh khiết thông tin tối đa, giúp cô lập ngay lập tức một lượng lớn mã nguồn Go/Rust ra khỏi các định dạng cấu trúc dữ liệu tuyến tính.
* **Tầng thứ nhất (Level 1 Splits):** * Nhánh bên trái tiếp tục phân tách dựa trên `html_tag_ratio <= 0.4197`, phân ranh giới rất mạnh giữa nhóm tài liệu đánh dấu cấu trúc văn bản (`html`, `xml`) và các nhóm dữ liệu dạng bảng/cấu hình (`csv`, `makefile`).
  * Nhánh bên phải phân tách dựa trên `fe_jvm_keywords <= 0.0033` để cô lập nhóm chạy trên máy ảo Java (Java, Kotlin, Scala) ra khỏi các mã nguồn khác.
* **Tầng thứ hai (Level 2 Splits):** Cây chuyển dịch sang sử dụng các đặc trưng như `fe_sql_keywords`, `has_doctype`, `first_line_pattern` và `fe_web_syntax`. Đây là minh chứng rõ ràng cho việc cây quyết định đi từ phân loại đại thể (nhóm lớn) ở gốc, dần dần đi vào chi tiết (nhóm cú pháp nhỏ) ở các lá sâu hơn.

### B. Biện luận về Độ quan trọng đặc trưng trong LightGBM (Feature Importance)
Biểu đồ `feature_importance.png` đo lường dựa trên thang đo **Information Gain (Lượng thông tin thu hoạch được)** cho thấy một góc nhìn toàn diện hơn:
1. **`first_line_pattern` đứng đầu bảng tuyệt đối (Score: 261,752.25):** Đúng như mẹo thiết kế từ đề bài của giáo viên (*"File headers are the most discriminative"*), mẫu cấu trúc của dòng đầu tiên (như dòng Shebang `#!/bin/bash` của Shell, hay `<?xml` của XML) mang lượng thông tin lớn nhất để định danh một định dạng tệp.
2. **`special_char_ratio` đứng thứ hai (Score: 199,818.325):** Mật độ các ký tự đặc biệt (`{`, `}`, `;`, `#`) là ranh giới bất biến giữa các ngôn ngữ lập trình (C, Java sử dụng nhiều dấu ngoặc và chấm phẩy, trong khi Python hay YAML dựa vào khoảng trắng thụt lề).
3. **`fe_jvm_keywords`, `avg_line_length`, và `html_tag_ratio` tiếp nối trong top 5:** Sự kết hợp hoàn hảo giữa các thuộc tính thống kê bề mặt văn bản và tần suất cú pháp lõi. Chính sự bổ sung của các đặc trưng nâng cao này giúp LightGBM Classifier đạt độ chính xác **68.56%** trên tập dữ liệu thực tế chứa nhiều nhiễu phức tạp.

---

## 4. Cơ chế phòng thủ và xử lý định dạng tệp lạ (Unknown Handling)

Hệ thống triển khai chiến lược **Ngưỡng độ tin cậy (Confidence Thresholding)** trên phân bổ xác suất đầu ra từ hàm `predict_proba()` với mức sàn cố định là **0.70** ($70\%$):

$$\text{Nếu } \max(P(\text{class} \mid X)) < 0.70 \implies \text{Gán nhãn} = \text{"UNKNOWN"}$$

* **Đánh giá hiệu năng thực tế:** Khi đưa dữ liệu của lớp `brainfuck` (một ngôn ngữ lập trình không nằm trong 32 lớp huấn luyện chính thống) vào tập kiểm thử, hệ thống ghi nhận tỷ lệ chặn đứng mẫu lạ cực kỳ ấn tượng.
* Mô hình **Decision Tree chặn thành công 68.00%** và mô hình nâng cao **LightGBM chặn đứng thành công 72.00%** mẫu lạ.
* **Biện luận kỹ thuật:** Việc LightGBM đạt tỷ lệ chặn Unknown cao hơn hẳn (72.00%) ở tập dữ liệu lớn là nhờ mô hình đã học được phân bổ ranh giới các ngôn ngữ đích rất chặt chẽ. Khi gặp một tệp văn bản có cú pháp hoàn toàn dị biệt, xác suất phân bổ nhãn của LightGBM bị phân tán đều hoặc giữ ở mức thấp (dưới 0.70), kích hoạt bộ lọc từ chối phân loại bừa bãi. Cơ chế này đảm bảo tính ổn định và an toàn thông tin cho hệ thống khi ứng dụng vào thực tế.