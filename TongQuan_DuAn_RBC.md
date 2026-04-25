# 🔬 Báo Cáo Quy Trình Kỹ Thuật & Cấu Trúc Dự Án Hệ Thống Chẩn Đoán Hồng Cầu (RBC Analysis)

Dưới đây là tài liệu chi tiết từ A đến Z, mô tả toàn bộ kiến trúc, phương pháp luận, thuật toán và cấu hình tham số được sử dụng trong 4 file Jupyter Notebook và ứng dụng Web `app.py`.

---

## 📗 1. Tệp `01_Segmentation_Comparison.ipynb` (Giải Bài Toán Phân Đoạn Tế Bào)

**🎯 Mục đích:**  
Xây dựng và so sánh 3 kiến trúc mô hình học sâu (Deep Learning) để thực hiện tác vụ bóc tách (segmentation) tế bào hồng cầu ra khỏi vùng nền (background) của tiêu bản máu. 

**📊 Dữ liệu & Tiền xử lý (Preprocessing):**
- Kích thước ảnh đầu vào (Image Size): Đưa về chuẩn **256 x 256 pixel**.
- Tham số tải dữ liệu (Batch Size): Nhóm **32 ảnh** cho mỗi lần huấn luyện để tối ưu VRAM.
- Tăng cường dữ liệu (Data Augmentation): Lật ngang, lật dọc, xoay ngẫu nhiên giúp mô hình nhận diện tế bào ở mọi góc độ.

**🧠 Ba Mô Hình Thử Nghiệm:**
1. **CNN (Mini U-Net):**  
   - Cấu trúc: Hình chữ U gồm bộ nén (Encoder) dùng Tích chập (Conv2d) + Rút trích (MaxPooling) và bộ giải mã (Decoder) dùng UpSampling. Đặc biệt sử dụng Skip-Connection để ghép nối thông tin vị trí từ Encoder sang Decoder giúp giữ được các cạnh biên sắc nét của tế bào.
2. **RNN (ConvLSTM U-Net):**  
   - Cấu trúc: Thay thế một phần trung tâm của U-Net bằng khối **C-LSTM** (Convolutional Long Short-Term Memory). Tính chất "có trí nhớ" của LSTM kết hợp không gian của Tích chập giúp mô hình hiểu được chuỗi tiếp nối của các mảng ảnh liên kề nhau.
3. **Transformer (ViT-UNet):**  
   - Cấu trúc: Kết hợp U-Net với lõi **Vision Transformer**. Ảnh được băm thành các miếng nhỏ (patches), sau đó chiếu (Projection) vào cơ chế Self-Attention đa đầu. Sức mạnh này giúp mô hình nhìn thấy "bức tranh toàn cảnh", hiểu được một tế bào méo ở góc trái có nét giống với một tế bào méo ở góc phải.

**📐 Hàm Đánh Giá & Tối Ưu:**
- **Hàm mất mát (Loss Function):** Kết hợp hai thước đo là **BCE Loss** (Binary Cross Entropy - tốt cho phân loại nhị phân pixel) và **Dice Loss** (cực kỳ nhạy với các vùng vật thể siêu nhỏ). 
- **Chỉ số đo lường (Metrics):** IoU (Intersection over Union - mức độ chồng lấp) và F1-Score.

---

## 📘 2. Tệp `02_Classification_Comparison.ipynb` (Phân Loại 9 Dạng Tế Bào - Mô Hình Cơ Sở)

**🎯 Mục đích:**  
Huấn luyện các mô hình AI nhỏ gọn nhưng hiệu suất cao để phân loại những hình dạng bình thường/bất thường của Hồng cầu (ví dụ: Tròn, Bờ răng cưa, Giọt nước, Chồng lấp, v.v.).

**📊 Phân bổ Dữ liệu Tuyệt Đối Chống Bias:**
- Tập dữ liệu hơn 200.000 tấm ảnh cắt nhỏ (crop) về quy chuẩn **80 x 80 pixel**.
- **Chia tập dữ liệu (Stratified Split 70/15/15):** Kỹ thuật này cam kết tỷ lệ 70% Train, 15% Val và 15% Test luôn giữ được tỷ lệ phân bổ của các nhóm bệnh hiếm gặp. (Fix lỗi các bệnh hiếm không có hình trong tập Validation).
- **Trọng số Cân bằng (Weighted Random Sampler):** Xử lý triệt để việc mô hình "bị mù" dạng bệnh lý khác và lúc nào cũng chỉ trả về "Overlapping". Mô hình sẽ luôn được nạp các thẻ ảnh lớp thiếu một cách dày đặc tương đương với thẻ ảnh có nhiều.

**🧠 Ba Mô Hình Thử Nghiệm:**
1. **Custom CNN với Cơ Chế Tập Trung (CBAM):**  
   - Thiết kế tích chập tự chế + Bơm thêm Module CBAM. Mạng này tự mô phỏng con mắt con người: "Nhìn vào đâu trên tế bào" (Spatial Attention) và "Góc nhìn nào đáng giá nhất" (Channel Attention) -> Cho ra độ chính xác vượt trội nhất ~90%.
2. **RNN (Bi-directional GRU Classifier):**  
   - Băm ảnh thành chuỗi dãy vector 1 chiều và đọc từ trái sang phải, phải sang trái bằng Cổng vòng lặp 2 chiều (Bi-directional).
3. **Custom Vision Transformer (ViT):**  
   - Mô phỏng Transformer cho ảnh hẹp. Nhược điểm của nó là đòi hỏi cực lớn lượng Data và dễ bị OOM với ảnh kích thước nhỏ hơn 16x16 patch, cho val_acc đuối hơn CNN.

**⚙️ Chỉ Số Huấn Luyện:**
- Tốc độ học (Learning Rate): $1e-3$
- Bộ Cắt Giảm Tốc Độ (ReduceLROnPlateau): Nếu sau 3 Epochs máy không thông minh lên, giảm tốc độ học đi x0.5.
- Kỹ thuật Label Smoothing: Phạt 10% sự tự tin của mô hình -> Ngăn việc mô hình đoán sai nhưng luôn bảo thủ với tỷ lệ 99.9%. Tránh việc học vẹt.
- Chỉ Lưu Đỉnh Cao (`*_best.pt`): Cho phép mô hình chạy mệt mỏi nhưng hệ thống chỉ lưu khoảnh khắc hệ thống trả về Validation Accuracy cao nhất dọc chặng đường.

---

## 📙 3. Tệp `04_SwinTransformer_Classification.ipynb` (Phân Loại Bằng Transformer Tiên Tiến)

**🎯 Mục đích:**  
Khai thác phiên bản cải tiến, vĩ đại hơn của Vision Transformer là **Swin Transformer** (Shifted Windows Transformer) để khắc phục nhược điểm học cục bộ kém.

**🧠 Cấu Trúc Đột Phá:**
- **Patch Partition & Merging:** Không chia ảnh thành miếng chết cố định, mà chia nhỏ dần (phân cấp kiến trúc kim tự tháp). Rất hiệu quả việc tóm gọi các cạnh và hình bao (dấu hiệu bệnh máu) với độ phân giải linh động.
- **W-MSA (Window Multi-Head Self-Attention):** Tính tương quan Attention ngay trong Cửa sô hẹp để hạn chế độ lố cho RAM.
- **SW-MSA (Shifted Window):** Bước tiếp theo, mạng sẽ lệch khung trượt (Shifted) bằng 1/2 kích thước cửa sổ cũ, để móc nối góc nhìn cửa sổ trái sang cửa sổ phải, triệt tiêu viền giới hạn của Attention cơ bản.
- Mô hình chạy cùng một Config về Stratified và Sampler nghiêm ngặt (chuẩn y tế) giống File số 02.

---

## 📕 4. Tệp `03_EndToEnd_Comparison.ipynb` (Đánh Giá Hệ Sinh Thái Toàn Diện)

**🎯 Mục đích:**  
Mô phỏng lại toàn bộ lộ trình dòng đời của bức hình chuẩn bị trên kính hiển vi (Pipelines Full Flow). 
Bức hình lớn -> Segmentation -> Contour (Bảo vệ đường viền cắt ô vuông) -> Resize 80x80 -> Classification 9 Lớp.

**🧪 Phương pháp Đánh Giá:**
- Vì mô hình bóc tách đôi khi cắt lệch hoặc lấy quá nhiều nhiễu đen, hệ thống End-to-End Test đánh giá độ rớt hiệu năng khi Classifier phải phán đoán hình cắt từ máy (thay vì hình cắt đẹp bằng tay). 
- Ánh xạ ma trận Confusion đối chiếu giữa Nhãn thực tế và Nhãn AI sinh ra dọc hệ thống liên kết nội bộ. Chứng minh tính khả thi của hệ thống trong hoàn cảnh thực tế.

---

## 💻 5. Hệ Thống Dashboard Web Lâm Sàng (`app.py`)

**🎯 Mục đích:**  
Dựng Front-end (Giao diện) chạy bằng framework `Streamlit` để đưa toàn bộ khối kiến thức hàn lâm phía trên thành phần mềm Trợ lực Y Tế Siêu Tốc sử dụng cho bệnh viện và phòng thí nghiệm.

**🎨 Thiết Kế Kỹ Thuật Giao Diện (UI/UX Engineering):**
- Sử dụng Custom CSS (Glassmorphism hiệu ứng kính trong, nền Dark Mode tối ưu mỏi mắt của bác sĩ). Hệ thống báo cáo được thiết kế viền nổi bật (Premium Aesthetic).

**⚙️ Hai Chế Độ Mạch Lạc, Phân Ly Chức Năng:**
1. **Chế Độ Single Cell (Phân Lớp Độc Lập):** 
   - Đưa trực tiếp ảnh vi phẫu 1 tế bào duy nhất vào luồng phân loại (Skipping Segmentation). Trả liền về **Top 3% Phán đoán** với độ tin cậy được rọi sáng bằng thanh trạng thái. Điều này xử lý tận gốc Lỗi "Sai râu cắm cằm bôi bẩn" của phiên bản cũ.

2. **Chế Độ Full Slide (Rà Soát Mảng Rộng Tiêu Bản):**
   - **Tuyến 1**: Băm hình vô mạng UNet chạy Forward rọi bóng hình bóng viền. Nâng Threshold để xóa Noise.
   - **Tuyến 2 (Crop Inspector):** Nảy OpenCV hàm `findContours`, hệ thống thuật toán tự loại bỏ các hộp đánh dấu dị dạng như Dài, Loằng ngoằng (H/W Aspect Ratio vô lý) hay quá tí hon (< 100 Pixel) do bụi. Mở ra một bảng Grid Show (Thanh tra phân mảnh) in trọn từng mặt các khối máu bị bắt giữ.
   - **Tuyến 3**: Các mảnh chắt lọc này nối đuôi chui qua mạng Classification (CNN/RNN/Swin). 

**🔬 Công Nghệ Trí Tuệ Lâm Sàng Cốt Lõi (AI Clinical Insights):**
Hệ thống không chỉ trả số liệu, mà biến số liệu thành lời khuyên chẩn đoán như ở bệnh viện sinh học dựa vào Mode toán học (Đếm phần tử cực trị của chuỗi).
- Quét qua hàng loạt các hồng cầu trên diện rộng. Tính tổng.
- Nếu Cực trị xuất hiện Overlapping => "Bản mạch hiện nhiều sự gắn kết xếp chồng, cẩn trọng kết tủa vi mô máu". 
- Nếu Cực trị phát hiện Fragmented (Vỡ lở) => Cảnh báo khẩn! Báo hiệu tổn thương, vỡ màng nhầy do chấn thương mao mạch.

👉 Đây là một ứng dụng có tính bao đóng bảo mật, an toàn vòng nhớ và không bao giờ rò rỉ bộ nhớ (không chạy lưu state dư thừa), có khả năng chịu tải trên Localhost ổn định và giao diện ấn tượng ngay từ ánh nhìn đầu tiên.
