# Ứng dụng Phân loại Trái cây & Rau củ

Ứng dụng web đơn giản sử dụng mô hình Deep Learning để phân loại các loại trái cây và rau củ. Backend được xây dựng bằng Flask (Python) tích hợp mô hình ONNX, và Frontend là HTML/CSS/JavaScript cơ bản.

Dự án này được phát triển như một demo kỹ năng xây dựng ứng dụng hoàn chỉnh (full-stack) sử dụng mô hình học máy, phù hợp để đưa vào CV.

## Tính năng chính

* **Phân loại ảnh tải lên:** Người dùng có thể chọn một file ảnh từ máy tính của mình để tải lên và dự đoán.
* **Sử dụng ảnh mẫu:** Cung cấp một bộ sưu tập ảnh mẫu có sẵn để người dùng click và test nhanh khả năng của mô hình mà không cần tải ảnh lên.
* **Dự đoán từ URL:** Cho phép người dùng dán URL của một file ảnh từ Internet để backend tải về và dự đoán (lưu ý: tốc độ tải ảnh phụ thuộc vào nguồn ảnh bên ngoài).
* **Hiển thị danh sách lớp:** Liệt kê rõ ràng 30 loại trái cây/rau củ mà mô hình đã được huấn luyện để nhận diện.

## Công nghệ sử dụng

* **Backend:** Python, Flask
* **Machine Learning:** PyTorch, ONNX Runtime
* **Xử lý ảnh:** Pillow (PIL), TorchVision
* **Tải dữ liệu:** Requests
* **Frontend:** HTML, CSS, JavaScript (Vanilla JS)
* **Quản lý Dependencies:** pip
* **Quản lý Version:** Git
* **Hosting Code:** GitHub

## Mô hình Deep Learning

* **Kiến trúc:** Sử dụng mô hình **EfficientNetB0**, một kiến trúc mạng nơ-ron tích chập hiệu quả về tham số.
* **Tập dữ liệu:** Mô hình được fine-tuned trên một tập dữ liệu gồm ~30,000 ảnh thuộc 30 loại trái cây và rau củ khác nhau.
* **Deployment:** Mô hình PyTorch đã huấn luyện được xuất sang định dạng **ONNX** để tối ưu hóa cho quá trình suy luận (inference) hiệu quả và độc lập với framework PyTorch gốc trong môi trường triển khai (sử dụng ONNX Runtime).

## Cài đặt và Chạy cục bộ

Để chạy ứng dụng này trên máy tính của bạn:

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/KimLyNgan/FruitClassifiierApp.git
    ```

2.  **Tạo và Kích hoạt Môi trường ảo (Optional nhưng Recommended):**
    ```bash
    # Đối với Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Đối với macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Cài đặt các Thư viện cần thiết:**
    Đảm bảo môi trường ảo đang hoạt động, sau đó cài đặt các thư viện từ `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải File Mô hình ONNX:**
    * Bạn cần tải file mô hình ONNX đã export (`.onnx`) từ Kaggle hoặc nguồn khác mà bạn đã lưu.
    * Đặt file `.onnx` này vào một vị trí trên máy tính của bạn.
    * **Cập nhật đường dẫn file ONNX** trong file `app.py`, tìm biến `ONNX_MODEL_LOCAL_PATH` và sửa lại đường dẫn cho đúng.

5.  **Chuẩn bị Ảnh Mẫu (Optional):**
    * Nếu bạn muốn sử dụng tính năng ảnh mẫu, tạo thư mục `static/examples/` trong thư mục gốc project của bạn.
    * Đặt các file ảnh mẫu vào thư mục đó.
    * Cập nhật danh sách tên file ảnh mẫu trong biến `EXAMPLE_IMAGES` ở file `app.py`.

6.  **Chạy Ứng dụng Flask:**
    Trong Terminal, ở thư mục gốc project và trong môi trường ảo đã kích hoạt:
    ```bash
    # Đảm bảo biến môi trường FLASK_APP được set đúng tên file chính (app.py)
    # Đối với Windows
    set FLASK_APP=app.py

    # Đối với macOS / Linux
    export FLASK_APP=app.py

    # Chạy server Flask trên cổng 5050
    flask run --host 0.0.0.0 --port 5050
    ```

7.  **Truy cập ứng dụng:**
    Mở trình duyệt web và truy cập địa chỉ: `http://127.0.0.1:5050/`

## Cấu trúc Project

FruitVegClassifierApp/
├── .venv/              # Môi trường ảo (nếu sử dụng)
├── static/             # Chứa các file tĩnh (CSS, JS, ảnh mẫu)
│   └── examples/       # Ảnh mẫu
├── templates/          # Chứa các file HTML
│   └── index.html      # Giao diện chính của ứng dụng
├── app.py              # File Flask chính, định nghĩa routes, tải model
├── model_inference.py  # Chứa logic tải ONNX và chạy suy luận
├── utils.py            # Chứa các hàm tiện ích (transforms, danh sách lớp)
├── requirements.txt    # Danh sách các thư viện Python cần cài đặt
└── README.md           # File README