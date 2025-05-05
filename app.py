from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import io
from model_inference import load_onnx_model, predict_image_with_onnx
from utils import CLASSES
import requests
import time

app = Flask(__name__)

ONNX_MODEL_LOCAL_PATH = 'fruit_vegetable_classifier_efficientnetb0.onnx'


print('>>> Flask startup: Loading ONNX model...')
try:
    load_onnx_model(ONNX_MODEL_LOCAL_PATH)
    print('>>> Flask startup: ONNX model loaded successfully.')
except Exception as e:
    print(f'>>> Flask startup: Failed to load ONNX model: {e}')
    raise e

EXAMPLE_IMAGES =[
    'bap ngo.jpg',
    'du du.jpg',
    'dua chuot.jpg',
    'dua hau.png',
    'qua buoi.jpg',
    'qua cam.jpg',
    'qua ot.jpg',
    'quaxoai.jpg',
    'ca tim.jpg',
    'qua oi.png'
]


@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES, example_images=EXAMPLE_IMAGES)

@app.route('/predict', methods=['POST'])
def predict():
    print('>>> Predict endpint called')
    if 'file' not in request.files:
        print('>>> No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        print('>>> No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file:
        print(f'>>> File received: {file.filename}')
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            prediction_result = predict_image_with_onnx(img)
            print(f'>>> Prediction result: {prediction_result}')
            return jsonify(prediction_result)
        except FileNotFoundError:
            # Lỗi này có thể xảy ra nếu ONNX_MODEL_LOCAL_PATH sai ngay từ đầu
            print(f">>> /predict endpoint: Lỗi: File ONNX model không tồn tại.")
            return jsonify({"error": "Server gặp lỗi tải mô hình."}), 500
        except ValueError as ve:
            # Lỗi này có thể xảy ra nếu ort_session là None
            print(f">>> /predict endpoint: Lỗi ValueError: {ve}")
            return jsonify({"error": f"Server gặp lỗi mô hình: {ve}"}), 500
        except Exception as e:
            print(f">>> /predict endpoint: Lỗi xử lý hoặc suy luận: {e}")
            # Trả về lỗi server nếu có bất kỳ exception nào khác
            return jsonify({"error": f"Lỗi xử lý ảnh hoặc mô hình: {e}"}), 500
    print(">>> /predict endpoint: Lỗi không xác định khi xử lý file.")
    return jsonify({"error": "Lỗi không xác định."}), 500

@app.route('/predict_from_url', methods=['POST'])
def predict_from_url():
    print(">>> /predict_from_url endpoint hit!")

    # Nhận dữ liệu JSON từ request body
    data = request.get_json()

    if not data or 'image_url' not in data or not data['image_url']:
        print(">>> /predict_from_url endpoint: Không nhận được 'image_url' trong JSON body.")
        return jsonify({"error": "Vui lòng cung cấp 'image_url' trong JSON body."}), 400

    image_url = data['image_url']
    print(f">>> /predict_from_url endpoint: Nhận URL: {image_url}")

    # --- Tải ảnh từ URL sử dụng requests ---
    start_time_fetch = time.time() # <-- Bắt đầu đo thời gian fetch
    try:
        response = requests.get(image_url, stream=True, timeout=15) # Tăng timeout lên 15s cho chắc

        if response.status_code != 200:
            print(f">>> /predict_from_url endpoint: Lỗi tải ảnh từ URL. Status: {response.status_code}")
            return jsonify({"error": f"Không tải được ảnh từ URL. Status Code: {response.status_code}"}), 400

        content_type = response.headers.get('Content-Type')
        if not content_type or not content_type.startswith('image/'):
            print(f">>> /predict_from_url endpoint: URL không trỏ đến file ảnh. Content-Type: {content_type}")
            return jsonify({"error": f"URL không trỏ đến file ảnh. Content-Type: {content_type}"}), 400

        image_bytes = response.content
        if not image_bytes:
             print(f">>> /predict_from_url endpoint: Dữ liệu ảnh từ URL rỗng.")
             return jsonify({"error": "Dữ liệu ảnh từ URL rỗng."}), 400

        end_time_fetch = time.time() # <-- Kết thúc đo thời gian fetch
        print(f">>> /predict_from_url endpoint: Tải ảnh từ URL hoàn tất sau {end_time_fetch - start_time_fetch:.4f} giây.")

        start_time_pil = time.time() # <-- Bắt đầu đo thời gian xử lý PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        end_time_pil = time.time() # <-- Kết thúc đo thời gian xử lý PIL
        print(f">>> /predict_from_url endpoint: Xử lý ảnh PIL hoàn tất sau {end_time_pil - start_time_pil:.4f} giây.")


    except requests.exceptions.MissingSchema:
        print(f">>> /predict_from_url endpoint: Lỗi URL không hợp lệ: Thiếu Schema (http/https).")
        return jsonify({"error": "URL không hợp lệ. Hãy bắt đầu bằng http:// hoặc https:// ."}), 400
    except requests.exceptions.ConnectionError:
         print(f">>> /predict_from_url endpoint: Lỗi kết nối khi tải ảnh từ URL.")
         return jsonify({"error": "Không kết nối được đến URL ảnh."}), 400
    except requests.exceptions.Timeout:
         print(f">>> /predict_from_url endpoint: Hết thời gian chờ khi tải ảnh từ URL.")
         return jsonify({"error": "Hết thời gian chờ khi tải ảnh từ URL."}), 408
    except requests.exceptions.RequestException as e:
        print(f">>> /predict_from_url endpoint: Lỗi khi tải ảnh từ URL: {e}")
        return jsonify({"error": f"Lỗi khi tải ảnh từ URL: {e}"}), 500
    except Exception as e:
        print(f">>> /predict_from_url endpoint: Lỗi xử lý ảnh sau khi tải (PIL): {e}") # Ghi rõ là lỗi PIL
        return jsonify({"error": f"Lỗi xử lý ảnh sau khi tải: {e}"}), 500


    # --- Chạy suy luận bằng hàm đã tạo trong model_inference ---
    start_time_predict = time.time() # <-- Bắt đầu đo thời gian dự đoán
    try:
        prediction_result = predict_image_with_onnx(img) # Gọi hàm dự đoán ONNX
        end_time_predict = time.time() # <-- Kết thúc đo thời gian dự đoán
        print(f">>> /predict_from_url endpoint: Suy luận ONNX hoàn tất sau {end_time_predict - start_time_predict:.4f} giây.")

        print(f">>> /predict_from_url endpoint: Dự đoán hoàn tất. Kết quả: {prediction_result}")
        return jsonify(prediction_result)

    except ValueError as ve:
         print(f">>> /predict_from_url endpoint: Lỗi ValueError khi dự đoán: {ve}")
         return jsonify({"error": f"Lỗi mô hình khi dự đoán: {ve}"}), 500
    except Exception as e:
        print(f">>> /predict_from_url endpoint: Lỗi suy luận: {e}")
        return jsonify({"error": f"Lỗi khi chạy suy luận mô hình: {e}"}), 500
