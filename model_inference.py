import onnxruntime as ort
import os
import numpy as np
from PIL import Image
from utils import preprocess_image, postprocess_output

ONNX_MODEL_PATH = 'frozen_inference_graph.onnx'
ort_session = None

def load_onnx_model(model_path: str):
    global ort_session
    global ONNX_MODEL_PATH
    ONNX_MODEL_PATH = model_path
    print(f'Đang tải mô hình ONNX từ {model_path}')
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f'Lỗi không tìm thấy file ONNX tại đường dẫn {ONNX_MODEL_PATH}')
        ort_session = None
        raise FileNotFoundError(f'Không tìm thấy file ONNX tại đường dẫn {ONNX_MODEL_PATH}')
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        print('Mô hình ONNX đã được tải thành công.')
    except Exception as e:
        print(f'Lỗi khi tải mô hình ONNX: {e}')
        ort_session = None
        raise e


def predict_image_with_onnx(image: Image.Image)->dict:
    if ort_session is None:
        raise ValueError('Mô hình chưa được tải')
    try:
        input_numpy = preprocess_image(image)
        onnx_input_name = ort_session.get_inputs()[0].name
        ort_inputs = {onnx_input_name: input_numpy}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_logits = ort_outputs[0]
        result = postprocess_output(ort_logits)
        return result
    except Exception as e:
        print(f'Lỗi khi dự đoán hình ảnh: {e}')
        raise e