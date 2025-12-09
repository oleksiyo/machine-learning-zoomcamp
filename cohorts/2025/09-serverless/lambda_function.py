from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def pre_process_image(img):
    x = np.array(img).astype("float32") / 255

    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    x_norm = (x - mean) / std
    return x_norm

def apply_model(img_array, model_path='hair_classifier_empty.onnx'):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    ort_session = ort.InferenceSession(model_path)

    # add batch dimension
    X = np.expand_dims(img_array, axis=0)

    # ✅ CHANGE CHANNEL ORDER
    X = np.transpose(X, (0, 3, 1, 2))

    # ✅ float32
    X = X.astype(np.float32)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    outputs = ort_session.run([output_name], {input_name: X})
    return outputs[0][0][0]


def lambda_handler(event, context=None):
    url = event.get("url")
    if not url:
        return {"error": "No 'url' key found"}
    try:
        img = download_image(url)
        img = prepare_image(img, (200, 200))

        img_array = pre_process_image(img)
        output = apply_model(img_array)
        
        return {"Prediction": float(output)}
    except Exception as e:
        return {"error": str(e)}