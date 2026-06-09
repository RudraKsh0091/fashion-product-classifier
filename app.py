import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

interpreter = None
input_details = None
output_details = None
le_classes = None

def load_model():
    global interpreter, input_details, output_details, le_classes
    import tflite_runtime.interpreter as tflite
    print("[StyleScan] Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path='models/fashion_classifier.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    le_classes = np.load('models/label_classes.npy', allow_pickle=True)
    print(f"[StyleScan] Model loaded. Classes: {len(le_classes)}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0), img

def predict_tflite(img_tensor):
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None or le_classes is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        img_tensor, pil_img = preprocess_image(image_bytes)

        preds = predict_tflite(img_tensor)
        top3_idx = np.argsort(preds)[::-1][:3]
        predictions = [
            {
                'label': str(le_classes[idx]),
                'confidence': float(preds[idx]) * 100
            }
            for idx in top3_idx
        ]

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'last_upload.jpg')
        pil_img.save(save_path)

        return jsonify({
            'predictions': predictions,
            'similar_products': [],
            'uploaded_image': 'static/uploads/last_upload.jpg'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': interpreter is not None,
        'classes': len(le_classes) if le_classes is not None else 0
    })

try:
    load_model()
except Exception as e:
    print(f"[StyleScan] Startup error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)