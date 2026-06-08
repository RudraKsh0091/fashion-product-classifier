import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import tensorflow as tf
import keras

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs('static', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

model = None
le_classes = None

def load_model():
    global model, le_classes
    print("[StyleScan] Loading model...")
    model = keras.saving.load_model('models/fashion_classifier.keras')
    le_classes = np.load('models/label_classes.npy', allow_pickle=True)
    print(f"[StyleScan] Model loaded. Classes: {len(le_classes)}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le_classes is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        img_tensor, pil_img = preprocess_image(image_bytes)

        preds = model.predict(img_tensor, verbose=0)[0]
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
        'model_loaded': model is not None,
        'classes': len(le_classes) if le_classes is not None else 0
    })

try:
    load_model()
except Exception as e:
    print(f"[StyleScan] Startup error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)