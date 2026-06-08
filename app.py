import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ── Model & label loading ──────────────────────────────────────────────────────
model = None
feature_extractor = None
le_classes = None
similar_products_index = None  # precomputed feature vectors

def load_model():
    global model, feature_extractor, le_classes

    print("[StyleScan] Loading model...")
    model = keras.models.load_model('models/fashion_classifier.keras')
    le_classes = np.load('models/label_classes.npy', allow_pickle=True)

    # Feature extractor: same base but output before final Dense
    # Reuse layers from loaded model
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # output of GlobalAveragePooling2D
    )
    print(f"[StyleScan] Model loaded. Classes: {len(le_classes)}")

def preprocess_image(image_bytes):
    """Preprocess image bytes for MobileNetV2."""
    import tensorflow as tf
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

def load_similar_products_index():
    """Load precomputed feature index for similar products search."""
    global similar_products_index
    index_path = 'models/feature_index.npz'
    if os.path.exists(index_path):
        data = np.load(index_path, allow_pickle=True)
        similar_products_index = {
            'features': data['features'],
            'ids': data['ids'],
            'labels': data['labels']
        }
        print(f"[StyleScan] Feature index loaded: {len(similar_products_index['ids'])} products")
    else:
        print("[StyleScan] No feature index found. Run build_index.py first.")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_extractor is None or le_classes is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_bytes = file.read()

    # preprocess
    img_tensor, pil_img = preprocess_image(image_bytes)

    # ── Top-3 predictions ──
    preds = model.predict(img_tensor, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    predictions = [
        {
            'label': str(le_classes[idx]),
            'confidence': float(preds[idx]) * 100
        }
        for idx in top3_idx
    ]

    # ── Similar products ──
    similar = []
    if similar_products_index is not None:
        query_features = feature_extractor.predict(img_tensor, verbose=0)[0]
        query_features = query_features / (np.linalg.norm(query_features) + 1e-8)

        all_features = similar_products_index['features']
        scores = all_features @ query_features  # cosine similarity (features are pre-normalized)

        top5_idx = np.argsort(scores)[::-1][:5]
        for idx in top5_idx:
            product_id = str(similar_products_index['ids'][idx])
            img_path = f'static/dataset_images/{product_id}.jpg'
            similar.append({
                'id': product_id,
                'label': str(similar_products_index['labels'][idx]),
                'score': float(scores[idx]),
                'image_path': img_path
            })

    # save uploaded image for display
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'last_upload.jpg')
    pil_img.save(save_path)

    return jsonify({
        'predictions': predictions,
        'similar_products': similar,
        'uploaded_image': 'static/uploads/last_upload.jpg'
    })

@app.route('/classes')
def get_classes():
    if le_classes is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({'classes': le_classes.tolist(), 'count': len(le_classes)})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(le_classes) if le_classes is not None else 0,
        'index_loaded': similar_products_index is not None
    })

# ── Entry point ────────────────────────────────────────────────────────────────

try:
    load_model()
    load_similar_products_index()
except Exception as e:
    print(f"Startup failed: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
