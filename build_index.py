"""
build_index.py — Run this ONCE before starting the app.
Extracts MobileNetV2 feature vectors for all dataset images
and saves them as a compressed numpy index for fast similarity search.

Usage:
    python build_index.py --images_dir /path/to/fashion-dataset/images --csv /path/to/styles.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tqdm import tqdm

def build_index(images_dir, csv_path, model_path='models/fashion_classifier.keras',
                output_path='models/feature_index.npz', batch_size=64, max_per_class=50):

    print("[build_index] Loading model...")
    model = keras.models.load_model(model_path)

    # Feature extractor — output of GlobalAveragePooling2D
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )

    print("[build_index] Loading CSV...")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    counts = df['articleType'].value_counts()
    valid_classes = counts[counts >= 200].index.tolist()
    df = df[df['articleType'].isin(valid_classes)].copy()

    # limit to max_per_class per category to keep index small
    df = df.groupby('articleType').head(max_per_class).reset_index(drop=True)

    # filter to existing images
    df['image_path'] = df['id'].astype(str).apply(lambda x: os.path.join(images_dir, x + '.jpg'))
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"[build_index] Indexing {len(df)} products...")

    all_features = []
    all_ids = []
    all_labels = []

    def preprocess(path):
        img = Image.open(path).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    # process in batches
    for start in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[start:start + batch_size]
        imgs = []
        valid_rows = []
        for _, row in batch.iterrows():
            try:
                imgs.append(preprocess(row['image_path']))
                valid_rows.append(row)
            except Exception:
                continue

        if not imgs:
            continue

        batch_tensor = np.stack(imgs)
        features = feature_extractor.predict(batch_tensor, verbose=0)

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        all_features.append(features)
        all_ids.extend([r['id'] for r in valid_rows])
        all_labels.extend([r['articleType'] for r in valid_rows])

    all_features = np.vstack(all_features).astype(np.float32)
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)

    np.savez_compressed(output_path,
                        features=all_features,
                        ids=all_ids,
                        labels=all_labels)

    print(f"[build_index] Saved {len(all_ids)} vectors to {output_path}")
    print(f"[build_index] Index shape: {all_features.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True, help='Path to dataset images folder')
    parser.add_argument('--csv', required=True, help='Path to styles.csv')
    parser.add_argument('--max_per_class', type=int, default=50, help='Max products per class in index')
    args = parser.parse_args()

    build_index(
        images_dir=args.images_dir,
        csv_path=args.csv,
        max_per_class=args.max_per_class
    )
