import numpy as np
import keras
import time

model = keras.saving.load_model(
    "models/fashion_classifier.keras"
)

dummy = np.zeros((1,224,224,3), dtype=np.float32)

start = time.time()
preds = model.predict(dummy, verbose=0)
print("Time:", time.time() - start)

import os

size_mb = os.path.getsize(
    "models/fashion_classifier.keras"
) / (1024 * 1024)

print(size_mb)