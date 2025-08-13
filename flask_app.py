# flask_app.py

import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model
model = load_model('trained_model.h5')

# Class names corresponding to the model's output
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Optional: Add disease-specific suggestions
disease_info = {
    'Apple___Apple_scab': {
        'suggestion': 'Use fungicides and remove infected leaves.',
        'severity': 'medium'
    },
    'Apple___Black_rot': {
        'suggestion': 'Prune infected branches and use copper sprays.',
        'severity': 'high'
    },
    'Corn_(maize)___Common_rust_': {
        'suggestion': 'Apply resistant hybrid seeds and fungicide if severe.',
        'severity': 'medium'
    },
    'Tomato___Late_blight': {
        'suggestion': 'Remove infected leaves and apply appropriate fungicides.',
        'severity': 'high'
    },
    # You can extend this for all classes as needed
}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((128, 128))
        img_arr = image.img_to_array(img)
        # img_arr = img_arr / 255.0  # normalize if model expects normalized input
        img_arr = np.expand_dims(img_arr, axis=0)

        # Make prediction
        prediction = model.predict(img_arr)
        result_index = np.argmax(prediction)
        label = class_names[result_index]
        confidence = float(np.max(prediction))

        # Get additional info
        info = disease_info.get(label, {
            'suggestion': 'No specific suggestion available.',
            'severity': 'unknown'
        })

        return jsonify({
            'prediction_index': int(result_index),
            'label': label,
            'confidence': confidence,
            'suggestion': info['suggestion'],
            'severity': info['severity']
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    
# port = int(os.environ.get('PORT', 5000))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# import os
# import io
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # --- Make TF faster/more predictable on small CPU instances ---
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
# os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
# # If you see odd slowdowns, try disabling oneDNN:
# # os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# app = Flask(__name__)
# CORS(app)

# # --------- Labels ----------
# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#     'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#     'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# disease_info = {
#     'Apple___Apple_scab': {'suggestion': 'Use fungicides and remove infected leaves.','severity': 'medium'},
#     'Apple___Black_rot': {'suggestion': 'Prune infected branches and use copper sprays.','severity': 'high'},
#     'Corn_(maize)___Common_rust_': {'suggestion': 'Apply resistant hybrid seeds and fungicide if severe.','severity': 'medium'},
#     'Tomato___Late_blight': {'suggestion': 'Remove infected leaves and apply appropriate fungicides.','severity': 'high'},
# }

# # --------- Model handling ----------
# model = None
# MODEL_PATH = "trained_model.h5"

# def get_model():
#     """Lazy load (used as fallback if preload fails)."""
#     global model
#     if model is None:
#         from tensorflow.keras.models import load_model
#         print(f"🔍 Looking for model at: {os.path.abspath(MODEL_PATH)}")
#         print("📂 CWD contents:", os.listdir())
#         if not os.path.exists(MODEL_PATH):
#             raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
#         print("⏳ Loading model...")
#         model = load_model(MODEL_PATH)
#         print("✅ Model loaded (lazy).")
#     return model

# def warmup():
#     """Preload and run one dummy prediction so first real request is fast."""
#     try:
#         mdl = get_model()
#         dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
#         _ = mdl.predict(dummy)
#         print("🔥 Warmup inference completed.")
#     except Exception as e:
#         # Don't crash app on warmup failure; prediction path can still lazy-load.
#         print(f"⚠️ Warmup failed: {e}")

# # Preload/warmup at import time so Gunicorn --preload can do it before the first request
# if os.environ.get("PRELOAD_MODEL", "1") == "1":
#     warmup()

# # --------- Routes ----------
# @app.route("/", methods=["GET"])
# def root():
#     return jsonify({"status": "ok", "message": "Plant Disease API is running"}), 200

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok"}), 200

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     try:
#         # Read & preprocess image safely
#         raw = file.read()
#         if not raw:
#             return jsonify({'error': 'Empty file'}), 400

#         img = Image.open(io.BytesIO(raw)).convert('RGB')
#         img = img.resize((128, 128))
#         img_arr = image.img_to_array(img)
#         img_arr = np.expand_dims(img_arr, axis=0).astype(np.float32)
#         # If your model expects normalization, uncomment the next line:
#         # img_arr = img_arr / 255.0

#         mdl = get_model()
#         preds = mdl.predict(img_arr)
#         idx = int(np.argmax(preds))
#         label = class_names[idx]
#         confidence = float(np.max(preds))

#         info = disease_info.get(label, {'suggestion': 'No specific suggestion available.','severity': 'unknown'})

#         return jsonify({
#             'prediction_index': idx,
#             'label': label,
#             'confidence': confidence,
#             'suggestion': info['suggestion'],
#             'severity': info['severity']
#         }), 200

#     except Exception as e:
#         return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)
