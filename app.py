import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from deepface import DeepFace

# --- Initialize Flask App ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow our React app to call this API
CORS(app)

# --- Database Setup (using local files like your original script) ---
DATABASE_DIR = 'voter_database'
EMBEDDINGS_FILE = os.path.join(DATABASE_DIR, 'voter_embeddings.pkl')

if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

def load_database():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(db, f)

# --- Liveness Detection Helper ---
def is_live(image_np):
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    face_mesh.close()
    return results.multi_face_landmarks is not None

# --- Helper to decode images from React App ---
def decode_image(image_data_b64):
    import base64
    image_data = base64.b64decode(image_data_b64)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- API Endpoints ---

@app.route('/api/register', methods=['POST'])
def register_face():
    try:
        data = request.json
        user_id = data['userId']
        image_data_b64 = data['imageData']
        
        img = decode_image(image_data_b64)

        embedding_obj = DeepFace.represent(
            img_path=img, 
            model_name='ArcFace', 
            enforce_detection=True,
            detector_backend='retinaface'
        )
        embedding = embedding_obj[0]["embedding"]

        db = load_database()
        db[user_id] = embedding
        save_database(db)

        return jsonify({"status": "success", "message": "Face registered successfully."})
    except ValueError as ve:
        return jsonify({"status": "error", "message": "No face detected. Please try again."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        user_id = data['userId']
        image_data_b64 = data['imageData']

        live_img = decode_image(image_data_b64)
        
        if not is_live(live_img):
             return jsonify({"verified": False, "error": "Liveness check failed."}), 403

        db = load_database()
        if user_id not in db:
            return jsonify({"verified": False, "error": "User not found in face database."}), 404
        
        stored_embedding = db[user_id]

        result = DeepFace.verify(
            img1_path=live_img,
            img2_path=np.array(stored_embedding),
            model_name='ArcFace',
            enforce_detection=True,
            detector_backend='retinaface'
        )
        
        return jsonify({"verified": bool(result["verified"])})
    except ValueError:
        return jsonify({"verified": False, "error": "No face detected in live image."})
    except Exception as e:
        return jsonify({"verified": False, "error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
