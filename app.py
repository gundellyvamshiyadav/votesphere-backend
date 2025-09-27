import os
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from deepface import DeepFace
import mediapipe as mp
import traceback

# --- Professional Initialization with Error Handling ---
# This is the secret "ID card" for our API.
# Render automatically sets the GOOGLE_APPLICATION_CREDENTIALS path
# to where it stores our secret file.
SERVICE_ACCOUNT_FILE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')
db = None

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin SDK initialized successfully.")
except Exception as e:
    # This robust error handling is CRITICAL.
    # The server will still start, even if the key is missing.
    print(f"🔥 CRITICAL ERROR: Failed to initialize Firebase Admin SDK.")
    print(f"🔥 Make sure the '{SERVICE_ACCOUNT_FILE}' secret file is uploaded to Render.")
    print(f"🔥 Error details: {e}")

app = Flask(__name__)
# Allow requests from your React app (localhost and your future deployed site)
CORS(app)

# --- Liveness Detection Logic (from your working code) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_blink_ratio(landmarks, frame_shape):
    # This logic remains the same as your working version
    try:
        rh_right = (landmarks.landmark[33].x * frame_shape[1], landmarks.landmark[33].y * frame_shape[0])
        rh_left = (landmarks.landmark[133].x * frame_shape[1], landmarks.landmark[133].y * frame_shape[0])
        rv_top = (landmarks.landmark[159].x * frame_shape[1], landmarks.landmark[159].y * frame_shape[0])
        rv_bottom = (landmarks.landmark[145].x * frame_shape[1], landmarks.landmark[145].y * frame_shape[0])
        
        rh_dist = np.linalg.norm(np.array(rh_right) - np.array(rh_left))
        rv_dist = np.linalg.norm(np.array(rv_top) - np.array(rv_bottom))
        
        return rh_dist / (rv_dist + 1e-6)
    except Exception:
        return None

# --- API Endpoints ---
@app.route('/register', methods=['POST'])
def register_face():
    # This check prevents the app from crashing if Firebase failed to initialize
    if not db:
        return jsonify({"error": "Database connection failed. Check server logs."}), 500

    data = request.get_json()
    user_id = data.get('userId')
    image_data_url = data.get('imageDataUrl')

    if not user_id or not image_data_url:
        return jsonify({"error": "Missing userId or imageDataUrl"}), 400

    try:
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        embedding_objs = DeepFace.represent(
            img_path=frame, model_name='ArcFace', enforce_detection=True, detector_backend='retinaface'
        )
        embedding = embedding_objs[0]['embedding']

        user_ref = db.collection('users').document(user_id)
        user_ref.update({
            'faceEmbedding': embedding,
            'faceRegistered': True,
            'isNewUser': False
        })

        return jsonify({"message": "Face registered successfully"}), 200

    except ValueError:
        return jsonify({"error": "No face detected. Please ensure your face is clear and centered."}), 400
    except Exception as e:
        print(f"🔥 REGISTER ERROR: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500


@app.route('/verify', methods=['POST'])
def verify_face():
    if not db:
        return jsonify({"error": "Database connection failed. Check server logs."}), 500

    data = request.get_json()
    user_id = data.get('userId')
    image_data_url = data.get('imageDataUrl')

    if not user_id or not image_data_url:
        return jsonify({"error": "Missing userId or imageDataUrl"}), 400

    try:
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # --- Liveness Check ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_instance.process(rgb_frame)

        if not results.multi_face_landmarks:
             return jsonify({"verified": False, "reason": "Liveness check failed: No face visible."})

        ratio = get_blink_ratio(results.multi_face_landmarks[0], frame.shape)
        
        # A low ratio means the eye is closed (a blink). A high ratio means it's open.
        # We verify only when eyes are OPEN.
        if ratio is None or ratio < 3.5:
            return jsonify({"verified": False, "reason": "Liveness check failed: Please keep your eyes open."})

        # --- Verification Logic ---
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        stored_embedding_list = user_doc.to_dict().get('faceEmbedding')
        if not stored_embedding_list:
             return jsonify({"verified": False, "reason": "No face registered for this user."})
        
        # Convert list back to numpy array for verification
        stored_embedding = np.array(stored_embedding_list)

        result = DeepFace.verify(
            img1_path=frame, 
            img2_path=stored_embedding, 
            model_name='ArcFace', 
            detector_backend='retinaface',
            enforce_detection=False # Don't crash if the face disappears for a frame
        )

        return jsonify(result)

    except ValueError as ve:
        # This can happen if DeepFace fails to find a face in the live frame
        return jsonify({"verified": False, "reason": f"Could not detect a face in the live video."})
    except Exception as e:
        print(f"🔥 VERIFY ERROR: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # This is for local testing if you ever need it
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

