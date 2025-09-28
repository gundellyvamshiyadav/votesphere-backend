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
from scipy.spatial import distance as dist

# --- Professional Initialization ---
# This ensures the API can find the secret key you uploaded to Render.
SERVICE_ACCOUNT_FILE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')
db = None
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"🔥 CRITICAL ERROR: Failed to initialize Firebase Admin SDK.")
    print(f"🔥 Ensure 'serviceAccountKey.json' is uploaded as a Secret File on Render.")
    print(f"🔥 Error details: {e}")

app = Flask(__name__)
CORS(app)

# --- AI Model Initialization ---
# We initialize the models once when the server starts for better performance.
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Liveness Detection Logic (Adapted from your original, working code) ---
# These are the specific landmark indices for the eyes from MediaPipe
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# When the EAR drops below this value, the eye is considered closed.
EYE_AR_THRESH = 0.20

def calculate_ear(eye_landmarks, frame_shape):
    """Calculates the Eye Aspect Ratio for a single eye."""
    coords_px = [(int(p.x * frame_shape[1]), int(p.y * frame_shape[0])) for p in eye_landmarks]
    A = dist.euclidean(coords_px[1], coords_px[5])
    B = dist.euclidean(coords_px[2], coords_px[4])
    C = dist.euclidean(coords_px[0], coords_px[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- API Endpoints ---
@app.route('/register', methods=['POST'])
def register_face():
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
        
        # --- Liveness Check: The API is stateless, so we perform a single-frame "eyes open" check. ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_instance.process(rgb_frame)

        if not results.multi_face_landmarks:
             return jsonify({"verified": False, "reason": "Liveness check failed: No face visible."})

        face_landmarks = results.multi_face_landmarks[0].landmark
        
        left_eye = [face_landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [face_landmarks[i] for i in RIGHT_EYE_INDICES]

        left_ear = calculate_ear(left_eye, frame.shape)
        right_ear = calculate_ear(right_eye, frame.shape)
        
        ear = (left_ear + right_ear) / 2.0

        # If the EAR is below the threshold, the eyes are closed. This fails the liveness check.
        if ear < EYE_AR_THRESH:
            return jsonify({"verified": False, "reason": "Liveness check failed: Please keep your eyes open and look at the camera."})

        # --- If Liveness Passes, Proceed to Verification ---
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        stored_embedding_list = user_doc.to_dict().get('faceEmbedding')
        if not stored_embedding_list:
             return jsonify({"verified": False, "reason": "No face registered for this user."})
        
        stored_embedding = np.array(stored_embedding_list)

        result = DeepFace.verify(
            img1_path=frame, 
            img2_path=stored_embedding, 
            model_name='ArcFace', 
            detector_backend='retinaface',
            enforce_detection=False
        )

        return jsonify(result)

    except ValueError as ve:
        return jsonify({"verified": False, "reason": "Could not detect a face in the live video."})
    except Exception as e:
        print(f"🔥 VERIFY ERROR: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))