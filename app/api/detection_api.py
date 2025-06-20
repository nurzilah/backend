from flask import Blueprint, request, jsonify
import threading
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import base64

detection_api = Blueprint('detection_api', _name_)

# Global variables untuk model
model = None
scaler = None
mp_face_mesh = None
face_mesh = None
lock = threading.Lock()

# Feature columns sesuai dengan training model
FEATURE_COLUMNS = ['eyebrow_dist', 'eye_asymmetry', 'mar', 'mouth_asymmetry', 'pucker_asymmetry']

def load_model():
    """Load model dan scaler saat startup"""
    global model, scaler, mp_face_mesh, face_mesh
    
    try:
        # Load model ML
        model_path = os.path.join('models', 'model_mlp.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Initialize MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        print("✅ Bell's Palsy detection model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading Bell's Palsy model: {e}")
        return False

def euclidean(p1, p2):
    """Hitung jarak euclidean antara dua titik"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def decode_base64_image(image_data):
    """Decode base64 image ke format OpenCV"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_features_from_landmarks(landmarks):
    """Extract fitur dari MediaPipe landmarks"""
    try:
        def get_point(i):
            return (landmarks.landmark[i].x, landmarks.landmark[i].y)

        # Ekstraksi fitur sesuai dengan training model
        left_eyebrow = get_point(55)
        right_eyebrow = get_point(285)
        eyebrow_dist = euclidean(left_eyebrow, right_eyebrow)

        # Eye Aspect Ratio calculations
        left_eye_top = get_point(159)
        left_eye_bottom = get_point(145)
        left_eye_left = get_point(33)
        left_eye_right = get_point(133)
        left_ear = euclidean(left_eye_top, left_eye_bottom) / (euclidean(left_eye_left, left_eye_right) + 1e-8)

        right_eye_top = get_point(386)
        right_eye_bottom = get_point(374)
        right_eye_left = get_point(362)
        right_eye_right = get_point(263)
        right_ear = euclidean(right_eye_top, right_eye_bottom) / (euclidean(right_eye_left, right_eye_right) + 1e-8)

        eye_asymmetry = abs(left_ear - right_ear)

        # Mouth Aspect Ratio
        top_lip = get_point(13)
        bottom_lip = get_point(14)
        left_mouth = get_point(61)
        right_mouth = get_point(291)
        mar = euclidean(top_lip, bottom_lip) / (euclidean(left_mouth, right_mouth) + 1e-8)

        mouth_asymmetry = abs(left_mouth[1] - right_mouth[1])
        
        left_pucker = get_point(78)
        right_pucker = get_point(308)
        pucker_asymmetry = abs(left_pucker[0] - right_pucker[0])

        feature_dict = {
            'eyebrow_dist': eyebrow_dist,
            'eye_asymmetry': eye_asymmetry,
            'mar': mar,
            'mouth_asymmetry': mouth_asymmetry,
            'pucker_asymmetry': pucker_asymmetry
        }

        return pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@detection_api.route('/predict_bellspalsy', methods=['POST'])
def predict_bellspalsy():
    """Endpoint untuk prediksi Bell's Palsy"""
    global model, scaler, face_mesh
    
    if model is None or scaler is None or face_mesh is None:
        return jsonify({
            'success': False,
            'error': 'Bell\'s Palsy detection model not loaded'
        }), 503
    
    try:
        data = request.json
        
        if 'frames' not in data:
            return jsonify({
                'success': False,
                'error': 'No frames provided'
            }), 400
        
        frames = data['frames']
        print(f"📸 Processing {len(frames)} frames for Bell's Palsy detection")
        
        # Process frames dengan thread lock
        with lock:
            features_list = []
            processed_frames = 0
            
            for i, frame_data in enumerate(frames):
                image = decode_base64_image(frame_data)
                if image is None:
                    continue
                
                # Convert ke RGB untuk MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        feature_df = extract_features_from_landmarks(face_landmarks)
                        if feature_df is not None:
                            features_list.append(feature_df)
                            processed_frames += 1
            
            print(f"✅ Successfully processed {processed_frames} frames with face landmarks")
            
            if not features_list:
                return jsonify({
                    'success': False,
                    'error': 'No valid face landmarks detected in any frame'
                }), 400
            
            # Gabungkan semua features
            all_features = pd.concat(features_list, ignore_index=True)
            
            # Normalisasi menggunakan scaler
            features_scaled = scaler.transform(all_features)
            
            # Prediksi menggunakan model
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)[:, 1]
            
            # Analisis hasil
            total_frames = len(predictions)
            bellspalsy_frames = int((predictions == 1).sum())
            normal_frames = int((predictions == 0).sum())
            avg_probability = float(probabilities.mean())
            
            # Keputusan akhir
            threshold = 0.5
            is_positive = avg_probability >= threshold
            
            confidence_level = "Tinggi" if (avg_probability > 0.7 or avg_probability < 0.3) else "Sedang"
            
            result = {
                'success': True,
                'is_positive': is_positive,
                'prediction': 'Bell\'s Palsy' if is_positive else 'Normal',
                'confidence': avg_probability,
                'confidence_level': confidence_level,
                'percentage': round(avg_probability * 100, 1),
                'total_frames': total_frames,
                'bellspalsy_frames': bellspalsy_frames,
                'normal_frames': normal_frames,
                'probabilities': {
                    'normal': float(1 - avg_probability),
                    'bells_palsy': avg_probability
                }
            }
            
            print(f"🎯 Prediction: {result['prediction']} with {result['percentage']}% confidence")
            return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in Bell's Palsy prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@detection_api.route('/detection_health', methods=['GET'])
def detection_health():
    """Health check untuk detection API"""
    global model, scaler, face_mesh
    
    return jsonify({
        'status': 'healthy' if all([model is not None, scaler is not None, face_mesh is not None]) else 'unhealthy',
        'message': 'Bell\'s Palsy Detection API',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'mediapipe_initialized': face_mesh is not None
    })

# Load model saat blueprint diimport
load_model()