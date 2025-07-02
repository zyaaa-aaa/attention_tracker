import cv2
import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime
import os
import sys
import torch
from collections import deque
from statistics import mode, multimode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

class AttentionDetectionModel:
    """Machine Learning Model for Attention Prediction"""
    
    def __init__(self, dataset_path="attention_detection_dataset_v1.csv"):
        self.dataset_path = dataset_path
        self.model_pipeline = None
        self.is_trained = False
        self.feature_columns = []  # Will be set dynamically from dataset
        self.expected_columns = [
            'no_of_face', 'face_x', 'face_y', 'face_w', 'face_h', 'face_con',
            'no_of_hands', 'pose', 'pose_x', 'pose_y', 'phone', 'phone_x', 
            'phone_y', 'phone_w', 'phone_h', 'phone_con'
        ]
        
    def load_and_train_model(self):
        """Load dataset and train the attention prediction model"""
        try:
            # Load the dataset
            print("📊 Loading attention detection dataset...")
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded: {len(df)} samples")
            
            # Display available columns
            print(f"📋 Available columns: {list(df.columns)}")
            
            # Check if 'label' column exists
            if 'label' not in df.columns:
                print("❌ Error: 'label' column not found in dataset!")
                print("🔧 Please ensure your dataset has a 'label' column for the target variable")
                return False
            
            # Automatically detect feature columns (all except 'label')
            self.feature_columns = [col for col in df.columns if col != 'label']
            print(f"📊 Using feature columns: {self.feature_columns}")
            
            # Check for missing expected columns and warn user
            missing_columns = [col for col in self.expected_columns if col not in self.feature_columns]
            if missing_columns:
                print(f"⚠️ Warning: Some expected columns are missing: {missing_columns}")
                print("🔧 The model will work with available columns, but performance may be affected")
            
            # Separate features and label
            X = df[self.feature_columns]
            y = df["label"]
            
            print(f"📊 Features shape: {X.shape}")
            print(f"📊 Target distribution: {y.value_counts().to_dict()}")
            
            # Identify categorical and numerical columns automatically
            categorical_cols = []
            numerical_cols = []
            
            for col in self.feature_columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_cols.append(col)
                else:
                    # Check if it looks like a categorical column (few unique values)
                    unique_values = X[col].nunique()
                    if unique_values <= 10 and col in ['pose']:  # Known categorical columns
                        categorical_cols.append(col)
                    else:
                        numerical_cols.append(col)
            
            print(f"📊 Categorical columns: {categorical_cols}")
            print(f"📊 Numerical columns: {numerical_cols}")
            
            # Create preprocessing pipeline
            if categorical_cols:
                # Both categorical and numerical columns
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                        ("num", "passthrough", numerical_cols)
                    ]
                )
            else:
                # Only numerical columns
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", numerical_cols)
                    ]
                )
            
            # Build pipeline with RandomForestClassifier
            self.model_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                ))
            ])
            
            # Check if we have enough samples for train/test split
            if len(df) < 10:
                print("⚠️ Warning: Very small dataset. Training on all data without test split.")
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, 
                    stratify=y if len(y.unique()) > 1 else None
                )
            
            # Fit model
            print("🤖 Training attention prediction model...")
            self.model_pipeline.fit(X_train, y_train)
            
            # Evaluate model
            try:
                y_proba = self.model_pipeline.predict_proba(X_test)
                if y_proba.shape[1] > 1:  # Binary classification
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba[:, 0]
                    
                y_pred = self.model_pipeline.predict(X_test)
                
                # Calculate metrics
                if len(y_test.unique()) > 1:  # Only calculate AUC if we have both classes
                    roc_auc = roc_auc_score(y_test, y_proba_pos)
                    print(f"📈 ROC-AUC Score: {roc_auc:.3f}")
                
                report = classification_report(y_test, y_pred, output_dict=True)
                print(f"📈 Accuracy: {report['accuracy']:.3f}")
                
                # Print class-specific metrics if available
                for class_label in report.keys():
                    if class_label.isdigit() or class_label in ['0', '1']:
                        class_name = "Attentive" if class_label == '1' else "Not Attentive"
                        if isinstance(report[class_label], dict):
                            print(f"📈 {class_name} - Precision: {report[class_label]['precision']:.3f}, "
                                  f"Recall: {report[class_label]['recall']:.3f}")
                
            except Exception as e:
                print(f"⚠️ Evaluation error (model still trained): {e}")
            
            print("✅ Model trained successfully!")
            self.is_trained = True
            return True
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {self.dataset_path}")
            print("🔧 Please ensure the dataset file exists in the current directory")
            print("📋 Expected format: CSV file with 'label' column and feature columns")
            return False
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            print("🔍 Full error traceback:")
            traceback.print_exc()
            return False
    
    def predict_attention(self, detection_data):
        """Predict attention level from detection data"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None, None
        
        try:
            # Convert detection data to DataFrame
            df = pd.DataFrame([detection_data])
            
            # Ensure all feature columns are present with default values
            for col in self.feature_columns:
                if col not in df.columns:
                    # Set default values based on column type
                    if col == 'pose':
                        df[col] = 'Forward'
                    else:
                        df[col] = 0
            
            # Select only the required columns in the correct order
            df = df[self.feature_columns]
            
            print(f"🔍 Prediction input shape: {df.shape}")
            print(f"🔍 Input columns: {list(df.columns)}")
            
            # Predict
            prediction = self.model_pipeline.predict(df)[0]
            probability = self.model_pipeline.predict_proba(df)[0]
            
            # Handle different probability array shapes
            if len(probability) > 1:
                attention_prob = probability[1]  # Probability of being attentive (class 1)
            else:
                attention_prob = probability[0] if prediction == 1 else (1 - probability[0])
            
            return int(prediction), float(attention_prob)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            print("🔍 Prediction error traceback:")
            traceback.print_exc()
            return None, None

class CameraDetectionSystem:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.auto_capture = False
        self.capture_interval = 10  # seconds
        self.capture_counter = 0
        
        # Data collection for 10-second prediction
        self.prediction_mode = False
        self.data_collection_active = False
        self.collected_data = deque(maxlen=10)  # Store 10 seconds of data
        self.collection_start_time = None
        
        # Initialize model_type as fallback first
        self.model_type = 'fallback'
        
        # Detection data structure
        self.detection_data = {
            'no_of_face': 0,
            'face_x': 0,
            'face_y': 0,
            'face_w': 0,
            'face_h': 0,
            'face_con': 0.0,
            'no_of_hands': 0,
            'pose': 'Forward',
            'pose_x': 0,
            'pose_y': 0,
            'phone': 0,
            'phone_x': 0,
            'phone_y': 0,
            'phone_w': 0,
            'phone_h': 0,
            'phone_con': 0.0
        }
        
        # Initialize ML model
        self.attention_model = AttentionDetectionModel()
        
        # Initialize detection models
        self.init_detectors()
        
        # Create output directory
        if not os.path.exists('detection_output'):
            os.makedirs('detection_output')
    
    def init_detectors(self):
        """Initialize pre-trained detection models with error handling"""
        print("🔧 Initializing detection models...")
        
        # Initialize flags
        self.face_detector_available = False
        self.hands_detector_available = False
        self.phone_detector_available = False
        
        try:
            # Face detection using Haar Cascade (most reliable)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.face_cascade.empty():
                    self.face_detector_available = True
                    print("✅ Face detector (Haar Cascade) - OK")
                else:
                    print("❌ Face detector (Haar Cascade) - Failed to load")
            else:
                print("❌ Face detector - Haar cascade file not found")
                
        except Exception as e:
            print(f"❌ Face detector error: {e}")
        
        # Try YOLOv8 Pose for hand detection
        try:
            from ultralytics import YOLO
            
            # Load YOLOv8 pose model for hand/keypoint detection
            self.yolo_pose = YOLO('yolov8n-pose.pt')  # Nano pose model
            self.hands_detector_available = True
            self.hand_detection_method = 'yolov8_pose'
            print("✅ Hand detector (YOLOv8 Pose) - OK")
            
        except ImportError:
            print("❌ YOLOv8 not available - install with: pip install ultralytics")
            self.hands_detector_available = False
            self.hand_detection_method = 'fallback'
        except Exception as e:
            print(f"❌ Hand detector (YOLOv8 Pose) error: {e}")
            print("📝 Trying fallback hand detection...")
            self.hands_detector_available = False
            self.hand_detection_method = 'fallback'
            
        # Load phone detection model
        self.load_phone_detector()
        
        # Summary
        working_models = sum([
            self.face_detector_available,
            self.hands_detector_available, 
            self.phone_detector_available
        ])
        
        print(f"\n📊 Detection Summary: {working_models}/3 models loaded")
        print(f"🤲 Hand detection method: {self.hand_detection_method}")
        
        if working_models == 0:
            print("⚠️ WARNING: No detection models loaded successfully!")
            print("🔧 System will use basic fallback methods")
        else:
            print("✅ Detection system ready!")
    
    def load_phone_detector(self):
        """Load phone detection model with fallbacks"""
        try:
            # Try YOLOv8 first (best performance)
            try:
                from ultralytics import YOLO
                # Use the same YOLO model for both hands and phones if possible
                if hasattr(self, 'yolo_pose'):
                    # Try to use a general YOLOv8 model for phone detection
                    self.yolo_model = YOLO('yolov8n.pt')  # General object detection model
                else:
                    self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
                    
                self.model_type = 'yolov8'
                self.phone_detector_available = True
                print("✅ Phone detector (YOLOv8n) - OK")
                return
            except ImportError:
                pass
            except Exception as e:
                print(f"📥 YOLOv8 loading failed: {e}")
            
            # Final fallback
            print("📝 Using geometric fallback for phone detection")
            self.model_type = 'fallback'
            self.phone_detector_available = True  # Fallback is always available
            
        except Exception as e:
            print(f"❌ Phone detector initialization error: {e}")
            self.model_type = 'fallback'
            self.phone_detector_available = True
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            print("🎥 Camera started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        self.auto_capture = False
        self.data_collection_active = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("📷 Camera stopped")
    
    def detect_faces(self, frame, gray):
        """Detect faces using Haar Cascade"""
        if not self.face_detector_available:
            # Reset face data
            self.detection_data.update({
                'no_of_face': 0, 'face_x': 0, 'face_y': 0, 
                'face_w': 0, 'face_h': 0, 'face_con': 0.0,
                'pose': 'Forward', 'pose_x': 0, 'pose_y': 0
            })
            return
        
        try:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            self.detection_data['no_of_face'] = len(faces)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                self.detection_data['face_x'] = int(x + w//2)  # Center X
                self.detection_data['face_y'] = int(y + h//2)  # Center Y
                self.detection_data['face_w'] = int(w)
                self.detection_data['face_h'] = int(h)
                self.detection_data['face_con'] = 0.85  # Haar cascade doesn't provide confidence
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Face (0.85)', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Estimate pose based on face position
                self.estimate_pose(x, y, w, h, frame.shape)
                
            else:
                # No face detected
                self.detection_data.update({
                    'face_x': 0, 'face_y': 0, 'face_w': 0, 'face_h': 0, 'face_con': 0.0,
                    'pose': 'Forward', 'pose_x': 0, 'pose_y': 0
                })
                
        except Exception as e:
            print(f"Face detection error: {e}")
            self.detection_data['no_of_face'] = 0
    
    def estimate_pose(self, face_x, face_y, face_w, face_h, frame_shape):
        """Estimate face pose based on position"""
        center_x = face_x + face_w // 2
        center_y = face_y + face_h // 2
        
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        # Calculate pose angles (simplified)
        pose_x = int(((center_x - frame_center_x) / frame_center_x) * 45)
        pose_y = int(((center_y - frame_center_y) / frame_center_y) * 30)
        
        self.detection_data['pose_x'] = pose_x
        self.detection_data['pose_y'] = pose_y
        
        # Determine pose direction
        if abs(pose_x) < 15 and abs(pose_y) < 10:
            self.detection_data['pose'] = 'Forward'
        elif pose_x > 20:
            self.detection_data['pose'] = 'Right'
        elif pose_x < -20:
            self.detection_data['pose'] = 'Left'
        elif pose_y > 15:
            self.detection_data['pose'] = 'Down'
        else:
            self.detection_data['pose'] = 'Forward'
    
    def detect_hands(self, frame):
        """Detect hands using YOLOv8 Pose or fallback method"""
        if not self.hands_detector_available:
            self.detect_hands_fallback(frame)
            return
        
        if self.hand_detection_method == 'yolov8_pose':
            self.detect_hands_yolov8_pose(frame)
        else:
            self.detect_hands_fallback(frame)
    
    def detect_hands_yolov8_pose(self, frame):
        """Detect hands using YOLOv8 pose estimation"""
        try:
            # Run YOLOv8 pose detection
            results = self.yolo_pose(frame, verbose=False)
            
            hand_count = 0
            detected_hands = []
            
            for result in results:
                # Get pose keypoints
                if result.keypoints is not None and len(result.keypoints.xy) > 0:
                    # Get keypoints for the first person (most confident)
                    keypoints = result.keypoints.xy[0]  # Shape: [17, 2] for 17 COCO keypoints
                    confidence = result.keypoints.conf[0] if result.keypoints.conf is not None else None
                    
                    h, w, _ = frame.shape
                    
                    # Check left wrist (index 9)
                    if len(keypoints) > 9:
                        left_wrist = keypoints[9]
                        left_wrist_conf = confidence[9] if confidence is not None else 0.5
                        
                        # Check if wrist is detected with good confidence
                        if (not torch.isnan(left_wrist).any() and 
                            left_wrist[0] > 0 and left_wrist[1] > 0 and
                            left_wrist_conf > 0.3):  # Lower threshold for wrist detection
                            
                            x, y = int(left_wrist[0]), int(left_wrist[1])
                            
                            # Validate wrist is within frame
                            if 0 < x < w and 0 < y < h:
                                hand_count += 1
                                detected_hands.append({
                                    'type': 'Left',
                                    'x': x,
                                    'y': y,
                                    'confidence': float(left_wrist_conf)
                                })
                                
                                # Draw left wrist/hand detection
                                cv2.circle(frame, (x, y), 12, (255, 0, 0), 3)
                                cv2.circle(frame, (x, y), 25, (255, 0, 0), 2)
                                cv2.putText(frame, f'L Hand ({left_wrist_conf:.2f})', 
                                           (x-40, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Check right wrist (index 10)
                    if len(keypoints) > 10:
                        right_wrist = keypoints[10]
                        right_wrist_conf = confidence[10] if confidence is not None else 0.5
                        
                        # Check if wrist is detected with good confidence
                        if (not torch.isnan(right_wrist).any() and 
                            right_wrist[0] > 0 and right_wrist[1] > 0 and
                            right_wrist_conf > 0.3):  # Lower threshold for wrist detection
                            
                            x, y = int(right_wrist[0]), int(right_wrist[1])
                            
                            # Validate wrist is within frame
                            if 0 < x < w and 0 < y < h:
                                hand_count += 1
                                detected_hands.append({
                                    'type': 'Right',
                                    'x': x,
                                    'y': y,
                                    'confidence': float(right_wrist_conf)
                                })
                                
                                # Draw right wrist/hand detection
                                cv2.circle(frame, (x, y), 12, (0, 0, 255), 3)
                                cv2.circle(frame, (x, y), 25, (0, 0, 255), 2)
                                cv2.putText(frame, f'R Hand ({right_wrist_conf:.2f})', 
                                           (x-40, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    break  # Only process first person
            
            self.detection_data['no_of_hands'] = min(hand_count, 10)  # Reasonable max
            
        except Exception as e:
            print(f"YOLOv8 hand detection error: {e}")
            # Fallback to alternative method
            self.detect_hands_fallback(frame)
    
    def detect_hands_fallback(self, frame):
        """Fallback hand detection using skin color and contours"""
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Additional skin range
            lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            # Combine masks
            skin_mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hand_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (hand-like size)
                if 1500 < area < 25000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Hand-like criteria
                    if 0.7 < aspect_ratio < 2.5:
                        # Additional validation using convex hull
                        hull = cv2.convexHull(contour, returnPoints=False)
                        if len(hull) > 3:
                            defects = cv2.convexityDefects(contour, hull)
                            
                            if defects is not None and len(defects) > 1:
                                hand_count += 1
                                
                                # Draw hand detection
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                                cv2.putText(frame, f'Hand {hand_count} (Fallback)', 
                                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                
                                # Draw contour
                                cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
            
            self.detection_data['no_of_hands'] = min(hand_count, 10)
            
        except Exception as e:
            print(f"Fallback hand detection error: {e}")
            self.detection_data['no_of_hands'] = 0
    
    def detect_phone(self, frame):
        """Detect phone using available method"""
        if not self.phone_detector_available:
            self.detection_data.update({
                'phone': 0, 'phone_x': 0, 'phone_y': 0,
                'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
            })
            return
        
        try:
            if self.model_type == 'yolov8':
                self.detect_phone_yolov8(frame)
            else:
                self.detect_phone_fallback(frame)
                
        except Exception as e:
            print(f"Phone detection error: {e}")
            self.detect_phone_fallback(frame)
    
    def detect_phone_yolov8(self, frame):
        """Detect phone using YOLOv8"""
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a cell phone (class 67 in COCO)
                        if class_id == 67 and confidence > 0.6:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            # Update detection data
                            self.detection_data.update({
                                'phone': 1,
                                'phone_x': x + w//2,  # Center X
                                'phone_y': y + h//2,  # Center Y
                                'phone_w': w,
                                'phone_h': h,
                                'phone_con': round(confidence, 2)
                            })
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, f'Phone ({confidence:.2f})', (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            return
            
            # No phone detected
            self.detection_data.update({
                'phone': 0, 'phone_x': 0, 'phone_y': 0,
                'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
            })
            
        except Exception as e:
            print(f"YOLOv8 error: {e}")
            self.detect_phone_fallback(frame)
    
    def detect_phone_fallback(self, frame):
        """Fallback phone detection using geometric analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_phone = None
            best_score = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Phone-like criteria
                if (area > 1500 and area < 20000 and 
                    1.5 < h/w < 3.5 and  # Phone aspect ratio
                    40 < w < 250 and 80 < h < 400):
                    
                    # Calculate score based on geometric properties
                    aspect_ratio = h / w
                    rect_area = w * h
                    extent = area / rect_area if rect_area > 0 else 0
                    
                    score = (aspect_ratio - 1.5) * 0.3 + extent * 0.5 + (area / 10000) * 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_phone = (x, y, w, h, min(0.75, 0.4 + score))
            
            if best_phone and best_score > 0.3:
                x, y, w, h, confidence = best_phone
                
                self.detection_data.update({
                    'phone': 1,
                    'phone_x': x + w//2,
                    'phone_y': y + h//2,
                    'phone_w': w,
                    'phone_h': h,
                    'phone_con': round(confidence, 2)
                })
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f'Phone ({confidence:.2f})', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                self.detection_data.update({
                    'phone': 0, 'phone_x': 0, 'phone_y': 0,
                    'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
                })
                
        except Exception as e:
            print(f"Fallback detection error: {e}")
            self.detection_data.update({
                'phone': 0, 'phone_x': 0, 'phone_y': 0,
                'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
            })
    
    def start_prediction_mode(self):
        """Start 10-second data collection for attention prediction"""
        if self.data_collection_active:
            print("⚠️ Data collection already in progress!")
            return
        
        print("🎯 Starting 10-second attention prediction...")
        print("📊 Collecting data every second...")
        
        self.data_collection_active = True
        self.collected_data.clear()
        self.collection_start_time = time.time()
        
        # Start collection thread
        collection_thread = threading.Thread(target=self.data_collection_worker)
        collection_thread.daemon = True
        collection_thread.start()
    
    def data_collection_worker(self):
        """Worker thread for collecting data every second for 10 seconds"""
        start_time = time.time()
        
        for second in range(10):
            if not self.data_collection_active:
                break
            
            # Wait for next second
            target_time = start_time + (second + 1)
            current_time = time.time()
            
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Collect current detection data
            data_snapshot = self.detection_data.copy()
            data_snapshot['timestamp'] = time.time()
            self.collected_data.append(data_snapshot)
            
            print(f"📊 Collected data point {second + 1}/10")
        
        # After 10 seconds, process and predict
        if self.data_collection_active:
            self.process_collected_data()
        
        self.data_collection_active = False
    
    def process_collected_data(self):
        """Process collected data and make attention prediction"""
        if len(self.collected_data) == 0:
            print("❌ No data collected!")
            return
        
        print(f"\n📊 Processing {len(self.collected_data)} data points...")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(list(self.collected_data))
        
        # Calculate averages for numerical features
        numerical_features = [
            'no_of_face', 'face_x', 'face_y', 'face_w', 'face_h', 'face_con',
            'no_of_hands', 'pose_x', 'pose_y', 'phone', 'phone_x', 
            'phone_y', 'phone_w', 'phone_h', 'phone_con'
        ]
        
        averaged_data = {}
        
        for feature in numerical_features:
            if feature in df.columns:
                averaged_data[feature] = df[feature].mean()
            else:
                averaged_data[feature] = 0
        
        # For categorical features like 'pose', use mode (most frequent)
        pose_values = df['pose'].tolist()
        if pose_values:
            try:
                # Get the most frequent pose
                pose_modes = multimode(pose_values)
                averaged_data['pose'] = pose_modes[0]  # Take first mode if multiple
            except:
                averaged_data['pose'] = 'Forward'  # Default
        else:
            averaged_data['pose'] = 'Forward'
        
        print("📈 Averaged detection data:")
        for key, value in averaged_data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Make prediction using ML model
        if self.attention_model.is_trained:
            prediction, probability = self.attention_model.predict_attention(averaged_data)
            
            if prediction is not None:
                attention_status = "ATTENTIVE" if prediction == 1 else "NOT ATTENTIVE"
                confidence = probability if prediction == 1 else (1 - probability)
                
                print(f"\n🎯 ATTENTION PREDICTION:")
                print(f"📊 Status: {attention_status}")
                print(f"📊 Confidence: {confidence:.1%}")
                print(f"📊 Raw Prediction: {prediction}")
                print(f"📊 Probability (Attentive): {probability:.3f}")
                
                # Save prediction result
                self.save_prediction_result(averaged_data, prediction, probability)
                
            else:
                print("❌ Failed to make prediction!")
        else:
            print("❌ ML model not trained!")
    
    def save_prediction_result(self, averaged_data, prediction, probability):
        """Save prediction result to file"""
        timestamp = datetime.now().isoformat()
        
        result_data = {
            'timestamp': timestamp,
            'averaged_features': averaged_data,
            'prediction': int(prediction),
            'probability_attentive': float(probability),
            'attention_status': "ATTENTIVE" if prediction == 1 else "NOT ATTENTIVE",
            'data_points_collected': len(self.collected_data)
        }
        
        filename = f"detection_output/attention_prediction_{timestamp.replace(':', '-').replace('.', '_')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"💾 Prediction saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
    
    def save_detection_data(self):
        """Save current detection data to JSON file"""
        timestamp = datetime.now().isoformat()
        
        capture_data = {
            'timestamp': timestamp,
            'frame_data': self.detection_data.copy()
        }
        
        filename = f"detection_output/detection_data_{timestamp.replace(':', '-').replace('.', '_')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(capture_data, f, indent=2)
            
            self.capture_counter += 1
            print(f"📁 Data saved: {filename} (Total: {self.capture_counter})")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def print_detection_summary(self):
        """Print current detection data in the required format"""
        data = self.detection_data
        print(f"\n📊 Current Detection Data:")
        print(f"no_of_face: {data['no_of_face']}")
        print(f"face_x: {data['face_x']}")
        print(f"face_y: {data['face_y']}")
        print(f"face_w: {data['face_w']}")
        print(f"face_h: {data['face_h']}")
        print(f"face_con: {data['face_con']}")
        print(f"no_of_hands: {data['no_of_hands']}")
        print(f"pose: {data['pose']}")
        print(f"pose_x: {data['pose_x']}")
        print(f"pose_y: {data['pose_y']}")
        print(f"phone: {data['phone']}")
        print(f"phone_x: {data['phone_x']}")
        print(f"phone_y: {data['phone_y']}")
        print(f"phone_w: {data['phone_w']}")
        print(f"phone_h: {data['phone_h']}")
        print(f"phone_con: {data['phone_con']}")
    
    def run(self):
        """Main detection loop"""
        if not self.start_camera():
            return
        
        print("\n🎮 CONTROLS:")
        print("  SPACE - Save current data")
        print("  'p' - Print detection data")
        print("  'r' - Start 10-second attention prediction")
        print("  's' - Stop data collection")
        print("  'h' - Toggle hand detection method")
        print("  'q' - Quit")
        print("\n▶️ Detection started...")
        
        fps_counter = 0
        fps_timer = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Perform detections
            self.detect_faces(frame, gray)
            self.detect_hands(frame)
            self.detect_phone(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            else:
                fps = fps_counter
            
            # Display info on frame
            status_info = [
                f"FPS: {fps}",
                f"Face: {'✅' if self.face_detector_available else '❌'} ({self.detection_data['no_of_face']})",
                f"Hands: {'✅' if self.hands_detector_available else '❌'} ({self.detection_data['no_of_hands']}) [{self.hand_detection_method}]",
                f"Phone: {'✅' if self.phone_detector_available else '❌'} ({'Yes' if self.detection_data['phone'] else 'No'})",
                f"Pose: {self.detection_data['pose']}",
                f"ML Model: {'✅' if self.attention_model.is_trained else '❌'}",
                f"Collecting: {'YES' if self.data_collection_active else 'NO'}",
                f"Saves: {self.capture_counter}",
                f"Model: {self.model_type.upper()}"
            ]
            
            # Show collection progress if active
            if self.data_collection_active:
                elapsed = time.time() - self.collection_start_time
                remaining = max(0, 10 - elapsed)
                progress = f"Collection: {len(self.collected_data)}/10 ({remaining:.1f}s left)"
                status_info.append(progress)
                
                # Draw progress bar
                bar_width = 300
                bar_height = 20
                bar_x = 10
                bar_y = frame.shape[0] - 40
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
                
                # Progress
                progress_width = int((len(self.collected_data) / 10) * bar_width)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                
                # Text
                cv2.putText(frame, f"Collecting: {len(self.collected_data)}/10", 
                           (bar_x + 5, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, text in enumerate(status_info):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('AI Attention Detection System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.save_detection_data()
                self.print_detection_summary()
            elif key == ord('p'):
                self.print_detection_summary()
            elif key == ord('r'):
                if not self.data_collection_active:
                    self.start_prediction_mode()
                else:
                    print("⚠️ Data collection already in progress!")
            elif key == ord('s'):
                if self.data_collection_active:
                    self.data_collection_active = False
                    print("⏹️ Data collection stopped")
                else:
                    print("⚠️ No data collection in progress")
            elif key == ord('h'):
                # Toggle hand detection method
                if self.hand_detection_method == 'yolov8_pose':
                    self.hand_detection_method = 'fallback'
                    print("🔄 Switched to fallback hand detection")
                else:
                    if self.hands_detector_available:
                        self.hand_detection_method = 'yolov8_pose'
                        print("🔄 Switched to YOLOv8 pose hand detection")
                    else:
                        print("⚠️ YOLOv8 hand detection not available")
        
        self.stop_camera()
    
    def get_detection_data(self):
        """Get current detection data"""
        return self.detection_data.copy()

def check_dataset_structure(dataset_path="attention_detection_dataset_v1.csv"):
    """Helper function to check dataset structure"""
    try:
        print("🔍 Checking dataset structure...")
        df = pd.read_csv(dataset_path)
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📋 Data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype} (unique values: {df[col].nunique()})")
        
        print(f"\n📋 First 3 rows:")
        print(df.head(3).to_string())
        
        if 'label' in df.columns:
            print(f"\n📊 Label distribution:")
            print(df['label'].value_counts())
        else:
            print("❌ No 'label' column found!")
            
        return True
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def main():
    print("🤖 AI Attention Detection System")
    print("=" * 60)
    print(f"🐍 Python version: {sys.version}")
    print("=" * 60)
    
    # First, check dataset structure
    print("\n🔍 Dataset Structure Check:")
    check_dataset_structure()
    print("=" * 60)
    
    try:
        detector = CameraDetectionSystem()
        
        # Load and train the ML model
        print("\n🧠 Loading Machine Learning Model...")
        if detector.attention_model.load_and_train_model():
            print("✅ ML model ready for predictions!")
        else:
            print("❌ ML model failed to load. Predictions will not be available.")
            print("🔧 Please check the dataset structure above and ensure:")
            print("   1. File 'attention_detection_dataset_v1.csv' exists")
            print("   2. File has a 'label' column")
            print("   3. File has feature columns (numeric data)")
        
        # Show initialization results
        working_models = sum([
            detector.face_detector_available,
            detector.hands_detector_available,
            detector.phone_detector_available
        ])
        
        print(f"\n📱 Detection Summary:")
        print(f"📱 Working models: {working_models}/3")
        print(f"📱 Phone detector: {detector.model_type.upper()}")
        print(f"🤲 Hand detector: {detector.hand_detection_method.upper()}")
        print(f"🧠 ML model: {'READY' if detector.attention_model.is_trained else 'NOT AVAILABLE'}")
        print("=" * 60)
        
        if working_models > 0:
            print("💡 Key Features:")
            print("   📊 Real-time face, hand, and phone detection")
            print("   🎯 10-second attention prediction with ML")
            print("   📈 Automatic feature averaging and mode calculation")
            print("   💾 JSON data export for analysis")
            print("\n🎮 Instructions:")
            print("   1. Press 'r' to start 10-second attention prediction")
            print("   2. System will collect data every second for 10 seconds")
            print("   3. After collection, ML model predicts attention level")
            print("   4. Results are saved automatically to JSON files")
            print("=" * 60)
            detector.run()
        else:
            print("❌ No detection models available. Cannot proceed.")
            print("\n🔧 Troubleshooting:")
            print("1. Install YOLOv8: pip install ultralytics")
            print("2. Install PyTorch: pip install torch")
            print("3. Install OpenCV: pip install opencv-python")
            print("4. Install scikit-learn: pip install scikit-learn")
            print("5. Check camera permissions")
            print("6. Ensure dataset file exists: attention_detection_dataset_v1.csv")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Attention detection system stopped")

if __name__ == "__main__":
    main()