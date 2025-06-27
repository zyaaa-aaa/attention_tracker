import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime
import os
import sys

class CameraDetectionSystem:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.auto_capture = False
        self.capture_interval = 10  # seconds
        self.capture_counter = 0
        
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
        
        # Try MediaPipe Hands with error handling
        try:
            import mediapipe as mp
            
            # Try with minimal configuration first
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=10,
                model_complexity=0,  # Use lightest model
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands_detector_available = True
            print("✅ Hand detector (MediaPipe) - OK")
            
        except Exception as e:
            print(f"❌ Hand detector (MediaPipe) error: {e}")
            print("📝 Try: pip install --upgrade mediapipe")
            self.hands_detector_available = False
            
        # Load phone detection model
        self.load_phone_detector()
        
        # Summary
        working_models = sum([
            self.face_detector_available,
            self.hands_detector_available, 
            self.phone_detector_available
        ])
        
        print(f"\n📊 Detection Summary: {working_models}/3 models loaded")
        
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
                self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
                self.model_type = 'yolov8'
                self.phone_detector_available = True
                print("✅ Phone detector (YOLOv8n) - OK")
                return
            except ImportError:
                pass
            except Exception as e:
                print(f"📥 YOLOv8 loading failed: {e}")
            
            # Try YOLOv5
            try:
                import torch
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.conf = 0.6
                self.yolo_model.iou = 0.4
                self.model_type = 'yolov5'
                self.phone_detector_available = True
                print("✅ Phone detector (YOLOv5s) - OK")
                return
            except Exception as e:
                print(f"📥 YOLOv5 loading failed: {e}")
            
            # Fallback to OpenCV DNN
            try:
                self.load_opencv_dnn()
                return
            except Exception as e:
                print(f"📥 OpenCV DNN loading failed: {e}")
            
            # Final fallback
            print("📝 Using geometric fallback for phone detection")
            self.model_type = 'fallback'
            self.phone_detector_available = True  # Fallback is always available
            
        except Exception as e:
            print(f"❌ Phone detector initialization error: {e}")
            self.model_type = 'fallback'
            self.phone_detector_available = True
    
    def load_opencv_dnn(self):
        """Load OpenCV DNN model"""
        # Try to download a lightweight ONNX model
        model_path = "yolov5s.onnx"
        
        if not os.path.exists(model_path):
            print("📥 Downloading YOLOv5 ONNX model...")
            try:
                import urllib.request
                url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"
                urllib.request.urlretrieve(url, model_path)
                print("✅ Model downloaded successfully")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise e
        
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.model_type = 'opencv_onnx'
        self.phone_detector_available = True
        print("✅ Phone detector (OpenCV ONNX) - OK")
    
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
        """Detect hands using MediaPipe or fallback"""
        if not self.hands_detector_available:
            self.detection_data['no_of_hands'] = 0
            return
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Count detected hands with validation
            validated_hands = 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Get confidence score
                    confidence = handedness.classification[0].score
                    
                    # Only count hands with high confidence
                    if confidence > 0.7:
                        h, w, _ = frame.shape
                        
                        # Calculate hand bounding box
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            landmarks.append((x, y))
                        
                        x_coords = [point[0] for point in landmarks]
                        y_coords = [point[1] for point in landmarks]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # Validate hand size
                        hand_area = (x_max - x_min) * (y_max - y_min)
                        
                        if 1000 < hand_area < 30000:  # Reasonable hand size
                            validated_hands += 1
                            
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x_min-5, y_min-5), (x_max+5, y_max+5), (255, 0, 0), 2)
                            
                            # Hand type
                            hand_type = handedness.classification[0].label
                            cv2.putText(frame, f'{hand_type} ({confidence:.2f})', 
                                       (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            self.detection_data['no_of_hands'] = min(validated_hands, 10)  # Max 2 hands
            
        except Exception as e:
            print(f"Hand detection error: {e}")
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
            elif self.model_type == 'yolov5':
                self.detect_phone_yolov5(frame)
            elif self.model_type == 'opencv_onnx':
                self.detect_phone_onnx(frame)
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
    
    def detect_phone_yolov5(self, frame):
        """Detect phone using YOLOv5"""
        try:
            results = self.yolo_model(frame)
            
            # Parse results
            for *box, conf, cls in results.xyxy[0].tolist():
                class_id = int(cls)
                confidence = float(conf)
                
                if class_id == 67 and confidence > 0.6:  # cell phone
                    x1, y1, x2, y2 = map(int, box)
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    
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
                    return
            
            # No phone detected
            self.detection_data.update({
                'phone': 0, 'phone_x': 0, 'phone_y': 0,
                'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
            })
            
        except Exception as e:
            print(f"YOLOv5 error: {e}")
            self.detect_phone_fallback(frame)
    
    def detect_phone_onnx(self, frame):
        """Detect phone using ONNX model"""
        try:
            height, width = frame.shape[:2]
            
            # Prepare input
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            # Process YOLOv5 outputs
            for detection in outputs[0][0]:
                confidence = detection[4]
                if confidence > 0.5:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    
                    if class_id == 67 and scores[class_id] > 0.6:  # cell phone
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        final_conf = confidence * scores[class_id]
                        
                        self.detection_data.update({
                            'phone': 1,
                            'phone_x': center_x,
                            'phone_y': center_y,
                            'phone_w': w,
                            'phone_h': h,
                            'phone_con': round(final_conf, 2)
                        })
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, f'Phone ({final_conf:.2f})', (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        return
            
            # No phone detected
            self.detection_data.update({
                'phone': 0, 'phone_x': 0, 'phone_y': 0,
                'phone_w': 0, 'phone_h': 0, 'phone_con': 0.0
            })
            
        except Exception as e:
            print(f"ONNX detection error: {e}")
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
    
    def auto_capture_worker(self):
        """Worker thread for auto capture"""
        while self.auto_capture:
            time.sleep(self.capture_interval)
            if self.auto_capture:
                self.save_detection_data()
    
    def start_auto_capture(self):
        """Start automatic data capture"""
        if not self.auto_capture:
            self.auto_capture = True
            capture_thread = threading.Thread(target=self.auto_capture_worker)
            capture_thread.daemon = True
            capture_thread.start()
            print(f"⏰ Auto capture started! (every {self.capture_interval}s)")
        else:
            print("⚠️ Auto capture already running!")
    
    def stop_auto_capture(self):
        """Stop automatic data capture"""
        self.auto_capture = False
        print("⏸️ Auto capture stopped")
    
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
        print("  'a' - Toggle auto capture")
        print("  'p' - Print detection data")
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
                f"Hands: {'✅' if self.hands_detector_available else '❌'} ({self.detection_data['no_of_hands']})",
                f"Phone: {'✅' if self.phone_detector_available else '❌'} ({'Yes' if self.detection_data['phone'] else 'No'})",
                f"Pose: {self.detection_data['pose']}",
                f"Auto: {'ON' if self.auto_capture else 'OFF'}",
                f"Saves: {self.capture_counter}",
                f"Model: {self.model_type.upper()}"
            ]
            
            for i, text in enumerate(status_info):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('AI Detection System - Fixed Version', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.save_detection_data()
                self.print_detection_summary()
            elif key == ord('a'):
                if self.auto_capture:
                    self.stop_auto_capture()
                else:
                    self.start_auto_capture()
            elif key == ord('p'):
                self.print_detection_summary()
        
        self.stop_camera()
    
    def get_detection_data(self):
        """Get current detection data"""
        return self.detection_data.copy()

def main():
    print("🤖 AI Camera Detection System - Fixed Version")
    print("=" * 60)
    print(f"🐍 Python version: {sys.version}")
    print("=" * 60)
    
    try:
        detector = CameraDetectionSystem()
        
        # Show initialization results
        working_models = sum([
            detector.face_detector_available,
            detector.hands_detector_available,
            detector.phone_detector_available
        ])
        
        print(f"📱 Working models: {working_models}/3")
        print(f"📱 Phone detector: {detector.model_type.upper()}")
        print("=" * 60)
        
        if working_models > 0:
            detector.run()
        else:
            print("❌ No detection models available. Cannot proceed.")
            print("\n🔧 Troubleshooting:")
            print("1. Update MediaPipe: pip install --upgrade mediapipe")
            print("2. Try Python 3.9-3.11: pyenv install 3.11.0")
            print("3. Install missing packages: pip install opencv-python torch ultralytics")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Detection system stopped")

if __name__ == "__main__":
    main()