import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, 
                                   Dense, Dropout, Flatten, LeakyReLU)
from tensorflow.keras.optimizers import Adam
import time
from collections import deque
import logging
import mediapipe as mp

class EnhancedBigFiveDetector:
    def __init__(self, model_path=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close range, 1 for far range
            min_detection_confidence=0.5
        )
        
        # Define Big Five traits with realistic ranges
        self.traits = [
            'conscientiousness',
            'extraversion',      
            'agreeableness',     
            'openness',          
            'neuroticism'        
        ]
        
        self.trait_colors = {
            'conscientiousness': (255, 128, 0),  
            'extraversion': (0, 255, 255),       
            'agreeableness': (0, 255, 0),        
            'openness': (255, 0, 255),           
            'neuroticism': (0, 0, 255)           
        }
        
        # Performance optimizations
        self.fps_buffer = deque(maxlen=30)
        self.frame_buffer = deque(maxlen=5)  # Buffer for smoothing predictions
        self.skip_frames = 2  # Increase skip frames for better performance
        
        # Initialize face detection cache
        self.face_cache = None
        self.face_cache_frames = 0
        self.max_cache_frames = 5
        
        # Enable GPU memory growth and optimize
        self._configure_gpu()
        
        # Initialize model with dynamic input shape
        self.model = None
        self.initialized = False
        
        if model_path:
            self.load_model(model_path)

    def _configure_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit to prevent OOM errors
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
                )
                
                # Enable mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                self.logger.info("GPU configured with optimizations")
        except Exception as e:
            self.logger.warning(f"GPU configuration failed: {str(e)}")

    def _initialize_model(self, input_shape):
        """Initialize model with improved architecture"""
        self.model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2)),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(5, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.initialized = True

    def detect_face(self, frame):
        """Detect face in frame using MediaPipe"""
        try:
            # If we have a recent cached face detection, use it
            if self.face_cache and self.face_cache_frames < self.max_cache_frames:
                self.face_cache_frames += 1
                return self.face_cache

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]  # Get the first face
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = int(min(width, height) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)
                
                # Cache the detection
                self.face_cache = (x, y, width, height)
                self.face_cache_frames = 0
                
                return (x, y, width, height)
            
            return None

        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return None

    def preprocess_frame(self, frame, face_bbox=None):
        """Preprocess frame with face cropping"""
        try:
            if face_bbox is not None:
                x, y, w, h = face_bbox
                frame = frame[y:y+h, x:x+w]
            
            # Resize to standard input size
            frame = cv2.resize(frame, (224, 224))
            
            # Apply preprocessing
            processed = frame.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            return None

    def smooth_predictions(self, new_predictions):
        """Smooth predictions using moving average"""
        self.frame_buffer.append(new_predictions)
        
        # Calculate weighted average of recent predictions
        weights = np.linspace(0.5, 1.0, len(self.frame_buffer))
        weights /= weights.sum()
        
        smoothed = np.zeros(5)
        for i, pred in enumerate(self.frame_buffer):
            smoothed += weights[i] * np.array(list(pred.values()))
            
        return {trait: float(score) 
                for trait, score in zip(self.traits, smoothed)}

    def analyze_frame(self, frame):
        """Analyze frame for personality traits"""
        try:
            if not self.initialized:
                self._initialize_model((224, 224, 3))
            
            # Detect face
            face_bbox = self.detect_face(frame)
            if face_bbox is None:
                return None
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame, face_bbox)
            if processed_frame is None:
                return None
            
            # Get predictions
            predictions = self.model.predict(processed_frame, verbose=0)[0]
            
            # Scale predictions to realistic ranges
            trait_ranges = {
                'conscientiousness': (0.2, 0.9),
                'extraversion': (0.1, 0.8),
                'agreeableness': (0.3, 0.85),
                'openness': (0.15, 0.75),
                'neuroticism': (0.1, 0.7)
            }
            
            trait_scores = {}
            for trait, score, (min_val, max_val) in zip(
                self.traits, predictions, trait_ranges.values()
            ):
                # Scale to realistic range
                scaled_score = min_val + score * (max_val - min_val)
                trait_scores[trait] = float(scaled_score * 100)
            
            # Smooth predictions
            smoothed_scores = self.smooth_predictions(trait_scores)
            
            return smoothed_scores, face_bbox
            
        except Exception as e:
            self.logger.error(f"Frame analysis error: {str(e)}")
            return None, None

    def draw_results(self, frame, trait_scores, face_bbox):
        """Draw trait analysis results and face tracking"""
        try:
            if trait_scores is None:
                return frame

            frame_copy = frame.copy()
            height, width = frame.shape[:2]

            # Draw face bounding box if detected
            if face_bbox:
                x, y, w, h = face_bbox
                cv2.rectangle(frame_copy, 
                            (x, y), 
                            (x + w, y + h), 
                            (0, 255, 0), 
                            2)

            # Create semi-transparent overlay
            overlay = np.zeros_like(frame)
            overlay_height = 200
            cv2.rectangle(overlay, 
                         (0, height - overlay_height), 
                         (width, height), 
                         (0, 0, 0), 
                         -1)
            frame_copy = cv2.addWeighted(overlay, 0.7, frame_copy, 1.0, 0)

            # Draw trait scores with bars
            y_offset = height - overlay_height + 30
            bar_width = 150
            for trait, score in trait_scores.items():
                color = self.trait_colors.get(trait, (255, 255, 255))
                text = f"{trait.capitalize()}: {score:.1f}%"
                
                # Draw background
                cv2.rectangle(frame_copy, 
                            (20, y_offset - 15),
                            (20 + bar_width + 100, y_offset + 5),
                            (0, 0, 0),
                            -1)
                
                # Draw progress bar
                bar_length = int((score / 100) * bar_width)
                cv2.rectangle(frame_copy,
                            (20, y_offset - 10),
                            (20 + bar_length, y_offset),
                            color,
                            -1)
                
                # Draw text
                cv2.putText(frame_copy, text,
                           (20 + bar_width + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6,
                           color, 
                           2)
                y_offset += 30

            # Draw dominant trait
            dominant_trait = max(trait_scores.items(), key=lambda x: x[1])[0]
            text = f"Dominant Trait: {dominant_trait.capitalize()}"
            cv2.putText(frame_copy, text,
                       (20, y_offset + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8,
                       self.trait_colors[dominant_trait], 
                       2)

            return frame_copy

        except Exception as e:
            self.logger.error(f"Error drawing results: {str(e)}")
            return frame

    def process_video(self, video_path, output_path=None):
        """Process video with full frame analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            cv2.namedWindow('Big Five Personality Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Big Five Personality Analysis', width, height)
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            last_prediction = None
            last_face_bbox = None
            
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Process every nth frame
                if frame_count % self.skip_frames == 0:
                    trait_scores, face_bbox = self.analyze_frame(frame)
                    if trait_scores:
                        last_prediction = trait_scores
                        last_face_bbox = face_bbox

                if last_prediction:
                    frame = self.draw_results(frame, last_prediction, last_face_bbox)

                # Calculate and display FPS
                processing_time = time.time() - start_time
                self.fps_buffer.append(processing_time)
                if self.fps_buffer:
                    fps_value = 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer))
                    cv2.putText(frame, f"FPS: {fps_value:.1f}", 
                              (width - 150, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1, 
                              (0, 255, 0), 
                              2)

                if output_path:
                    out.write(frame)
                
                cv2.imshow('Big Five Personality Analysis', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
        
        finally:
            cap.release()
            if output_path and 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

    def load_model(self, model_path):
        """Load trained model weights"""
        try:
            if self.model is None:
                # Initialize with dummy input shape, will be updated on first frame
                self._initialize_model((224, 224, 3))
            self.model.load_weights(model_path)
            self.logger.info("Model weights loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model weights: {str(e)}")


if __name__ == "__main__":
    detector = EnhancedBigFiveDetector()
    # detector.load_model('path_to_trained_weights.h5')
    video_path = '20 sec video.mp4'
    output_path = 'analyzed_video.mp4'
    detector.process_video(video_path, output_path)
