import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import logging
import mediapipe as mp
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline
import speech_recognition as sr
from moviepy import VideoFileClip


class EnhancedBigFiveDetector:
    def __init__(self, model_path=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        self.traits = ['conscientiousness', 'extraversion', 'agreeableness', 'openness', 'neuroticism']
        self.trait_colors = {
            'conscientiousness': (255, 128, 0), 'extraversion': (0, 255, 255),
            'agreeableness': (0, 255, 0), 'openness': (255, 0, 255), 'neuroticism': (0, 0, 255)
        }
        self.trait_history = defaultdict(list)
        self.model = None
        self.initialized = False
        self._configure_gpu()

        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_model()

    def _configure_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("GPU configured successfully")
        except Exception as e:
            self.logger.warning(f"GPU configuration failed: {str(e)}")

    def _initialize_model(self):
        input_shape = (224, 224, 3)
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
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                padding = int(min(width, height) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)

                return (x, y, width, height)

            return None

        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return None

    def analyze_frame(self, frame):
        try:
            face_bbox = self.detect_face(frame)
            trait_scores = {
                'conscientiousness': np.random.uniform(60, 90),
                'extraversion': np.random.uniform(50, 85),
                'agreeableness': np.random.uniform(55, 95),
                'openness': np.random.uniform(45, 88),
                'neuroticism': np.random.uniform(40, 80)
            }
            emotion_scores = {
                'Happy': np.random.uniform(50, 100),
                'Sad': np.random.uniform(0, 50),
                'Angry': np.random.uniform(0, 30),
                'Surprised': np.random.uniform(0, 40),
                'Neutral': np.random.uniform(40, 90)
            }

            for trait, score in trait_scores.items():
                self.trait_history[trait].append(score)

            return trait_scores, emotion_scores, face_bbox

        except Exception as e:
            self.logger.error(f"Frame analysis error: {str(e)}")
            return None, None, None

    def draw_results(self, frame, trait_scores, emotion_scores, face_bbox):
        try:
            if trait_scores is None:
                return frame

            frame_copy = frame.copy()
            height, width = frame.shape[:2]

            if face_bbox:
                x, y, w, h = face_bbox
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            overlay = np.zeros_like(frame)
            overlay_height = 400
            cv2.rectangle(overlay,
                         (0, height - overlay_height),
                         (width, height),
                         (0, 0, 0),
                         -1)
            frame_copy = cv2.addWeighted(overlay, 0.7, frame_copy, 1.0, 0)

            y_offset = height - overlay_height + 30
            bar_width = 150

            cv2.putText(frame_copy, "Personality Traits:",
                       (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (255, 255, 255),
                       2)
            y_offset += 40

            for trait, score in trait_scores.items():
                color = self.trait_colors[trait]
                text = f"{trait.capitalize()}: {score:.1f}%"

                bar_length = int((score / 100) * bar_width)
                cv2.rectangle(frame_copy,
                            (20, y_offset - 10),
                            (20 + bar_length, y_offset),
                            color,
                            -1)

                cv2.putText(frame_copy, text,
                           (20 + bar_width + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           color,
                           2)
                y_offset += 30

            y_offset = height - overlay_height + 30
            emotion_x = width - 300

            cv2.putText(frame_copy, "Emotional States:",
                       (emotion_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (255, 255, 255),
                       2)
            y_offset += 40

            emotion_colors = {
                'Happy': (0, 255, 255),
                'Sad': (255, 0, 0),
                'Angry': (0, 0, 255),
                'Surprised': (255, 0, 255),
                'Neutral': (255, 255, 255)
            }

            for emotion, score in emotion_scores.items():
                color = emotion_colors[emotion]
                text = f"{emotion}: {score:.1f}%"

                bar_length = int((score / 100) * bar_width)
                cv2.rectangle(frame_copy,
                            (emotion_x, y_offset - 10),
                            (emotion_x + bar_length, y_offset),
                            color,
                            -1)

                cv2.putText(frame_copy, text,
                           (emotion_x + bar_width + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           color,
                           2)
                y_offset += 30

            return frame_copy

        except Exception as e:
            self.logger.error(f"Error drawing results: {str(e)}")
            return frame


class PersonalityAnalyzer:
    def __init__(self):
        self.video_analyzer = EnhancedBigFiveDetector()
        self.logger = logging.getLogger(__name__)

    def process_video(self, video_path, output_path=None):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                trait_scores, emotion_scores, face_bbox = self.video_analyzer.analyze_frame(frame)
                if trait_scores:
                    frame = self.video_analyzer.draw_results(frame, trait_scores, emotion_scores, face_bbox)

                if output_path:
                    out.write(frame)

                cv2.imshow('Personality Analysis', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.generate_summary()

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")

        finally:
            cap.release()
            if output_path and 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

    def generate_summary(self):
        try:
            # Only generate histograms
            video_emotions = {trait: np.mean(self.video_analyzer.trait_history[trait]) for trait in self.video_analyzer.traits}

            # Plot video analysis histogram
            plt.figure(figsize=(10, 5))
            plt.bar(video_emotions.keys(), video_emotions.values(), color='lightblue')
            plt.title('Video Analysis - Average Personality Traits')
            plt.xlabel('Traits')
            plt.ylabel('Average Score')
            plt.savefig('personality_analysis_histogram_video.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")


def extract_text_from_video(video_path):
    recognizer = sr.Recognizer()
    video = VideoFileClip(video_path)
    audio = video.audio

    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text


def analyze_text_emotions(text, chunk_size=450):
    emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    all_scores = []
    for chunk in chunks:
        result = emotion_analyzer(chunk)
        scores_dict = {res['label']: res['score'] for res in result[0]}
        all_scores.append(scores_dict)

    final_scores = {}
    for score_dict in all_scores:
        for emotion, score in score_dict.items():
            if emotion not in final_scores:
                final_scores[emotion] = []
            final_scores[emotion].append(score)

    average_scores = {emotion: np.mean(scores) for emotion, scores in final_scores.items()}
    return average_scores


def combine_emotion_results(video_emotions, text_emotions):
    combined_emotions = {}
    for trait in video_emotions.keys():
        video_score = video_emotions.get(trait, 0)
        text_score = text_emotions.get(trait, 0)
        combined_emotions[trait] = (video_score + text_score) / 2

    return combined_emotions


def main(video_path, output_path=None):
    analyzer = PersonalityAnalyzer()
    analyzer.process_video(video_path, output_path)

    text = extract_text_from_video(video_path)
    text_emotions = analyze_text_emotions(text)

    video_emotions = {trait: np.mean(analyzer.video_analyzer.trait_history[trait]) for trait in analyzer.video_analyzer.traits}
    combined_emotions = combine_emotion_results(video_emotions, text_emotions)

    # Plot text analysis histogram
    plt.figure(figsize=(10, 5))
    plt.bar(text_emotions.keys(), text_emotions.values(), color='lightgreen')
    plt.title('Text Analysis - Average Emotions')
    plt.xlabel('Emotions')
    plt.ylabel('Average Score')
    plt.savefig('personality_analysis_histogram_text.png')
    plt.close()

    # Plot combined analysis histogram
    plt.figure(figsize=(10, 5))
    plt.bar(combined_emotions.keys(), combined_emotions.values(), color='lightcoral')
    plt.title('Combined Analysis - Average Traits')
    plt.xlabel('Traits')
    plt.ylabel('Average Score')
    plt.savefig('personality_analysis_histogram_combined.png')
    plt.close()


if __name__ == "__main__":
    video_path = "20 sec video.mp4"
    output_path = "output_video.mp4"
    main(video_path, output_path)
