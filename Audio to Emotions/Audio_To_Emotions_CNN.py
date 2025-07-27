import librosa
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import soundfile as sf
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AudioCNN(nn.Module):
    def __init__(self, num_emotions=7):
        super(AudioCNN, self).__init__()
        
        # CNN layers for processing mel spectrograms
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_emotions)

    def forward(self, x):
        # CNN blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten and feed through fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class AudioEmotionDetector:
    def __init__(self):
        try:
            # Text-based emotion classifier
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            self.recognizer = sr.Recognizer()
            
            # Initialize and load CNN model
            self.cnn_model = AudioCNN()
            # Note: In practice, you would load pre-trained weights here
            # self.cnn_model.load_state_dict(torch.load('emotion_cnn_weights.pth'))
            self.cnn_model.eval()
            
            # Emotion mapping for CNN outputs
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
        except Exception as e:
            raise Exception(f"Failed to initialize dependencies: {e}")

    def extract_audio_features(self, audio_segment, sample_rate):
        """Extract mel spectrogram features for CNN processing"""
        try:
            # Convert to mel spectrogram
            mel_spect = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=sample_rate,
                n_mels=64,
                n_fft=2048,
                hop_length=512
            )
            
            # Convert to log scale
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            
            # Normalize
            mel_spect = (mel_spect - mel_spect.mean()) / mel_spect.std()
            
            # Resize to expected input size (64x64)
            if mel_spect.shape[1] > 64:
                mel_spect = mel_spect[:, :64]
            else:
                pad_width = 64 - mel_spect.shape[1]
                mel_spect = np.pad(mel_spect, ((0, 0), (0, pad_width)), mode='constant')
            
            # Prepare for CNN (add batch and channel dimensions)
            mel_spect = torch.FloatTensor(mel_spect).unsqueeze(0).unsqueeze(0)
            
            return mel_spect
            
        except Exception as e:
            raise Exception(f"Feature extraction error: {e}")

    def analyze_audio_emotion(self, audio_segment, sample_rate):
        """Analyze emotion from audio using CNN"""
        try:
            features = self.extract_audio_features(audio_segment, sample_rate)
            
            with torch.no_grad():
                predictions = self.cnn_model(features)
                
            # Convert predictions to same format as text emotions
            emotions = [
                {
                    'emotion': self.emotion_labels[i],
                    'score': round(float(predictions[0][i]) * 100, 2)
                }
                for i in range(len(self.emotion_labels))
            ]
            
            return emotions
            
        except Exception as e:
            raise Exception(f"Audio emotion analysis error: {e}")

    def segment_audio(self, audio_path, segment_length=10):
        try:
            y, sr = librosa.load(audio_path)
            segment_samples = segment_length * sr
            segments = []
            
            for i in range(0, len(y), segment_samples):
                segment = y[i:i + segment_samples]
                if len(segment) >= sr:  # Only keep segments at least 1 second long
                    segments.append(segment)
            
            return segments, sr
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        except Exception as e:
            raise Exception(f"Error segmenting audio: {e}")

    def transcribe_audio(self, audio_segment, sample_rate):
        try:
            sf.write("temp_segment.wav", audio_segment, sample_rate)
            
            with sr.AudioFile("temp_segment.wav") as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except sr.RequestError:
            raise Exception("Could not connect to Google Speech Recognition service")
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            raise Exception(f"Transcription error: {e}")

    def analyze_text_emotion(self, text):
        if not text:
            return None
        
        try:
            emotions = self.emotion_classifier(text)[0]
            return [
                {
                    'emotion': score['label'],
                    'score': round(score['score'] * 100, 2)
                }
                for score in emotions
            ]
        except Exception as e:
            raise Exception(f"Text emotion analysis error: {e}")

    def analyze_audio(self, audio_path, output_path=None):
        print("Starting audio analysis...")
        
        try:
            segments, sample_rate = self.segment_audio(audio_path)
            results = []
            
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)}")
                
                # Analyze both text and audio emotions
                text = self.transcribe_audio(segment, sample_rate)
                text_emotions = self.analyze_text_emotion(text) if text else None
                audio_emotions = self.analyze_audio_emotion(segment, sample_rate)
                
                result = {
                    'segment': i+1,
                    'timestamp': f"{i*10}-{(i+1)*10}s",
                    'text': text,
                    'text_emotions': text_emotions,
                    'audio_emotions': audio_emotions
                }
                
                results.append(result)
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
        except Exception as e:
            print(f"Analysis failed: {e}")
            return None

if __name__ == "__main__":
    try:
        detector = AudioEmotionDetector()
        audio_path = 'audio.mp3'
        output_path = 'audio_analysis.json'
        results = detector.analyze_audio(audio_path, output_path)
        
        if results:
            print("\nAnalysis Results:")
            for result in results:
                print(f"\nSegment {result['segment']} ({result['timestamp']}):")
                print(f"Text: {result['text']}")
                print("\nText Emotions:")
                if result['text_emotions']:
                    for emotion in result['text_emotions']:
                        print(f"{emotion['emotion']}: {emotion['score']}%")
                print("\nAudio Emotions:")
                for emotion in result['audio_emotions']:
                    print(f"{emotion['emotion']}: {emotion['score']}%")
    except Exception as e:
        print(f"Program failed: {e}")
