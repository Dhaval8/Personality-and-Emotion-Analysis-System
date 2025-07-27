import librosa
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import soundfile as sf
import json
import time

class AudioEmotionDetector:
    def __init__(self):
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            self.recognizer = sr.Recognizer()
        except Exception as e:
            raise Exception(f"Failed to initialize dependencies: {e}")

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
            return ""  # Return empty string for unrecognizable audio
        except Exception as e:
            raise Exception(f"Transcription error: {e}")

    def analyze_emotion(self, text):
        if not text:
            return None
        
        try:
            emotions = self.emotion_classifier(text)[0]
            return {
                'text': text,
                'emotions': [
                    {
                        'emotion': score['label'],
                        'score': round(score['score'] * 100, 2)
                    }
                    for score in emotions
                ]
            }
        except Exception as e:
            raise Exception(f"Emotion analysis error: {e}")

    def analyze_audio(self, audio_path, output_path=None):
        print("Starting audio analysis...")
        
        try:
            segments, sample_rate = self.segment_audio(audio_path)
            results = []
            
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)}")
                
                text = self.transcribe_audio(segment, sample_rate)
                if text:
                    emotion_data = self.analyze_emotion(text)
                    if emotion_data:
                        results.append({
                            'segment': i+1,
                            'timestamp': f"{i*10}-{(i+1)*10}s",
                            **emotion_data
                        })
            
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
                print("\nEmotions:")
                for emotion in result['emotions']:
                    print(f"{emotion['emotion']}: {emotion['score']}%")
    except Exception as e:
        print(f"Program failed: {e}")