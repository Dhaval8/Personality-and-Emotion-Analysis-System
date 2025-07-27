import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Embedding, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class EmotionAnalyzer:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        
    def preprocess_text(self, texts):
        """Tokenize and pad the text data"""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)
    
    def create_lstm_model(self, num_classes):
        """Create LSTM model for emotion classification"""
        model = Sequential([ 
            Embedding(self.max_words, 128, input_length=self.max_len),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_cnn_model(self, num_classes):
        """Create CNN model for emotion classification"""
        model = Sequential([ 
            Embedding(self.max_words, 128, input_length=self.max_len),
            Conv1D(256, 5, activation='relu'),
            MaxPooling1D(2), 
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 5, activation='relu'),
            MaxPooling1D(2), 
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_ann_model(self, num_classes):
        """Create ANN model for emotion classification"""
        model = Sequential([ 
            Embedding(self.max_words, 128, input_length=self.max_len),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        return model

def prepare_data(texts, labels, test_size=0.2):
    """Prepare data for training"""
    from sklearn.model_selection import train_test_split
    return train_test_split(texts, labels, test_size=test_size, random_state=42)

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Train the model"""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.show()

def plot_emotion_histogram(y_true, y_pred, emotion_labels):
    """Plot histogram of predicted emotions"""
    predicted_emotions = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(np.argmax(y_true, axis=1), predicted_emotions)
    emotion_counts = cm.sum(axis=1)
    emotion_percentages = emotion_counts / emotion_counts.sum() * 100

    plt.figure(figsize=(8, 6))
    plt.bar(emotion_labels, emotion_percentages, color='skyblue')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example data (you'll need to replace this with your actual data)
    texts = ['''
        Im feeling very happy today,
        "This makes me so angry",
        "I'm sad about what happened",
        "I am so joyful and excited",
        "Feeling down and gloomy today",
        # Add more examples...
        '''
    ]
    
    # Example labels (one-hot encoded)
    emotions = ['happy', 'angry', 'sad', 'joyful', 'neutral']
    num_classes = len(emotions)
    
    # Initialize analyzer
    analyzer = EmotionAnalyzer()
    
    # Preprocess text
    X = analyzer.preprocess_text(texts)
    
    # Create dummy labels for example (you should replace this with actual labels)
    y = np.random.randint(num_classes, size=(len(texts),))
    y = tf.keras.utils.to_categorical(y, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Create and train different models
    models = {
        'LSTM': analyzer.create_lstm_model(num_classes),
        'CNN': analyzer.create_cnn_model(num_classes),
        'ANN': analyzer.create_ann_model(num_classes)
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = train_model(
            model, 
            X_train, y_train, 
            X_test, y_test,
            epochs=10  # Increased epochs for better accuracy
        )
        
        plot_training_history(history)
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        
        # Plot emotion distribution histogram
        y_pred = model.predict(X_test)
        plot_emotion_histogram(y_test, y_pred, emotions)
