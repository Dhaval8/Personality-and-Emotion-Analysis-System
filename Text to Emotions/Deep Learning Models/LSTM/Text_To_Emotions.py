import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128], output_dim=28, num_layers=2, dropout_rate=0.2):
        super(LSTMEmotionClassifier, self).__init__()
        
        # LSTM layers with multiple layers for better feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[1],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Batch normalization for better training stability
        self.batch_norm = nn.BatchNorm1d(hidden_dims[1] * 2)  # *2 for bidirectional
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dims[1] * 2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, x):
        # x shape should be [batch_size, sequence_length, input_dim]
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Get the final output from both directions
        lstm_out = lstm_out[:, -1, :]
        
        # Apply batch normalization
        lstm_out = self.batch_norm(lstm_out)
        
        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(lstm_out)))
        x = self.fc2(x)
        
        return F.softmax(x, dim=-1)

def analyze_emotions_roberta(text, chunk_size=450, model_name="SamLowe/roberta-base-go_emotions"):
    try:
        emotion_analyzer = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Improved chunking with overlap
        overlap = chunk_size // 4
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only process chunks of sufficient length
                chunks.append(chunk)
        
        all_scores = []
        for chunk in chunks:
            try:
                result = emotion_analyzer(chunk)
                if isinstance(result, list) and result:
                    scores_dict = {res['label']: res['score'] for res in result[0]}
                    all_scores.append(scores_dict)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        if not all_scores:
            print("No valid scores were generated.")
            return {}
        
        # Weighted average based on chunk length
        final_scores = {}
        weights = [len(chunk) for chunk in chunks]
        total_weight = sum(weights)
        
        for emotion in all_scores[0].keys():
            weighted_sum = sum(
                scores.get(emotion, 0) * weight 
                for scores, weight in zip(all_scores, weights)
            )
            final_scores[emotion] = weighted_sum / total_weight if total_weight > 0 else 0
            
        return dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))
        
    except Exception as e:
        print(f"Error in emotion analysis: {str(e)}")
        return {}

def plot_roberta_emotions(emotion_scores, top_n=10):
    """
    Plot the top N emotions using a bar chart
    
    Args:
        emotion_scores (dict): Dictionary of emotions and their scores
        top_n (int): Number of top emotions to plot
    """
    # Get top N emotions
    top_emotions = dict(list(emotion_scores.items())[:top_n])
    
    # Create lists of emotions and their scores
    emotions = list(top_emotions.keys())
    scores = list(top_emotions.values())
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bars = plt.bar(emotions, scores, color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title('Top Emotions Detected by RoBERTa Model')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.3f}',
            ha='center', 
            va='bottom'
        )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def process_emotion_scores_for_lstm(emotion_scores):
    """
    Process emotion scores into format suitable for LSTM
    
    Args:
        emotion_scores (dict): Dictionary of emotions and their scores
        
    Returns:
        tuple: (emotion_labels, emotion_values tensor)
    """
    emotion_labels = list(emotion_scores.keys())
    # Create a 1D array of scores
    emotion_values = np.array([score for score in emotion_scores.values()], dtype=np.float32)
    
    # Reshape to match LSTM input requirements (batch_size, sequence_length, input_dim)
    emotion_values = torch.tensor(emotion_values).reshape(1, -1, 1)
    
    return emotion_labels, emotion_values

def process_with_lstm(emotion_scores, lstm_model, device):
    try:
        if not emotion_scores:
            print("No emotion scores to process.")
            return None, None
            
        emotion_labels, emotion_values = process_emotion_scores_for_lstm(emotion_scores)
        
        # Move to device
        emotion_values = emotion_values.to(device)
        
        lstm_model.eval()
        with torch.no_grad():
            output = lstm_model(emotion_values)
            
        # Get probabilities and top emotions
        probs, indices = torch.topk(output, k=min(3, len(emotion_labels)))
        top_emotions = [(emotion_labels[idx], probs[0][i].item()) 
                       for i, idx in enumerate(indices[0])]
        
        return top_emotions, output
        
    except Exception as e:
        print(f"Error in LSTM processing: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Initialize the LSTM model
    lstm_model = LSTMEmotionClassifier(input_dim=1, hidden_dims=[64, 128], output_dim=28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)

    # Your input text
    text = """
    Okay, so I'm Dhaval Mehta, and let's begin the interview.
    So the first section planning and organizing.
    Tell me about a time when you had to manage multiple tasks or projects simultaneously, and how did you prioritize and organize your work to meet the deadlines? 
    Okay, so during my third year, I was managing an internship at dry Chem, working on a research paper on reinforcement learning and the stock market, forecasting and organizing adapt as a part of my IT committee, I prioritized tasks using a calendar system and allocated dedicated time slots for each of the four things. So for instance, I worked on a I worked on a coding SAP workflows on mornings, and conducted research meetings with my other research scholars in the afternoon and the evening slots were mainly dedicated for my IT committee work, and this structured approach allowed me to meet all of the deadlines successfully describe a situation where you Were responsible for planning the development of a complex app. 
    What steps did you take to ensure that the project was well organized and executed on schedule?
    """
    
    # Run the analysis
    print("Analyzing emotions using RoBERTa model...")
    emotion_scores = analyze_emotions_roberta(text)
    
    if emotion_scores:
        # Print the results
        print("\nDetailed Emotion Scores:")
        for emotion, score in emotion_scores.items():
            print(f"{emotion}: {score}")
        
        # Visualize the results
        plot_roberta_emotions(emotion_scores, top_n=10)
        
        # Process with LSTM for additional analysis
        top_emotions, lstm_output = process_with_lstm(emotion_scores, lstm_model, device)
        
        if top_emotions:
            print("\nTop Predicted Emotions from LSTM model:")
            for emotion, probability in top_emotions:
                print(f"{emotion}: {probability:.3f}")
    else:
        print("No emotions were detected in the text.")
