import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_emotions_roberta(text, chunk_size=450, model_name="SamLowe/roberta-base-go_emotions"):
    """
    Analyze emotions in text using RoBERTa model
    """
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

class RefinedANNEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 512, 256, 128], output_dim=28, dropout_rate=0.2):
        super(RefinedANNEmotionClassifier, self).__init__()
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        current_dim = input_dim
        
        # Build hidden layers with residual connections
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output layer with different activation
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Forward pass through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer with softmax
        x = self.output_layer(x)
        
        # Temperature scaling for sharper predictions
        temperature = 0.5
        return F.softmax(x / temperature, dim=-1)

def process_emotion_scores_for_ann(emotion_scores, scaler=None):
    """
    Process emotion scores with improved scaling
    """
    emotion_labels = list(emotion_scores.keys())
    emotion_values = np.array([score for score in emotion_scores.values()], dtype=np.float32)
    
    # Initialize scaler if not provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        emotion_values = scaler.fit_transform(emotion_values.reshape(-1, 1)).flatten()
    else:
        emotion_values = scaler.transform(emotion_values.reshape(-1, 1)).flatten()
    
    # Add feature engineering
    emotion_values_enhanced = np.concatenate([
        emotion_values,
        np.square(emotion_values),  # Square features
        np.abs(emotion_values),     # Absolute values
        np.exp(emotion_values)      # Exponential features
    ])
    
    # Reshape for ANN (batch_size, features)
    emotion_tensor = torch.tensor(emotion_values_enhanced).reshape(1, -1)
    return emotion_labels, emotion_tensor, scaler

def analyze_emotions_with_confidence(emotion_scores, ann_model, device, confidence_threshold=0.1):
    """
    Analyze emotions with confidence scoring
    """
    try:
        if not emotion_scores:
            return None, None
            
        emotion_labels, emotion_values, _ = process_emotion_scores_for_ann(emotion_scores)
        emotion_values = emotion_values.to(device)
        
        ann_model.eval()
        with torch.no_grad():
            output = ann_model(emotion_values)
            
            # Get probabilities and filter by confidence threshold
            probs, indices = torch.topk(output, k=len(emotion_labels))
            confident_emotions = [
                (emotion_labels[idx], probs[0][i].item())
                for i, idx in enumerate(indices[0])
                if probs[0][i].item() > confidence_threshold
            ]
            
            # Sort by probability
            confident_emotions.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 3 confident predictions
            top_emotions = confident_emotions[:3]
            
            return top_emotions, output
            
    except Exception as e:
        print(f"Error in ANN processing: {str(e)}")
        return None, None

def plot_emotion_comparison(roberta_scores, ann_predictions):
    """
    Plot comparison between RoBERTa and ANN predictions
    """
    plt.figure(figsize=(15, 6))
    
    # Plot RoBERTa scores
    plt.subplot(1, 2, 1)
    emotions_roberta = list(roberta_scores.keys())[:5]
    scores_roberta = [roberta_scores[e] for e in emotions_roberta]
    plt.bar(emotions_roberta, scores_roberta, color='lightcoral')
    plt.title('RoBERTa Top 5 Emotions')
    plt.xticks(rotation=45)
    
    # Plot ANN predictions
    plt.subplot(1, 2, 2)
    emotions_ann = [e[0] for e in ann_predictions]
    scores_ann = [e[1] for e in ann_predictions]
    plt.bar(emotions_ann, scores_ann, color='lightgreen')
    plt.title('ANN Top Predictions')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the refined ANN model with expanded input
    input_dim = 112  # 28 * 4 due to feature engineering
    ann_model = RefinedANNEmotionClassifier(
        input_dim=input_dim,
        hidden_dims=[256, 512, 256, 128],
        output_dim=28,
        dropout_rate=0.2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_model.to(device)

    # Your input text
    text = """
    Okay, so I'm Dhaval Mehta, and let's begin the interview.
    So the first section planning and organizing.
    Tell me about a time when you had to manage multiple tasks or projects simultaneously, and how did you prioritize and organize your work to meet the deadlines? 
    """
    
    # Run the analysis
    print("Analyzing emotions using RoBERTa model...")
    emotion_scores = analyze_emotions_roberta(text)
    
    if emotion_scores:
        # Print RoBERTa results
        print("\nRoBERTa Emotion Scores:")
        for emotion, score in list(emotion_scores.items())[:5]:
            print(f"{emotion}: {score:.3f}")
        
        # Process with refined ANN
        top_emotions, ann_output = analyze_emotions_with_confidence(
            emotion_scores, 
            ann_model, 
            device,
            confidence_threshold=0.1
        )
        
        if top_emotions:
            print("\nRefined ANN Predictions:")
            for emotion, probability in top_emotions:
                print(f"{emotion}: {probability:.3f}")
            
            # Plot comparison
            plot_emotion_comparison(emotion_scores, top_emotions)
    else:
        print("No emotions were detected in the text.")
