import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

class CNNEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128], output_dim=28, attention_heads=4, dropout_rate=0.2):
        super(CNNEmotionClassifier, self).__init__()
        
        # More robust CNN architecture
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=hidden_dims[0], kernel_size=3, padding=1),
            nn.Conv1d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, padding=1)
        ])
        
        # Batch normalization for better training stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[1])
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[1],
            num_heads=attention_heads,
            dropout=dropout_rate
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, x):
        # x shape should be [batch_size, channels, sequence_length]
        
        # Apply CNN layers with batch norm and activation
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = self.dropout(F.relu(bn(conv(x))))
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Self-attention
        # Reshape for attention: [sequence_length, batch_size, embed_dim]
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

def process_emotion_scores_for_cnn(emotion_scores):
    """
    Process emotion scores into format suitable for CNN
    
    Args:
        emotion_scores (dict): Dictionary of emotions and their scores
        
    Returns:
        tuple: (emotion_labels, emotion_values tensor)
    """
    emotion_labels = list(emotion_scores.keys())
    # Create a 1D array of scores
    emotion_values = np.array([score for score in emotion_scores.values()], dtype=np.float32)
    
    # Reshape to match CNN input requirements (batch_size, channels, sequence_length)
    emotion_values = torch.tensor(emotion_values).reshape(1, 1, -1)
    
    return emotion_labels, emotion_values

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

def process_with_cnn(emotion_scores, cnn_model, device):
    try:
        if not emotion_scores:
            print("No emotion scores to process.")
            return None, None
            
        emotion_labels, emotion_values = process_emotion_scores_for_cnn(emotion_scores)
        
        # Move to device
        emotion_values = emotion_values.to(device)
        
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(emotion_values)
            
        # Get probabilities and top emotions
        probs, indices = torch.topk(output, k=min(3, len(emotion_labels)))
        top_emotions = [(emotion_labels[idx], probs[0][i].item()) 
                       for i, idx in enumerate(indices[0])]
        
        return top_emotions, output
        
    except Exception as e:
        print(f"Error in CNN processing: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Initialize the CNN model
    cnn_model = CNNEmotionClassifier(input_dim=3, hidden_dims=[64, 128], output_dim=28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    # Your input text
    text = """
    Okay, so I'm Dhaval Mehta, and let's begin the interview.
    So the first section planning and organizing.
    Tell me about a time when you had to manage multiple tasks or projects simultaneously, and how did you prioritize and organize your work to meet the deadlines? 
    Okay, so during my third year, I was managing an internship at dry Chem, working on a research paper on reinforcement learning and the stock market, forecasting and organizing adapt as a part of my IT committee, I prioritized tasks using a calendar system and allocated dedicated time slots for each of the four things. So for instance, I worked on a I worked on a coding SAP workflows on mornings, and conducted research meetings with my other research scholars in the afternoon and the evening slots were mainly dedicated for my IT committee work, and this structured approach allowed me to meet all of the deadlines successfully describe a situation where you Were responsible for planning the development of a complex app. 
    What steps did you take to ensure that the project was well organized and executed on schedule?
    Okay, so while designing an Android app for detecting internet connectivity and low battery issues, I begin by outlining the requirements creating wire frames and breaking the project into milestones like UI design, broadcast receiver integration and testing. Weekly reviews were conducted to ensure the process and this organized approach ensured the app was completed within the timeline and met all the functionality requirements as I had to submit this particular app as a practical in the next week. So it was a tedious task and a last project. So I streamlined each and every and I broke each and every large task into smaller tasks and worked on them simultaneously, together and then the question is.
    Can you give an example of a project where you had to adjust your initial plan due to unexpected challenges? How did you adapt to your approach while maintaining focus on the overall goals? 
    Okay, so during my internship at dry Chem, I encountered delays in automating the QC testing workflow due to unforeseen complexities in the data mapping and the data mining work. So, I reallocated time from other phases, like from my committee work and from my research work to debug the issue, and I also collaborated with my other team members to simplify the data mapping work and streamlined and with all of these things, I also streamlined testing to stay on track. This adjustment ensured the project met all of its overall goals without compromising the quality of the project.
    Toolshen coming to phase two, the communication skills
    Describe a time when you had to explain a technical issue or concept to a non technical team member or client. How did you ensure they understood your explanation? 
    Okay, so at dry Chem, I explained the automated workflow of the purchasing team to the purchasing team, and to ensure clarity, I used simple language and relatable algorithms and relatable materials, like pictures and flow charts, And comparing the process to a postal service. I also added some visual aids, like videos and other flow charts so that they can easily get to know the the small, small concepts used in the project, which helped the team quickly grasp the concept of courses and then tell me about a situation where clear communication was crucial to the success of an app development project, and how did you ensure all stakeholders were on the same page throughout the so while working on an Android app, I maintained clear communication by holding weekly meetings which stakeholders to share progress, gather feedback and address concerns. I also use detailed documentations and regular status updates to ensure everyone was aligned and everyone was knowing that what exactly is going on in the project.
    Can you share an example of a time when you had to present your app development progress to a senior manager or client, and how did you tailor your message to meet the level of understanding and explanations, exceptions, expectations? 
    So during my internship, I presented the automated QC workflow to the senior management um under whom I was mainly doing the project, and I started with an overview of the problem and its impact. I explained the solution through visual aids and highlighted key results like time savings and improved accuracy. This approach made the presentation impactful, and it also made it easy to flow and follow.
    Then comes the phase three, the interpersonal skills.
    Tell me about a time when you worked closely with a colleague who had a different perspective of or work style than you. How did you manage to collaborate effectively? 
    There are many different times where I worked with I had actually worked with many different colleagues. So also at the time of my internship, I worked with a team of four to five members and who preferred a different database, data structure for the QC sample table. We discussed our approaches, evaluated their pros, cons, and my suggestions were also given and combined the best aspects for both the ideas, the collaboration improved our solution and strengthened our work relationship. Describe a situation where you had to handle conflict or disagreement with a team member during an app development project, how did you address the issue to maintain a positive working relationship. Okay, so this was a time while I was working for the IIT committee, so we were planning the ULECTRO event, there was a scheduling disagreement among the team members, and I facilitated a discussion to understand everyone's concerns and everyone's point of view, and also proposed a compromise, so that everybody get to know that what are the advantages and disadvantages of their ideas, and why should we not take their ideas, but we should take the other people's ideas, other person's ideas, and ensured alignment with the Event goals that ensured alignment with the event goals, yes, and this resolution maintained a positive environment and ensured the events success at the end.
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
        
        # Process with CNN for additional analysis
        top_emotions, cnn_output = process_with_cnn(emotion_scores, cnn_model, device)
        
        if top_emotions:
            print("\nTop Predicted Emotions from CNN model:")
            for emotion, probability in top_emotions:
                print(f"{emotion}: {probability:.3f}")
    else:
        print("No emotions were detected in the text.")
