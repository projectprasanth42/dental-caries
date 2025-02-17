import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class DentalRecommendationSystem(nn.Module):
    """
    BERT-based recommendation system for dental care advice.
    Fine-tuned on dental domain knowledge.
    """
    
    def __init__(self, hidden_size=256, num_recommendations=5):
        """
        Initialize the model.
        
        Args:
            hidden_size (int): Size of hidden layer
            num_recommendations (int): Number of possible recommendations
        """
        super().__init__()
        
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Custom head for recommendation generation
        self.network = nn.Sequential(
            nn.Linear(768, hidden_size),  # 768 is BERT's hidden size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_recommendations)
        )
        
        # Define recommendation templates
        self.recommendations = {
            0: "Regular dental check-ups are recommended every 6 months.",
            1: "Consider using fluoride toothpaste to strengthen enamel.",
            2: "Immediate dental visit is required for deep cavity treatment.",
            3: "Modify diet to reduce sugar intake and prevent cavity progression.",
            4: "Improve brushing technique, focusing on hard-to-reach areas."
        }
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (dict): Dictionary containing tokenized input
            
        Returns:
            Tensor: Recommendation logits
        """
        # Get BERT features
        outputs = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask']
        )
        
        # Use [CLS] token output as features
        features = outputs.last_hidden_state[:, 0, :]
        
        # Generate recommendation logits
        return self.network(features)
    
    def get_recommendations(self, condition, severity, confidence):
        """
        Generate personalized recommendations based on condition and severity.
        
        Args:
            condition (str): Type of dental condition
            severity (str): Severity level (superficial, medium, deep)
            confidence (float): Model's confidence in diagnosis
            
        Returns:
            list: Ranked list of recommendations
        """
        # Create input text from condition and severity
        input_text = f"Patient has {severity} {condition} with {confidence:.2f} confidence."
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Get recommendation scores
        self.eval()
        with torch.no_grad():
            scores = self.forward(inputs)
            probabilities = torch.softmax(scores, dim=1)
            ranked_indices = torch.argsort(probabilities, dim=1, descending=True)[0]
        
        # Return ranked recommendations with confidence scores
        recommendations = []
        for idx in ranked_indices:
            idx = idx.item()
            confidence = probabilities[0, idx].item()
            recommendations.append({
                'text': self.recommendations[idx],
                'confidence': confidence
            })
        
        return recommendations 