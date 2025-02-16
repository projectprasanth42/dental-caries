import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer

class RecommendationDataset(Dataset):
    """Dataset class for dental recommendation system."""
    
    def __init__(self, data_path, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to JSON file containing training data
            max_length (int): Maximum sequence length for BERT
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
        # Load training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        item = self.data[idx]
        
        # Create input text
        input_text = f"Patient has {item['severity']} {item['condition']} "
        if 'symptoms' in item:
            input_text += f"with symptoms: {', '.join(item['symptoms'])}. "
        if 'history' in item:
            input_text += f"Medical history: {item['history']}."
        
        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        label = torch.tensor(item['recommendation_idx'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }, label

def create_recommendation_templates():
    """
    Create a set of recommendation templates for different scenarios.
    
    Returns:
        dict: Mapping of recommendation indices to template strings
    """
    return {
        0: {
            'text': "Regular dental check-ups are recommended every 6 months.",
            'severity': 'low',
            'urgency': 'routine'
        },
        1: {
            'text': "Consider using fluoride toothpaste to strengthen enamel.",
            'severity': 'low',
            'urgency': 'preventive'
        },
        2: {
            'text': "Immediate dental visit is required for deep cavity treatment.",
            'severity': 'high',
            'urgency': 'immediate'
        },
        3: {
            'text': "Modify diet to reduce sugar intake and prevent cavity progression.",
            'severity': 'medium',
            'urgency': 'important'
        },
        4: {
            'text': "Improve brushing technique, focusing on hard-to-reach areas.",
            'severity': 'low',
            'urgency': 'educational'
        }
    }

def generate_training_data(num_samples=1000):
    """
    Generate synthetic training data for the recommendation system.
    
    Args:
        num_samples (int): Number of training samples to generate
        
    Returns:
        list: List of training examples
    """
    conditions = ['caries', 'cavity', 'decay']
    severities = ['superficial', 'medium', 'deep']
    symptoms = [
        'pain when eating',
        'sensitivity to hot/cold',
        'visible holes',
        'dark spots',
        'persistent toothache'
    ]
    histories = [
        'regular dental visits',
        'poor oral hygiene',
        'high sugar diet',
        'smoking',
        'previous cavities'
    ]
    
    import random
    
    data = []
    for _ in range(num_samples):
        severity = random.choice(severities)
        condition = random.choice(conditions)
        
        # Select recommendation based on severity
        if severity == 'deep':
            rec_idx = 2  # immediate treatment
        elif severity == 'medium':
            rec_idx = random.choice([1, 3, 4])  # preventive measures
        else:
            rec_idx = random.choice([0, 1, 4])  # routine care
            
        # Create sample
        sample = {
            'condition': condition,
            'severity': severity,
            'symptoms': random.sample(symptoms, random.randint(1, 3)),
            'history': random.choice(histories),
            'recommendation_idx': rec_idx
        }
        
        data.append(sample)
    
    return data

def format_recommendation(recommendation, confidence):
    """
    Format a recommendation with additional context.
    
    Args:
        recommendation (dict): Recommendation template
        confidence (float): Model's confidence in the recommendation
        
    Returns:
        str: Formatted recommendation string
    """
    urgency_emoji = {
        'immediate': '🚨',
        'important': '⚠️',
        'preventive': '💡',
        'routine': '✔️',
        'educational': '📚'
    }
    
    emoji = urgency_emoji.get(recommendation['urgency'], '')
    confidence_str = f"({confidence:.1%} confidence)"
    
    return f"{emoji} {recommendation['text']} {confidence_str}" 