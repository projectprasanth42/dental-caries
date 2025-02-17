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
            'text': "Schedule regular dental check-ups every 6 months for early detection and prevention. Professional cleaning and examination are essential for maintaining oral health.",
            'severity': 'low',
            'urgency': 'routine',
            'actions': [
                'Schedule bi-annual check-ups',
                'Get professional cleaning',
                'Monitor any changes'
            ]
        },
        1: {
            'text': "Use high-fluoride toothpaste and consider professional fluoride treatment. This helps strengthen tooth enamel and prevent cavity progression.",
            'severity': 'low',
            'urgency': 'preventive',
            'actions': [
                'Switch to fluoride toothpaste',
                'Consider fluoride treatment',
                'Use fluoride mouthwash'
            ]
        },
        2: {
            'text': "Immediate dental visit required for treatment of deep cavity. Delay may lead to more serious complications like infection or abscess.",
            'severity': 'high',
            'urgency': 'immediate',
            'actions': [
                'Schedule emergency appointment',
                'Manage pain appropriately',
                'Avoid pressure on affected area'
            ]
        },
        3: {
            'text': "Modify diet to reduce sugar and acid intake. Limit snacking frequency and choose tooth-friendly foods to prevent cavity progression.",
            'severity': 'medium',
            'urgency': 'important',
            'actions': [
                'Reduce sugar consumption',
                'Avoid acidic drinks',
                'Choose healthy snacks'
            ]
        },
        4: {
            'text': "Improve brushing technique focusing on problem areas. Use proper brushing method, ensure thorough cleaning, and consider using interdental cleaning tools.",
            'severity': 'low',
            'urgency': 'educational',
            'actions': [
                'Use proper brushing technique',
                'Implement interdental cleaning',
                'Brush for full 2 minutes'
            ]
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
    conditions = {
        'caries': {
            'symptoms': [
                'pain when eating sweets',
                'sensitivity to hot/cold',
                'visible dark spots',
                'toothache',
                'visible holes in teeth'
            ],
            'risk_factors': [
                'poor oral hygiene',
                'high sugar diet',
                'irregular dental visits',
                'dry mouth',
                'acidic foods/drinks'
            ]
        },
        'cavity': {
            'symptoms': [
                'tooth sensitivity',
                'mild to sharp pain',
                'visible pits',
                'pain when biting',
                'staining on teeth'
            ],
            'risk_factors': [
                'inadequate brushing',
                'frequent snacking',
                'lack of fluoride',
                'deep tooth grooves',
                'genetic predisposition'
            ]
        },
        'decay': {
            'symptoms': [
                'continuous pain',
                'bad breath',
                'bitter taste',
                'swollen gums',
                'difficulty chewing'
            ],
            'risk_factors': [
                'smoking',
                'medical conditions',
                'medications affecting saliva',
                'poor diet',
                'family history'
            ]
        }
    }
    
    severity_recommendations = {
        'superficial': [
            0,  # Regular check-ups
            1,  # Fluoride treatment
            4   # Improve brushing
        ],
        'medium': [
            1,  # Fluoride treatment
            3,  # Diet modification
            4   # Improve brushing
        ],
        'deep': [
            2,  # Immediate treatment
            3,  # Diet modification
            1   # Fluoride treatment
        ]
    }
    
    histories = [
        'regular dental visits every 6 months',
        'last dental visit over a year ago',
        'history of multiple cavities',
        'recent dental work',
        'first cavity detection',
        'ongoing dental treatment',
        'family history of dental problems',
        'good oral hygiene routine',
        'inconsistent oral care',
        'recent changes in oral health'
    ]
    
    import random
    
    data = []
    for _ in range(num_samples):
        # Select condition and severity
        condition = random.choice(list(conditions.keys()))
        severity = random.choice(['superficial', 'medium', 'deep'])
        
        # Get condition-specific data
        condition_data = conditions[condition]
        symptoms = random.sample(condition_data['symptoms'], 
                               random.randint(1, 3))
        risk_factors = random.sample(condition_data['risk_factors'], 
                                   random.randint(1, 2))
        
        # Select appropriate recommendation based on severity
        rec_idx = random.choice(severity_recommendations[severity])
            
        # Create sample with detailed context
        sample = {
            'condition': condition,
            'severity': severity,
            'symptoms': symptoms,
            'risk_factors': risk_factors,
            'history': random.choice(histories),
            'recommendation_idx': rec_idx,
            'context': f"Patient presents with {severity} {condition}, " + \
                      f"showing symptoms of {', '.join(symptoms)}. " + \
                      f"Risk factors include {', '.join(risk_factors)}. " + \
                      f"Medical history: {random.choice(histories)}."
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