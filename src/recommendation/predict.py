import torch
from .utils import create_recommendation_templates

def get_recommendation(severity, model=None):
    """
    Get personalized recommendation based on severity.
    
    Args:
        severity (str): Predicted severity class
        model (torch.nn.Module, optional): Pre-loaded model
        
    Returns:
        str: Formatted recommendation
    """
    templates = create_recommendation_templates()
    
    # Map severity to recommendation indices
    severity_recommendations = {
        'normal': [0, 4],  # Regular check-ups and oral hygiene
        'superficial': [1, 4],  # Fluoride treatment and oral hygiene
        'medium': [1, 3, 4],  # Fluoride, diet, and oral hygiene
        'deep': [2, 3]  # Immediate treatment and diet modification
    }
    
    # Get recommended indices for the severity
    rec_indices = severity_recommendations.get(severity, [0])
    
    # Get recommendations
    recommendations = []
    for idx in rec_indices:
        rec = templates[idx]
        recommendations.append(f"• {rec['text']}")
        
        # Add specific actions
        for action in rec['actions']:
            recommendations.append(f"  - {action}")
    
    return "\n".join(recommendations) 