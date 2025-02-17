import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.detection.train import train_model as train_detection
from src.classification.train import train_model as train_classification
from src.recommendation.train import train_model as train_recommendation

def test_training():
    """Run test training for detection, classification, and recommendation models."""
    data_dir = 'dental_ai_dataset_v4_augmented'
    
    try:
        # Test detection model
        print("\n=== Testing Detection Model Training ===")
        train_detection(data_dir, num_epochs=3, batch_size=2, max_samples=12)
    except Exception as e:
        print(f"Detection model training failed: {str(e)}")
    
    try:
        # Test classification model
        print("\n=== Testing Classification Model Training ===")
        train_classification(data_dir, num_epochs=3, batch_size=4, max_samples=12)
    except Exception as e:
        print(f"Classification model training failed: {str(e)}")
    
    try:
        # Test recommendation model
        print("\n=== Testing Recommendation Model Training ===")
        train_recommendation(data_dir, num_epochs=3, batch_size=4)
    except Exception as e:
        print(f"Recommendation model training failed: {str(e)}")

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models/detection', exist_ok=True)
    os.makedirs('models/classification', exist_ok=True)
    os.makedirs('models/recommendation', exist_ok=True)
    
    test_training() 