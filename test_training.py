import os
import sys
import torch
import platform

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.detection.train import train_model as train_detection
from src.classification.train import train_model as train_classification

def get_data_path():
    """Get the appropriate data path based on the environment."""
    if os.path.exists('/kaggle/input/augmented-dataset/dental_ai_dataset_v4_augmented'):
        # Kaggle environment
        return '/kaggle/input/augmented-dataset/dental_ai_dataset_v4_augmented'
    elif os.path.exists('/content/drive/MyDrive/dental_ai_dataset_v4_augmented'):
        # Google Colab environment
        return '/content/drive/MyDrive/dental_ai_dataset_v4_augmented'
    else:
        # Local environment
        return 'dental_ai_dataset_v4_augmented'

def test_training():
    """Run test training for both detection and classification models."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Get appropriate data path
    data_dir = get_data_path()
    print(f"Using data directory: {data_dir}")
    
    try:
        # Test detection model
        print("\n=== Testing Detection Model Training ===")
        train_detection(data_dir, num_epochs=1, batch_size=2, max_samples=4, device=device)
    except Exception as e:
        print(f"Detection model training failed: {str(e)}")
    
    try:
        # Test classification model
        print("\n=== Testing Classification Model Training ===")
        train_classification(data_dir, num_epochs=1, batch_size=4, max_samples=4, device=device)
    except Exception as e:
        print(f"Classification model training failed: {str(e)}")

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models/detection', exist_ok=True)
    os.makedirs('models/classification', exist_ok=True)
    
    # Print environment information
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    test_training() 