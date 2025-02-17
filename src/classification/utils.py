import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

class DentalClassificationDataset(Dataset):
    """Dataset class for dental X-ray classification."""
    
    def __init__(self, data_dir, transform=None, max_samples=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing class subdirectories
            transform (callable, optional): Optional transform to be applied
            max_samples (int, optional): Maximum number of samples to use
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get class names and create label mapping
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image files and their labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        class_idx
                    ))
        
        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

def preprocess_image(image_path, transform=None):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        transform (callable): Transform to be applied
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transform if provided
    if transform is not None:
        transformed = transform(image=image)
        image = transformed['image']
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image

def get_class_weights(data_dir):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        data_dir (str): Directory containing class subdirectories
        
    Returns:
        torch.Tensor: Class weights for weighted loss function
    """
    class_counts = {}
    total_samples = 0
    
    # Define expected classes
    expected_classes = ['normal', 'superficial', 'medium', 'deep']
    
    # Count samples in each class
    for class_name in expected_classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.endswith('.png')])
            class_counts[class_name] = count
            total_samples += count
        else:
            raise ValueError(f"Expected class directory not found: {class_name}")
    
    # Calculate weights
    weights = []
    for class_name in expected_classes:
        weight = total_samples / (len(expected_classes) * class_counts[class_name])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)

def visualize_predictions(image, pred_class, pred_prob, class_names):
    """
    Visualize classification predictions.
    
    Args:
        image (torch.Tensor): Input image
        pred_class (int): Predicted class index
        pred_prob (float): Prediction probability
        class_names (list): List of class names
        
    Returns:
        numpy.ndarray: Visualization image with predictions
    """
    # Convert image to numpy
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    # Add prediction text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{class_names[pred_class]}: {pred_prob:.2f}"
    cv2.putText(image, text, (10, 30), font, 1, (0, 255, 0), 2)
    
    return image 