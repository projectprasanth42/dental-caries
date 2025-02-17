import torch
import torch.nn as nn
import torchvision.models as models

class DentalCariesClassifier(nn.Module):
    """Dental caries classification model based on ResNet-50."""
    
    def __init__(self, num_classes=4):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of classes (normal, superficial, medium, deep)
        """
        super().__init__()
        
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=True)
        
        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Class logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def predict(self, x):
        """
        Make predictions on a batch of images.
        
        Args:
            x (Tensor): Batch of images [B, C, H, W]
            
        Returns:
            tuple: (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Debug information
            print("\nClassification Debug Info:")
            print(f"Logits: {logits}")
            print(f"Probabilities: {probabilities}")
            print(f"Predictions: {predictions}")
            
        return predictions, probabilities 