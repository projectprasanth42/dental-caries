import unittest
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.model import DentalCariesClassifier
from src.classification.utils import DentalClassificationDataset, get_class_weights

class TestDentalCariesClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = DentalCariesClassifier(num_classes=3)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = cls.model.to(cls.device)
    
    def test_model_initialization(self):
        """Test if model initializes correctly."""
        self.assertIsInstance(self.model, DentalCariesClassifier)
        
        # Check model architecture
        self.assertIsInstance(self.model.model.fc, torch.nn.Sequential)
        self.assertEqual(self.model.model.fc[-1].out_features, 3)
    
    def test_forward_pass(self):
        """Test if forward pass works with dummy data."""
        # Create dummy input
        batch_size = 4
        channels = 3
        height = 512
        width = 512
        
        images = torch.rand(batch_size, channels, height, width)
        images = images.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Check output shape and properties
        self.assertEqual(outputs.shape, (batch_size, 3))
        self.assertTrue(torch.allclose(outputs.sum(dim=1), 
                                     torch.ones(batch_size, device=self.device),
                                     atol=1e-6))
    
    def test_predict_method(self):
        """Test the predict method."""
        # Create dummy input
        batch_size = 4
        channels = 3
        height = 512
        width = 512
        
        images = torch.rand(batch_size, channels, height, width)
        images = images.to(self.device)
        
        # Get predictions
        predictions, probabilities = self.model.predict(images)
        
        # Check predictions
        self.assertEqual(predictions.shape, (batch_size,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < 3))
        
        # Check probabilities
        self.assertEqual(probabilities.shape, (batch_size, 3))
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.all(probabilities <= 1))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), 
                                     torch.ones(batch_size, device=self.device),
                                     atol=1e-6))

class TestDentalClassificationDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = 'data/dental_ai_dataset_v4_augmented/three_level_classification/train'
    
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly."""
        if os.path.exists(self.data_dir):
            dataset = DentalClassificationDataset(self.data_dir)
            self.assertGreater(len(dataset), 0)
            
            # Check class names
            self.assertTrue(hasattr(dataset, 'classes'))
            self.assertTrue(hasattr(dataset, 'class_to_idx'))
            self.assertEqual(len(dataset.classes), 3)
            
            # Test getting an item
            image, label = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(label, int)
            self.assertTrue(0 <= label < 3)
    
    def test_class_weights(self):
        """Test class weight calculation."""
        if os.path.exists(self.data_dir):
            weights = get_class_weights(self.data_dir)
            
            self.assertIsInstance(weights, torch.Tensor)
            self.assertEqual(weights.shape, (3,))
            self.assertTrue(torch.all(weights > 0))

if __name__ == '__main__':
    unittest.main() 