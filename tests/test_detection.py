import unittest
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.model import DentalCariesDetector
from src.detection.utils import DentalDataset, preprocess_image

class TestDentalCariesDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = DentalCariesDetector(num_classes=2)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = cls.model.to(cls.device)
    
    def test_model_initialization(self):
        """Test if model initializes correctly."""
        self.assertIsInstance(self.model, DentalCariesDetector)
        self.assertEqual(self.model.model.roi_heads.box_predictor.cls_score.out_features, 2)
    
    def test_forward_pass(self):
        """Test if forward pass works with dummy data."""
        # Create dummy input
        batch_size = 2
        channels = 3
        height = 512
        width = 512
        
        images = [torch.rand(channels, height, width) for _ in range(batch_size)]
        images = [img.to(self.device) for img in images]
        
        # Test training mode
        targets = [{
            'boxes': torch.tensor([[100, 100, 200, 200]], device=self.device),
            'labels': torch.tensor([1], device=self.device),
            'masks': torch.ones((1, height, width), device=self.device)
        } for _ in range(batch_size)]
        
        output = self.model(images, targets)
        self.assertIsInstance(output, dict)
        self.assertTrue('loss_classifier' in output)
        self.assertTrue('loss_box_reg' in output)
        self.assertTrue('loss_mask' in output)
        
        # Test inference mode
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        self.assertEqual(len(predictions), batch_size)
        for pred in predictions:
            self.assertTrue('boxes' in pred)
            self.assertTrue('labels' in pred)
            self.assertTrue('masks' in pred)
            self.assertTrue('scores' in pred)
    
    def test_predict_method(self):
        """Test the predict method."""
        # Create dummy input
        image = torch.rand(3, 512, 512)
        image = image.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict([image])
        
        self.assertEqual(len(predictions), 1)
        pred = predictions[0]
        self.assertIsInstance(pred['boxes'], torch.Tensor)
        self.assertIsInstance(pred['labels'], torch.Tensor)
        self.assertIsInstance(pred['masks'], torch.Tensor)
        self.assertIsInstance(pred['scores'], torch.Tensor)

class TestDentalDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = 'data/dental_ai_dataset_v4_augmented/detailed_segmentation/train'
    
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly."""
        if os.path.exists(self.data_dir):
            dataset = DentalDataset(self.data_dir)
            self.assertGreater(len(dataset), 0)
            
            # Test getting an item
            image, target = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(target, dict)
            self.assertTrue('boxes' in target)
            self.assertTrue('labels' in target)
            self.assertTrue('masks' in target)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create dummy image path
        image_path = 'test_image.png'
        if os.path.exists(image_path):
            # Test preprocessing
            image = preprocess_image(image_path)
            self.assertIsInstance(image, torch.Tensor)
            self.assertEqual(image.shape[0], 3)  # RGB channels
            self.assertEqual(image.shape[1], 512)  # Height
            self.assertEqual(image.shape[2], 512)  # Width

if __name__ == '__main__':
    unittest.main() 