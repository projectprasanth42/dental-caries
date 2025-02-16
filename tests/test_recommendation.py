import unittest
import torch
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommendation.model import DentalRecommendationSystem
from src.recommendation.utils import (
    DentalRecommendationDataset,
    create_recommendation_templates,
    generate_training_data,
    format_recommendation
)

class TestDentalRecommendationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = DentalRecommendationSystem(num_recommendations=5)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = cls.model.to(cls.device)
    
    def test_model_initialization(self):
        """Test if model initializes correctly."""
        self.assertIsInstance(self.model, DentalRecommendationSystem)
        
        # Check BERT model
        self.assertTrue(hasattr(self.model, 'bert'))
        self.assertTrue(hasattr(self.model, 'tokenizer'))
        
        # Check recommendation head
        self.assertTrue(hasattr(self.model, 'recommendation_head'))
        self.assertEqual(self.model.recommendation_head[-1].out_features, 5)
        
        # Check recommendations dictionary
        self.assertTrue(hasattr(self.model, 'recommendations'))
        self.assertEqual(len(self.model.recommendations), 5)
    
    def test_forward_pass(self):
        """Test if forward pass works with sample input."""
        # Create sample input
        input_text = "Patient has deep caries with severe pain"
        
        # Forward pass
        outputs = self.model(input_text)
        
        # Check output shape
        self.assertEqual(outputs.shape, (1, 5))  # batch_size=1, num_recommendations=5
    
    def test_get_recommendations(self):
        """Test recommendation generation."""
        # Test parameters
        condition = "caries"
        severity = "deep"
        confidence = 0.95
        
        # Get recommendations
        recommendations = self.model.get_recommendations(condition, severity, confidence)
        
        # Check output
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)
        for rec in recommendations:
            self.assertIsInstance(rec, str)

class TestRecommendationUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.templates = create_recommendation_templates()
        self.sample_data = generate_training_data(num_samples=10)
    
    def test_recommendation_templates(self):
        """Test recommendation template creation."""
        self.assertIsInstance(self.templates, dict)
        self.assertEqual(len(self.templates), 5)
        
        for idx, template in self.templates.items():
            self.assertIsInstance(template, dict)
            self.assertTrue('text' in template)
            self.assertTrue('severity' in template)
            self.assertTrue('urgency' in template)
    
    def test_training_data_generation(self):
        """Test synthetic training data generation."""
        self.assertIsInstance(self.sample_data, list)
        self.assertEqual(len(self.sample_data), 10)
        
        for sample in self.sample_data:
            self.assertIsInstance(sample, dict)
            self.assertTrue('condition' in sample)
            self.assertTrue('severity' in sample)
            self.assertTrue('symptoms' in sample)
            self.assertTrue('history' in sample)
            self.assertTrue('recommendation_idx' in sample)
            
            self.assertIsInstance(sample['symptoms'], list)
            self.assertIsInstance(sample['recommendation_idx'], int)
            self.assertTrue(0 <= sample['recommendation_idx'] < 5)
    
    def test_recommendation_formatting(self):
        """Test recommendation formatting."""
        recommendation = {
            'text': "Test recommendation",
            'urgency': 'immediate',
            'severity': 'high'
        }
        confidence = 0.95
        
        formatted = format_recommendation(recommendation, confidence)
        
        self.assertIsInstance(formatted, str)
        self.assertTrue('🚨' in formatted)  # Check emoji
        self.assertTrue('95.0%' in formatted)  # Check confidence
        self.assertTrue('Test recommendation' in formatted)

class TestRecommendationDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Generate sample data for testing
        self.data = generate_training_data(num_samples=10)
        self.test_file = 'test_recommendation_data.json'
        
        # Save test data
        with open(self.test_file, 'w') as f:
            json.dump(self.data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly."""
        dataset = DentalRecommendationDataset(self.test_file)
        
        self.assertEqual(len(dataset), 10)
        
        # Test getting an item
        inputs, label = dataset[0]
        
        self.assertIsInstance(inputs, dict)
        self.assertTrue('input_ids' in inputs)
        self.assertTrue('attention_mask' in inputs)
        self.assertIsInstance(label, torch.Tensor)
        self.assertTrue(0 <= label.item() < 5)

if __name__ == '__main__':
    unittest.main() 