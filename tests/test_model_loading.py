import torch
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
from src.detection.model import DentalCariesDetector
from src.classification.model import DentalCariesClassifier
from src.recommendation.model import DentalRecommendationSystem

def preprocess_image(image_path):
    """Preprocess image for model input."""
    try:
        # Read image
        image = Image.open(image_path).convert('RGB')
        
        # Create transforms
        detection_transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        detection_tensor = detection_transform(image)
        classification_tensor = classification_transform(image)
        
        return detection_tensor, classification_tensor
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def test_model_loading():
    print("\n=== Testing Model Loading ===")
    
    try:
        # Load Detection Model
        print("\nTesting Detection Model:")
        detection_model = DentalCariesDetector()
        detection_path = 'models/detection/model.pth'
        if os.path.exists(detection_path):
            checkpoint = torch.load(detection_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                detection_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                detection_model.load_state_dict(checkpoint)
            print("✓ Detection model loaded successfully")
        else:
            print("✗ Detection model file not found")
            return False
            
        # Load Classification Model
        print("\nTesting Classification Model:")
        classification_model = DentalCariesClassifier()
        classification_path = 'models/classification/model.pth'
        if os.path.exists(classification_path):
            checkpoint = torch.load(classification_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                classification_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                classification_model.load_state_dict(checkpoint)
            print("✓ Classification model loaded successfully")
        else:
            print("✗ Classification model file not found")
            return False
            
        # Load Recommendation Model
        print("\nTesting Recommendation Model:")
        recommendation_model = DentalRecommendationSystem()
        recommendation_path = 'models/recommendation/model.pth'
        if os.path.exists(recommendation_path):
            checkpoint = torch.load(recommendation_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                recommendation_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                recommendation_model.load_state_dict(checkpoint)
            print("✓ Recommendation model loaded successfully")
        else:
            print("✗ Recommendation model file not found")
            return False
            
        print("\nAll models loaded successfully!")
        
        # Test with sample images
        print("\n=== Testing with Sample Images ===")
        sample_images = {
            'normal': 'sample-test/normal_1.png',
            'superficial': 'sample-test/superficial_1.png',
            'medium': 'sample-test/medium_1.png',
            'deep': 'sample-test/deep_1.png'
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detection_model.to(device)
        classification_model.to(device)
        recommendation_model.to(device)
        
        detection_model.eval()
        classification_model.eval()
        recommendation_model.eval()
        
        for severity, image_path in sample_images.items():
            print(f"\nTesting {severity} image:")
            try:
                # Preprocess image
                detection_tensor, classification_tensor = preprocess_image(image_path)
                detection_tensor = detection_tensor.to(device)
                classification_tensor = classification_tensor.to(device)
                
                # Get detection results
                with torch.no_grad():
                    detections = detection_model([detection_tensor])
                    boxes = detections[0]['boxes'].cpu().numpy()
                    scores = detections[0]['scores'].cpu().numpy()
                    
                    # Filter by confidence
                    confidence_threshold = 0.5
                    high_conf_indices = scores > confidence_threshold
                    filtered_boxes = boxes[high_conf_indices]
                    filtered_scores = scores[high_conf_indices]
                    
                    print(f"Detection: Found {len(filtered_boxes)} regions")
                    
                    # Get classification results
                    class_pred, class_prob = classification_model.predict(classification_tensor.unsqueeze(0))
                    severity_map = {0: 'superficial', 1: 'medium', 2: 'deep'}
                    predicted_severity = severity_map[class_pred.item()]
                    confidence = float(class_prob[0][class_pred].item())
                    
                    print(f"Classification: Predicted {predicted_severity} with {confidence:.2f} confidence")
                    
                    # Get recommendations
                    recommendations = recommendation_model.get_recommendations('caries', predicted_severity, confidence)
                    print(f"Recommendations: Generated {len(recommendations)} recommendations")
                    
                print("✓ Test completed successfully")
                
            except Exception as e:
                print(f"✗ Test failed: {str(e)}")
                return False
        
        print("\nAll sample image tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return False

if __name__ == '__main__':
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    success = test_model_loading()
    if success:
        print("\nAll tests completed successfully. Ready to proceed with fresh implementation.")
    else:
        print("\nTests failed. Please fix the issues before proceeding.") 