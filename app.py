from flask import Flask, request, jsonify, render_template
import torch
import os
import sys
from PIL import Image
import io
import torchvision.transforms as transforms
import warnings
import traceback

# Filter out deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from detection.model import DentalCariesDetector
from classification.model import DentalCariesClassifier
from recommendation.model import DentalRecommendationSystem

app = Flask(__name__)

# Initialize models
detector = None
classifier = None
recommender = None

def log_error(error_msg, error_obj=None):
    """Log error with traceback if available."""
    print("\n=== Error Details ===")
    print(f"Error Message: {error_msg}")
    if error_obj:
        print("Stack Trace:")
        traceback.print_exc()
    print("==================\n")

def load_models():
    """Load all models from checkpoints."""
    global detector, classifier, recommender
    
    try:
        print("Loading models...")
        # Load detection model
        detector = DentalCariesDetector()
        if os.path.exists('models/detection/model.pth'):
            checkpoint = torch.load('models/detection/model.pth', weights_only=True)
            if 'model_state_dict' in checkpoint:
                detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                detector.load_state_dict(checkpoint)
        detector.eval()
        print("Detection model loaded")
        
        # Load classification model
        classifier = DentalCariesClassifier()
        if os.path.exists('models/classification/model.pth'):
            checkpoint = torch.load('models/classification/model.pth', weights_only=True)
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
        classifier.eval()
        print("Classification model loaded")
        
        # Load recommendation model
        recommender = DentalRecommendationSystem()
        if os.path.exists('models/recommendation/model.pth'):
            checkpoint = torch.load('models/recommendation/model.pth', weights_only=True)
            if 'model_state_dict' in checkpoint:
                recommender.load_state_dict(checkpoint['model_state_dict'])
            else:
                recommender.load_state_dict(checkpoint)
        recommender.eval()
        print("Recommendation model loaded")
        
        print("All models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def analyze_image(image_file):
    """
    Analyze a dental X-ray image.
    
    Args:
        image_file: File object from request
        
    Returns:
        dict: Analysis results
    """
    try:
        # Read and preprocess image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Create transforms for detection
        detection_transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create transforms for classification
        classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet-50 expected size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        detection_tensor = detection_transform(image)
        classification_tensor = classification_transform(image)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detection_tensor = detection_tensor.to(device)
        classification_tensor = classification_tensor.to(device)
        
        print("\nProcessing Image:")
        print(f"Detection tensor shape: {detection_tensor.shape}")
        print(f"Classification tensor shape: {classification_tensor.shape}")
        
        # Get detection results
        with torch.no_grad():
            detections = detector([detection_tensor])
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            
            # Filter by confidence
            confidence_threshold = 0.5
            high_conf_indices = scores > confidence_threshold
            filtered_boxes = boxes[high_conf_indices].tolist()
            filtered_scores = scores[high_conf_indices].tolist()
            
            print(f"\nDetection Results:")
            print(f"Found {len(filtered_boxes)} regions with confidence > {confidence_threshold}")
            
            # Get classification
            class_pred, class_prob = classifier.predict(classification_tensor.unsqueeze(0))
            severity_map = ['normal', 'superficial', 'medium', 'deep']
            severity = severity_map[class_pred.item()]
            confidence = float(class_prob[0][class_pred].item())
            
            print(f"\nClassification Results:")
            print(f"Predicted severity: {severity}")
            print(f"Confidence: {confidence:.4f}")
            
            # Get recommendations
            recommendations = recommender.get_recommendations('caries', severity, confidence)
            
            return {
                'detections': {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'num_caries': len(filtered_boxes)
                },
                'severity': severity,
                'confidence': confidence,
                'recommendations': recommendations
            }
            
    except Exception as e:
        log_error("Error analyzing image", e)
        raise

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image."""
    try:
        # Check if file exists
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image'}), 400
            
        # Analyze image
        results = analyze_image(file)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test/<severity>')
def test_sample(severity):
    """Test with sample image."""
    try:
        sample_path = f'sample-test/{severity}_1.png'
        if not os.path.exists(sample_path):
            return jsonify({'error': 'Sample image not found'}), 404
            
        with open(sample_path, 'rb') as f:
            results = analyze_image(f)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Exiting...") 