from flask import Flask, request, jsonify, render_template
import torch
import os
import sys

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model import DentalCariesDetector
from classification.model import DentalCariesClassifier
from recommendation.model import DentalRecommendationSystem

app = Flask(__name__)

# Initialize models
detector = None
classifier = None
recommender = None

def load_models():
    """Load all models from checkpoints."""
    global detector, classifier, recommender
    
    try:
        detector = DentalCariesDetector()
        detector.load_state_dict(torch.load('models/detection/model.pth'))
        detector.eval()
        
        classifier = DentalCariesClassifier()
        classifier.load_state_dict(torch.load('models/classification/model.pth'))
        classifier.eval()
        
        recommender = DentalRecommendationSystem()
        recommender.load_state_dict(torch.load('models/recommendation/model.pth'))
        recommender.eval()
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False
    return True

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze dental X-ray image and provide recommendations.
    
    Expects:
        - Image file in request.files['image']
    
    Returns:
        JSON with detection results, classification, and recommendations
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        # Get the image file
        image_file = request.files['image']
        
        # Process image and get predictions
        # TODO: Implement image preprocessing
        
        # Get detection results
        detections = detector.predict([image])
        
        # Get classification results
        class_pred, class_prob = classifier.predict(image)
        
        # Get severity level
        severity_map = {0: 'superficial', 1: 'medium', 2: 'deep'}
        severity = severity_map[class_pred.item()]
        confidence = class_prob[0][class_pred].item()
        
        # Get recommendations
        recommendations = recommender.get_recommendations('caries', severity, confidence)
        
        return jsonify({
            'detections': detections,
            'severity': severity,
            'confidence': confidence,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Load models before starting the server
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Exiting...") 