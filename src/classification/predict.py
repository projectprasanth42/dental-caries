import torch
import torchvision.transforms as transforms
from PIL import Image
from .model import DentalCariesClassifier

def predict_image(image, model=None):
    """
    Predict the severity of dental caries in an X-ray image.
    
    Args:
        image (PIL.Image): Input image
        model (torch.nn.Module, optional): Pre-loaded model
        
    Returns:
        str: Predicted severity class
    """
    if model is None:
        model = DentalCariesClassifier()
        model.load_state_dict(torch.load('models/classification/model.pth'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Map class index to severity label
    severity_classes = ['normal', 'superficial', 'medium', 'deep']
    predicted_severity = severity_classes[predicted.item()]
    
    return predicted_severity 