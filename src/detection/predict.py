import torch
from PIL import Image
import torchvision.transforms as transforms
from .model import DentalCariesDetector

def predict_image(image, model=None):
    """
    Predict caries regions in a dental X-ray image.
    
    Args:
        image (PIL.Image): Input image
        model (torch.nn.Module, optional): Pre-loaded model
        
    Returns:
        dict: Detection results including bounding boxes and masks
    """
    if model is None:
        model = DentalCariesDetector()
        model.load_state_dict(torch.load('models/detection/model.pth'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
        
    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()
    
    # Filter predictions by confidence threshold
    confidence_threshold = 0.5
    high_conf_indices = scores > confidence_threshold
    
    filtered_boxes = boxes[high_conf_indices]
    filtered_masks = masks[high_conf_indices]
    filtered_scores = scores[high_conf_indices]
    
    return {
        'num_caries': len(filtered_boxes),
        'boxes': filtered_boxes.tolist(),
        'scores': filtered_scores.tolist(),
        'masks': filtered_masks.tolist()
    } 