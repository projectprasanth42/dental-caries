import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class DentalCariesDetector(nn.Module):
    """Dental caries detection model based on Mask R-CNN."""
    
    def __init__(self, num_classes=2):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of classes (including background)
        """
        super().__init__()
        
        # Load pre-trained model
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images (List[Tensor]): Images to be processed
            targets (List[Dict], optional): Ground-truth boxes and labels
            
        Returns:
            loss_dict (Dict) or detections (List[Dict]): During training, returns
            a dictionary of losses. During inference, returns a list of detections.
        """
        return self.model(images, targets)
    
    def predict(self, images):
        """
        Make predictions on a batch of images.
        
        Args:
            images (list[Tensor]): Batch of images
            
        Returns:
            list[Dict[str, Tensor]]: List of predictions for each image
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions 