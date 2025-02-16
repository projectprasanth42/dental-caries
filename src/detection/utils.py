import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

class DentalDataset(Dataset):
    """Dataset class for dental X-ray images and their segmentation masks."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing images and masks
            transform (callable, optional): Optional transform to be applied
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image files
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            raise ValueError(f"Images or masks directory not found in {data_dir}")
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                 if f.endswith('.png')])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {self.images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        try:
            # Load image
            img_path = os.path.join(self.images_dir, self.image_files[idx])
            mask_path = os.path.join(self.masks_dir, self.image_files[idx])
            
            # Verify files exist
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                raise FileNotFoundError(f"Image or mask file not found: {self.image_files[idx]}")
            
            # Read image and mask
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            
            # Convert mask to binary (0 or 1)
            mask = (mask > 0).astype(np.uint8)
            
            # Ensure image and mask have same dimensions
            if image.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Get bounding boxes from mask
            boxes, labels = self._get_boxes_from_mask(mask)
            
            # Create target dictionary
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': torch.as_tensor(mask[None], dtype=torch.uint8)  # Add channel dimension
            }
            
            # Apply transformations
            if self.transform is not None:
                # Handle empty boxes
                if len(boxes) == 0:
                    boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                    labels = torch.tensor([0], dtype=torch.int64)
                
                # Convert boxes to numpy for albumentations
                if isinstance(boxes, torch.Tensor):
                    boxes_np = boxes.numpy()
                else:
                    boxes_np = boxes
                
                transformed = self.transform(image=image, 
                                          masks=[mask], 
                                          bboxes=boxes_np,
                                          labels=labels.numpy())
                
                image = transformed['image']  # This is now a tensor [C, H, W] in [0, 1]
                if len(transformed['masks']) > 0:
                    mask = np.array(transformed['masks'][0])  # Convert to numpy array
                    # Ensure mask is binary
                    mask = (mask > 0).astype(np.uint8)
                    target['masks'] = torch.as_tensor(mask[None], dtype=torch.uint8)
                if len(transformed['bboxes']) > 0:
                    target['boxes'] = torch.as_tensor(transformed['bboxes'], 
                                                    dtype=torch.float32)
                    target['labels'] = torch.as_tensor(transformed['labels'], 
                                                     dtype=torch.int64)
                else:
                    # If no valid boxes after transform, create dummy box
                    target['boxes'] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                    target['labels'] = torch.tensor([0], dtype=torch.int64)
            else:
                # If no transform, manually convert image to tensor and normalize
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                image = torch.clamp(image, 0, 1)  # Ensure values are in [0, 1]
            
            return image, target
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample
            return self._get_dummy_sample()
    
    def _get_boxes_from_mask(self, mask):
        """Extract bounding boxes from segmentation mask."""
        try:
            # Find connected components in binary mask
            num_labels, labels = cv2.connectedComponents(mask)
            
            boxes = []
            box_labels = []
            
            # Skip background (label 0)
            for label in range(1, num_labels):
                # Get points for current label
                points = np.where(labels == label)
                if len(points[0]) == 0:
                    continue
                    
                # Get bounding box coordinates
                y_min, x_min = np.min(points[0]), np.min(points[1])
                y_max, x_max = np.max(points[0]), np.max(points[1])
                
                # Add small padding to ensure the box contains the entire object
                x_min = max(0, x_min - 2)
                y_min = max(0, y_min - 2)
                x_max = min(mask.shape[1] - 1, x_max + 2)
                y_max = min(mask.shape[0] - 1, y_max + 2)
                
                # Add box if it's large enough (at least 5x5 pixels)
                if (x_max - x_min) >= 5 and (y_max - y_min) >= 5:
                    boxes.append([x_min, y_min, x_max, y_max])
                    box_labels.append(1)  # 1 for caries
            
            if len(boxes) == 0:
                # Return dummy box if no valid boxes found
                return torch.tensor([[0, 0, 1, 1]], dtype=torch.float32), \
                       torch.tensor([0], dtype=torch.int64)
                
            return torch.as_tensor(boxes, dtype=torch.float32), \
                   torch.as_tensor(box_labels, dtype=torch.int64)
                   
        except Exception as e:
            print(f"Error extracting boxes from mask: {str(e)}")
            return torch.tensor([[0, 0, 1, 1]], dtype=torch.float32), \
                   torch.tensor([0], dtype=torch.int64)
    
    def _get_dummy_sample(self):
        """Create a dummy sample for error cases."""
        image = torch.zeros((3, 512, 512), dtype=torch.float32)  # Float tensor in [0, 1]
        target = {
            'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.int64),
            'masks': torch.zeros((1, 512, 512), dtype=torch.uint8)  # Add channel dimension
        }
        return image, target

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Needed because each image may have a different number of objects.
    
    Args:
        batch (list): List of tuples (image, target)
        
    Returns:
        tuple: Tuple of (images, targets) where:
            - images is a list of tensors
            - targets is a list of dictionaries
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets

def preprocess_image(image_path, size=(512, 512)):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        size (tuple): Target size for resizing
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return torch.zeros((3, size[0], size[1]), dtype=torch.float32)

def visualize_predictions(image, predictions, threshold=0.5):
    """
    Visualize detection and segmentation predictions.
    
    Args:
        image (torch.Tensor): Input image
        predictions (dict): Model predictions
        threshold (float): Confidence threshold
        
    Returns:
        numpy.ndarray: Visualization image with predictions
    """
    try:
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        # Create visualization image
        vis_image = image.copy()
        
        # Draw predictions
        if all(k in predictions for k in ['boxes', 'scores', 'masks']):
            for box, score, mask in zip(predictions['boxes'], 
                                      predictions['scores'],
                                      predictions['masks']):
                if score > threshold:
                    # Draw box
                    box = box.numpy().astype(np.int32)
                    cv2.rectangle(vis_image, 
                                (box[0], box[1]), 
                                (box[2], box[3]),
                                (255, 0, 0), 2)
                    
                    # Draw mask
                    if isinstance(mask, torch.Tensor):
                        mask = mask.numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    vis_image[mask > 0] = vis_image[mask > 0] * 0.5 + \
                                        np.array([255, 0, 0]) * 0.5
        
        return vis_image
        
    except Exception as e:
        print(f"Error visualizing predictions: {str(e)}")
        return image 