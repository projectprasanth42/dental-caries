import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

from src.detection.model import DentalCariesDetector
from src.detection.utils import DentalDataset, collate_fn

def validate_boxes(boxes, image_size=(512, 512), min_size=5):
    """
    Validate and fix bounding boxes.
    
    Args:
        boxes (Tensor or ndarray): Bounding boxes
        image_size (tuple): Image size (height, width)
        min_size (int): Minimum box size
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    
    valid_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Ensure correct ordering of coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Clip to image bounds
        x1 = max(0, min(x1, image_size[1] - 1))
        x2 = max(0, min(x2, image_size[1] - 1))
        y1 = max(0, min(y1, image_size[0] - 1))
        y2 = max(0, min(y2, image_size[0] - 1))
        
        # Ensure minimum size
        if x2 - x1 < min_size:
            x2 = min(x1 + min_size, image_size[1] - 1)
        if y2 - y1 < min_size:
            y2 = min(y1 + min_size, image_size[0] - 1)
        
        # Add box if it's valid
        if x2 > x1 and y2 > y1:
            valid_boxes.append([x1, y1, x2, y2])
    
    if len(valid_boxes) == 0:
        # Return dummy box if no valid boxes
        return torch.tensor([[0, 0, min_size, min_size]], dtype=torch.float32)
    
    return torch.tensor(valid_boxes, dtype=torch.float32)

def train_model(data_dir, num_epochs=50, batch_size=2, learning_rate=0.0001, max_samples=None, device=None):
    """
    Train the dental caries detection model.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        max_samples (int): Maximum number of samples to use for training
        device (torch.device): Device to use for training
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize model
        model = DentalCariesDetector(num_classes=2)  # background and caries
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Define transforms
        train_transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                   label_fields=['labels'],
                                   min_visibility=0.3))
        
        # Create dataset and dataloader
        train_dataset = DentalDataset(
            os.path.join(data_dir, 'detailed_segmentation/train'),
            transform=train_transform
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No training samples found!")
            
        # Limit the number of samples for testing
        if max_samples and max_samples < len(train_dataset):
            print(f"Using {max_samples} samples for testing")
            from torch.utils.data import Subset
            indices = list(range(max_samples))
            train_dataset = Subset(train_dataset, indices)
        
        # Calculate optimal number of workers
        num_workers = min(4, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Created DataLoader with {len(train_loader)} batches")
        print(f"Using {num_workers} workers for data loading")
        
        # Initialize optimizer with gradient clipping
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        
        # Training loop
        print("Starting training...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for images, targets in progress_bar:
                try:
                    # Move data to device
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in targets]
                    
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass with gradient scaling
                    optimizer.zero_grad()
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update metrics
                    loss_value = losses.item()
                    if not torch.isnan(torch.tensor(loss_value)):
                        epoch_loss += loss_value
                        valid_batches += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': epoch_loss / valid_batches,
                            'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB"
                        })
                    
                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error in batch: {str(e)}")
                    continue
            
            # Calculate average loss
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
                # Save checkpoint if loss improved
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_dir = 'models/detection'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model state
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'scaler': scaler.state_dict(),  # Save scaler state
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_loss': best_loss,
            'scaler': scaler.state_dict(),
        }, 'models/detection/model.pth')
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train_model('data/dental_ai_dataset_v4_augmented') 