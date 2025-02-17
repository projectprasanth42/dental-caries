import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

def train_model(data_dir, num_epochs=50, batch_size=2, learning_rate=0.0001, max_samples=4):
    """
    Train the dental caries detection model.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        max_samples (int): Maximum number of samples to use for training
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = DentalCariesDetector(num_classes=2)  # background and caries
    model = model.to(device)
    
    # Define transforms
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', 
                               label_fields=['labels'],
                               min_visibility=0.3))
    
    try:
        # Create datasets and dataloaders
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for debugging
        )
        
        print(f"Created DataLoader with {len(train_loader)} batches")
        
        # Initialize optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=learning_rate)
        
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
                    images = [image.to(device) for image in images]  # List of tensors
                    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = model(images, targets)
                    
                    # Calculate total loss
                    losses = sum(loss for loss in loss_dict.values() 
                               if not torch.isnan(loss) and not torch.isinf(loss))
                    
                    # Skip batch if loss is invalid
                    if torch.isnan(losses) or torch.isinf(losses):
                        print(f"Skipping batch due to invalid loss: {losses}")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
                    optimizer.step()
                    
                    # Update metrics
                    loss_value = losses.item()
                    if loss_value > 0:  # Only count positive losses
                        epoch_loss += loss_value
                        valid_batches += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': epoch_loss / valid_batches
                        })
                
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
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))
                    print(f"Saved checkpoint with loss: {best_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = 'models/detection'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss if valid_batches > 0 else float('inf'),
                }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_loss': best_loss
        }, 'models/detection/model.pth')
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    train_model('data/dental_ai_dataset_v4_augmented') 