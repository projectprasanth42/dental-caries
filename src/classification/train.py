import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.classification.model import DentalCariesClassifier
from src.classification.utils import DentalClassificationDataset, get_class_weights

def train_model(data_dir, num_epochs=1, batch_size=8, learning_rate=0.001, max_samples=None, device=None):
    """
    Train the dental caries classification model.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        max_samples (int, optional): Maximum number of samples to use for training
        device (torch.device): Device to use for training
    """
    try:
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model
        model = DentalCariesClassifier(num_classes=3)  # superficial, medium, deep
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        print("Model initialized successfully")
        
        # Define transforms with error handling
        try:
            train_transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            print("Transforms defined successfully")
        except Exception as e:
            print(f"Error defining transforms: {str(e)}")
            raise
        
        # Create dataset with error handling
        try:
            train_dataset = DentalClassificationDataset(
                os.path.join(data_dir, 'three_level_classification/train'),
                transform=train_transform,
                max_samples=max_samples
            )
            
            if len(train_dataset) == 0:
                raise ValueError("No training samples found!")
                
            print(f"Dataset created successfully with {len(train_dataset)} samples")
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            raise
        
        # Create dataloader with error handling
        try:
            # Calculate optimal number of workers
            num_workers = min(4, os.cpu_count() or 1)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            print(f"DataLoader created successfully with {len(train_loader)} batches")
            print(f"Using {num_workers} workers for data loading")
        except Exception as e:
            print(f"Error creating DataLoader: {str(e)}")
            raise
        
        # Calculate class weights for balanced loss
        try:
            class_weights = get_class_weights(
                os.path.join(data_dir, 'three_level_classification/train')
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Loss function initialized with class weights")
        except Exception as e:
            print(f"Error calculating class weights: {str(e)}")
            print("Falling back to unweighted loss")
            criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer and scaler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        
        # Training loop
        print("\nStarting training...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                try:
                    # Move data to device
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    accuracy = 100 * correct / total
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                        'accuracy': f"{accuracy:.2f}%",
                        'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB"
                    })
                    
                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} completed:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Final accuracy: {accuracy:.2f}%")
            
            # Save checkpoint if loss improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_dir = 'models/classification'
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'accuracy': accuracy,
                        'scaler': scaler.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
                    print(f"Saved checkpoint with loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
        
        # Save final model
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'final_loss': best_loss,
                'final_accuracy': accuracy,
                'scaler': scaler.state_dict(),
            }, 'models/classification/model.pth')
            print("Training completed! Final model saved.")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train_model('data/dental_ai_dataset_v4_augmented') 