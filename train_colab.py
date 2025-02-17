import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def setup_colab():
    """Setup Google Colab environment."""
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Clone repository if not exists
    if not os.path.exists('dental-caries-project'):
        !git clone https://github.com/your-repo/dental-caries-project.git
        os.chdir('dental-caries-project')
    
    # Install requirements
    !pip install -r requirements_colab.txt
    
    # Add project to Python path
    project_path = os.getcwd()
    if project_path not in sys.path:
        sys.path.append(project_path)

def train_model_gpu(data_dir, num_epochs=50, batch_size=32, learning_rate=0.0001, validation_split=0.2):
    """
    Train the dental caries classification model on GPU.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        validation_split (float): Fraction of data to use for validation
    """
    try:
        # Set device and enable GPU optimizations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")
        
        # Initialize model with 4 classes
        from src.classification.model import DentalCariesClassifier
        model = DentalCariesClassifier(num_classes=4)
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        print("Model initialized successfully")
        
        # Define transforms with error handling
        try:
            train_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(224, 224),
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
            from src.classification.utils import DentalClassificationDataset, get_class_weights
            full_dataset = DentalClassificationDataset(
                os.path.join(data_dir, 'three_level_classification/train'),
                transform=train_transform
            )
            
            if len(full_dataset) == 0:
                raise ValueError("No training samples found!")
            
            # Split into train and validation sets
            val_size = int(validation_split * len(full_dataset))
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            print(f"\nDataset split:")
            print(f"Total samples: {len(full_dataset)}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            
            # Create dataloaders with GPU pinned memory
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,  # Adjust based on CPU cores
                pin_memory=True,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            
            print(f"\nDataLoaders created successfully:")
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            raise
        
        # Calculate class weights for balanced loss
        try:
            class_weights = get_class_weights(
                os.path.join(data_dir, 'three_level_classification/train')
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("\nLoss function initialized with class weights:")
            for i, w in enumerate(['normal', 'superficial', 'medium', 'deep']):
                print(f"{w}: {class_weights[i]:.4f}")
        except Exception as e:
            print(f"Error calculating class weights: {str(e)}")
            print("Falling back to unweighted loss")
            criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer with weight decay and gradient clipping
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # Training loop with mixed precision
        print("\nStarting training...")
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                try:
                    # Move data to device
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Mixed precision forward pass
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_loss += loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{train_loss/(batch_idx+1):.4f}",
                        'accuracy': f"{100.*train_correct/train_total:.2f}%"
                    })
                    
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    try:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                    except Exception as e:
                        print(f"Error in validation: {str(e)}")
                        continue
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                
                checkpoint_dir = 'models/classification'
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model to Google Drive
                save_path = os.path.join('/content/drive/MyDrive/dental_project/models', 
                                       'best_model.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'num_classes': 4,
                    'class_mapping': ['normal', 'superficial', 'medium', 'deep']
                }, save_path)
                
                print(f"Saved checkpoint with accuracy: {best_val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    # This will only run when executed directly, not in Colab
    print("Please run this script in Google Colab") 