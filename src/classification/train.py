import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.classification.model import DentalCariesClassifier
from src.classification.utils import DentalClassificationDataset, get_class_weights

def train_model(data_dir, num_epochs=50, batch_size=16, learning_rate=0.0001, validation_split=0.2):
    """
    Train the dental caries classification model.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        validation_split (float): Fraction of data to use for validation
    """
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model with 4 classes
        model = DentalCariesClassifier(num_classes=4)  # normal, superficial, medium, deep
        model = model.to(device)
        print("Model initialized successfully")
        
        # Define transforms with error handling
        try:
            train_transform = A.Compose([
                A.Resize(224, 224),  # ResNet-50 expected size
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
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
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
        
        # Initialize optimizer with weight decay for regularization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # Training loop
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
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
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
                        images = images.to(device)
                        labels = labels.to(device)
                        
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
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'num_classes': 4,
                    'class_mapping': ['normal', 'superficial', 'medium', 'deep']
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                
                print(f"Saved checkpoint with accuracy: {best_val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    train_model('dental_ai_dataset_v4_augmented') 