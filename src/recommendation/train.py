import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json

from src.recommendation.model import DentalRecommendationSystem
from src.recommendation.utils import RecommendationDataset, generate_training_data

def train_model(data_dir, num_epochs=1, batch_size=4, learning_rate=0.001):
    """
    Train the dental recommendation system.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Generate training data if not exists
        train_data_path = os.path.join(data_dir, 'recommendation_train_data.json')
        try:
            if not os.path.exists(train_data_path):
                print("Generating training data...")
                train_data = generate_training_data(data_dir)
                os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
                with open(train_data_path, 'w') as f:
                    json.dump(train_data, f)
                print(f"Training data saved to {train_data_path}")
            else:
                print("Loading existing training data...")
                with open(train_data_path, 'r') as f:
                    train_data = json.load(f)
        except Exception as e:
            print(f"Error handling training data: {str(e)}")
            raise
        
        # Initialize model
        try:
            model = DentalRecommendationSystem()
            model = model.to(device)
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
        
        # Create dataset
        try:
            train_dataset = RecommendationDataset(train_data)
            if len(train_dataset) == 0:
                raise ValueError("No training samples found!")
            print(f"Dataset created successfully with {len(train_dataset)} samples")
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            raise
        
        # Create dataloader
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True if torch.cuda.is_available() else False
            )
            print(f"DataLoader created successfully with {len(train_loader)} batches")
        except Exception as e:
            print(f"Error creating DataLoader: {str(e)}")
            raise
        
        # Initialize loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print("\nStarting training...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                try:
                    # Move data to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    predictions = (outputs > 0.5).float()
                    correct_predictions += (predictions == targets).sum().item()
                    total_predictions += targets.numel()
                    accuracy = 100 * correct_predictions / total_predictions
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    batch_loss = epoch_loss / (batch_idx + 1)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{batch_loss:.4f}",
                        'accuracy': f"{accuracy:.2f}%"
                    })
                    
                    # Print batch statistics
                    if (batch_idx + 1) % 10 == 0:
                        print(f"\nBatch {batch_idx + 1}/{len(train_loader)}:")
                        print(f"Loss: {loss.item():.4f}")
                        print(f"Accuracy: {accuracy:.2f}%")
                        print("Prediction distribution:", torch.bincount(predictions.long().flatten()))
                    
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
                checkpoint_dir = 'models/recommendation'
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'accuracy': accuracy
                    }, os.path.join(checkpoint_dir, 'best_model.pth'))
                    print(f"Saved checkpoint with loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
        
        # Save final model
        try:
            torch.save(model.state_dict(), 'models/recommendation/model.pth')
            print("Training completed! Final model saved.")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    train_model('data/dental_ai_dataset_v4_augmented') 