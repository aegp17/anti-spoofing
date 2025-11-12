"""
Training script for MobileNetV2 document vs selfie classifier.

Usage:
    python train_mobilenet.py --train-dir data/train --val-dir data/val \
        --epochs 30 --output models/model_mobilenet_v2.pt
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ImageFolder
from torchvision import models, transforms
from pathlib import Path
from datetime import datetime


class MobileNetV2Classifier:
    """MobileNetV2-based classifier for document vs selfie."""
    
    def __init__(self, device: torch.device = None):
        """Initialize classifier."""
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model()
        self.best_val_loss = float('inf')
    
    def _build_model(self) -> nn.Module:
        """Build MobileNetV2 model with binary classification head."""
        # Load pretrained MobileNetV2
        model = models.mobilenet_v2(pretrained=True)
        
        # Freeze earlier layers
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Replace classification head for binary classification
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)  # Binary output
        )
        
        return model.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, optimizer, criterion) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.unsqueeze(1).float().to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion) -> tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.unsqueeze(1).float().to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                correct += (predictions == labels.long()).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path: str) -> None:
        """Save model as TorchScript."""
        script_model = torch.jit.script(self.model)
        script_model.save(path)
        print(f"✓ Model saved to {path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 1e-3
    ) -> dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with training history
        """
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10}")
        print("-" * 50)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f"{epoch+1:<6} {train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<10.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  → Best model updated (val_loss: {val_loss:.4f})")
        
        return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for document vs selfie classification"
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        help="Path to validation dataset directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--output",
        default="models/model_mobilenet_v2.pt",
        help="Output path for trained model"
    )
    
    args = parser.parse_args()
    
    # Verify directories exist
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("📊 MobileNetV2 Document vs Selfie Classifier")
    print(f"{'='*50}")
    print(f"Train data: {train_dir}")
    print(f"Val data: {val_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {output_path}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)
    
    print(f"  ✓ Training samples: {len(train_dataset)}")
    print(f"  ✓ Validation samples: {len(val_dataset)}")
    print(f"  ✓ Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize and train
    print("\n🚀 Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    classifier = MobileNetV2Classifier(device=device)
    
    history = classifier.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Save model
    print(f"\n💾 Saving model...")
    classifier.save_model(str(output_path))
    
    print(f"\n✅ Training complete!")
    print(f"  Best validation loss: {classifier.best_val_loss:.4f}")
    print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()

