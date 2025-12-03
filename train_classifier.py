"""
Train classifier for steel defect classification
Supports baseline (real data only) and augmented (real + GAN) modes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from models.classifier import SteelDefectClassifier
from utils.helpers import save_checkpoint, load_checkpoint, ensure_dir, set_seed, get_class_weights


class SteelDefectDataset(Dataset):
    """Dataset for steel defect classification"""
    
    def __init__(self, csv_path, image_base_dir, augmented_dirs=None, transform=None):
        """
        Args:
            csv_path: Path to CSV file with image_id and class labels
            image_base_dir: Base directory containing original images
            augmented_dirs: Dictionary mapping class indices to directories of synthetic images
            transform: Data augmentation transforms
        """
        self.df = pd.read_csv(csv_path)
        self.image_base_dir = image_base_dir
        self.augmented_dirs = augmented_dirs or {}
        self.transform = transform
        
        # Add synthetic samples if in augmented mode
        if self.augmented_dirs:
            self._add_synthetic_samples()
    
    def _add_synthetic_samples(self):
        """Add synthetic samples to the dataset"""
        synthetic_data = []
        
        for class_idx, synth_dir in self.augmented_dirs.items():
            if os.path.exists(synth_dir):
                synth_images = [f for f in os.listdir(synth_dir) if f.endswith('.jpg')]
                class_name = CLASS_NAMES[class_idx]
                
                for img_name in synth_images:
                    synthetic_data.append({
                        'image_id': os.path.join(synth_dir, img_name),  # Full path for synthetic
                        'class': class_idx,
                        'class_name': class_name,
                        'is_synthetic': True
                    })
                
                print(f"  Added {len(synth_images)} synthetic {class_name} images")
        
        if synthetic_data:
            synthetic_df = pd.DataFrame(synthetic_data)
            self.df['is_synthetic'] = False
            self.df = pd.concat([self.df, synthetic_df], ignore_index=True)
            print(f"Total dataset size after augmentation: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
      # Load image
        if row.get('is_synthetic', False):
            img_path = row['image_id']  # Full path already
        else:
            img_path = os.path.join(self.image_base_dir, row['image_id'])
        
        image = Image.open(img_path).convert('RGB')
        label = int(row['class'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_classifier(mode='baseline'):
    """
    Train steel defect classifier
    
    Args:
        mode: 'baseline' (real data only) or 'augmented' (real + GAN data)
    """
    print("="*70)
    print(f"TRAINING STEEL DEFECT CLASSIFIER - {mode.upper()} MODE")
    print("="*70)
    
    set_seed(RANDOM_SEED)
    
    # Paths
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, mode)
    ensure_dir(checkpoint_dir)
    
    print(f"\\nConfiguration:")
    print(f"  Mode: {mode}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {CLASSIFIER_EPOCHS}")
    print(f"  Batch size: {CLASSIFIER_BATCH_SIZE}")
    print(f"  Learning rate: {CLASSIFIER_LR}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    # Prepare augmented directories if in augmented mode
    augmented_dirs = {}
    if mode == 'augmented':
        for class_idx in RARE_CLASSES:
            class_name = CLASS_NAMES[class_idx]
            synth_dir = os.path.join(GENERATED_DIR, class_name.lower())
            if os.path.exists(synth_dir):
                augmented_dirs[class_idx] = synth_dir
                print(f"  Will use synthetic {class_name} data from: {synth_dir}")
    
    # Datasets
    print("\\nLoading datasets...")
    train_dataset = SteelDefectDataset(
        os.path.join(PROCESSED_DIR, 'train.csv'),
        TRAIN_IMAGES,
        augmented_dirs=augmented_dirs if mode == 'augmented' else None,
        transform=train_transform
    )
    
    val_dataset = SteelDefectDataset(
        os.path.join(PROCESSED_DIR, 'val.csv'),
        TRAIN_IMAGES,
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, 
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    
    # Model
    model = SteelDefectClassifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print(f"\\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
    class_weights = get_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    print(f"\\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        print(f"\\nEpoch {epoch}/{CLASSIFIER_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, 'best.pth')
            )
            print("✓ Saved best model")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_loss,
        os.path.join(checkpoint_dir, 'final.pth')
    )
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    torch.save(history, os.path.join(checkpoint_dir, 'training_history.pth'))
    
    print(f"\\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"{'='*70}\\n")


def main():
    parser = argparse.ArgumentParser(description='Train steel defect classifier')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'augmented'],
                        help='Training mode: baseline (real only) or augmented (real + GAN)')
    
    args = parser.parse_args()
    
    train_classifier(args.mode)


if __name__ == "__main__":
    main()
