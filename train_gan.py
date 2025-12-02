"""
Train DCGAN for generating synthetic steel defect images
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from models.generator import Generator
from models.discriminator import Discriminator
from utils.helpers import save_checkpoint, ensure_dir, set_seed


class SteelDefectDataset(Dataset):
    """Dataset for loading steel defect images"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def train_gan(defect_type, epochs=None):
    """
    Train DCGAN for a specific defect type
    
    Args:
        defect_type: Type of defect (3 or 4)
        epochs: Number of training epochs (default from config)
    """
    print("="*70)
    print(f"TRAINING DCGAN FOR TYPE {defect_type} DEFECTS")
    print("="*70)
    
    set_seed(RANDOM_SEED)
    epochs = epochs or GAN_EPOCHS
    
    # Paths
    class_name = CLASS_NAMES[defect_type - 1]  # Convert to 0-indexed
    data_dir = os.path.join(PROCESSED_DIR, f'gan_{class_name.lower()}')
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, f'gan_type{defect_type}')
    sample_dir = os.path.join(RESULTS_DIR, 'gan_samples', f'type{defect_type}')
    
    ensure_dir(checkpoint_dir)
    ensure_dir(sample_dir)
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run prepare_dataset.py first!")
        return
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Sample directory: {sample_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {GAN_BATCH_SIZE}")
    print(f"  Learning rate: {GAN_LR}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((GAN_IMAGE_SIZE, GAN_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*IMAGE_CHANNELS, [0.5]*IMAGE_CHANNELS)
    ])
    
    # Dataset and DataLoader
    dataset = SteelDefectDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"\nDataset size: {len(dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Models
    generator = Generator(GAN_LATENT_DIM, IMAGE_CHANNELS).to(DEVICE)
    discriminator = Discriminator(IMAGE_CHANNELS).to(DEVICE)
    
    print(f"\nModel parameters:")
    print(f"  Generator: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, GAN_BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, GAN_BETA2))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, GAN_LATENT_DIM, device=DEVICE)
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    G_losses = []
    D_losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_G_loss = 0
        epoch_D_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for i, real_images in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(DEVICE)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
            
            # =================== Train Discriminator ===================
            optimizer_D.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, GAN_LATENT_DIM, device=DEVICE)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # =================== Train Generator ===================
            optimizer_G.zero_grad()
            
            # Generate fake images again and get discriminator output
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)  # Generator wants D to think images are real
            
            g_loss.backward()
            optimizer_G.step()
            
            # Update progress bar
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # Average losses
        avg_G_loss = epoch_G_loss / len(dataloader)
        avg_D_loss = epoch_D_loss / len(dataloader)
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        print(f"Epoch {epoch}/{epochs} - G_loss: {avg_G_loss:.4f}, D_loss: {avg_D_loss:.4f}")
        
        # Save samples every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                fake_samples = (fake_samples + 1) / 2  # Denormalize to [0, 1]
                grid = make_grid(fake_samples, nrow=8, normalize=False)
                save_image(grid, os.path.join(sample_dir, f'epoch_{epoch:03d}.png'))
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(
                generator, optimizer_G, epoch, avg_G_loss,
                os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
            )
            save_checkpoint(
                discriminator, optimizer_D, epoch, avg_D_loss,
                os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth')
            )
    
    # Save final models
    save_checkpoint(
        generator, optimizer_G, epochs, G_losses[-1],
        os.path.join(checkpoint_dir, 'generator_final.pth')
    )
    save_checkpoint(
        discriminator, optimizer_D, epochs, D_losses[-1],
        os.path.join(checkpoint_dir, 'discriminator_final.pth')
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'GAN Training Losses - Type {defect_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(sample_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Samples saved to: {sample_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN for steel defect generation')
    parser.add_argument('--defect_type', type=int, required=True, choices=[3, 4],
                        help='Type of defect to generate (3 or 4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of epochs (default: {GAN_EPOCHS})')
    
    args = parser.parse_args()
    
    train_gan(args.defect_type, args.epochs)


if __name__ == "__main__":
    main()
