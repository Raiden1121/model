"""
Generate synthetic defect images using trained GAN
"""
import torch
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from models.generator import Generator
from utils.helpers import ensure_dir, set_seed


def load_generator(defect_type, checkpoint_path=None):
    """Load trained generator model"""
    class_name = CLASS_NAMES[defect_type - 1]
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'gan_type{defect_type}', 'generator_final.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print(f"Please train GAN for Type {defect_type} first!")
        return None
    
    # Load model
    generator = Generator(GAN_LATENT_DIM, IMAGE_CHANNELS).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    print(f"Loaded generator from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return generator


def generate_samples(defect_type, num_samples=None, checkpoint_path=None):
    """
    Generate synthetic images for a specific defect type
    
    Args:
        defect_type: Type of defect (3 or 4)
        num_samples: Number of samples to generate
        checkpoint_path: Path to generator checkpoint (optional)
    """
    print("="*70)
    print(f"GENERATING SYNTHETIC TYPE {defect_type} DEFECT IMAGES")
    print("="*70)
    
    set_seed(RANDOM_SEED)
    
    class_name = CLASS_NAMES[defect_type - 1]
    num_samples = num_samples or NUM_SYNTHETIC_SAMPLES[class_name]
    
    # Output directory
    output_dir = os.path.join(GENERATED_DIR, class_name.lower())
    ensure_dir(output_dir)
    
    print(f"\nConfiguration:")
    print(f"  Defect type: Type {defect_type} ({class_name})")
    print(f"  Number of samples: {num_samples}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {DEVICE}")
    
    # Load generator
    generator = load_generator(defect_type, checkpoint_path)
    if generator is None:
        return
    
    # Generate samples
    print(f"\nGenerating {num_samples} synthetic images...")
    
    batch_size = 32
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    generated_count = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=\"Generating\"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - generated_count)
            
            # Generate random noise
            noise = torch.randn(current_batch_size, GAN_LATENT_DIM, device=DEVICE)
            
            # Generate images
            fake_images = generator(noise)
            fake_images = (fake_images + 1) / 2  # Denormalize to [0, 1]
            
            # Save images
            for i in range(current_batch_size):
                img_name = f\"synthetic_type{defect_type}_{generated_count:05d}.jpg\"
                img_path = os.path.join(output_dir, img_name)
                save_image(fake_images[i], img_path)
                generated_count += 1
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated {generated_count} images")
    print(f"Saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic steel defect images')
    parser.add_argument('--defect_type', type=int, required=True, choices=[3, 4],
                        help='Type of defect to generate (3 or 4)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (default from config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to generator checkpoint (default: use final checkpoint)')
    
    args = parser.parse_args()
    
    generate_samples(args.defect_type, args.num_samples, args.checkpoint)


if __name__ == \"__main__\":
    main()
