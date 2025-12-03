"""
Data Explorer - Analyze and visualize the Severstal Steel Defect Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.helpers import rle_decode, ensure_dir

def load_and_analyze_data():
    """Load and analyze the training data"""
    print("="*60)
    print("SEVERSTAL STEEL DEFECT DETECTION - DATA ANALYSIS")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    # Get unique images
    unique_images = df['ImageId'].nunique()
    print(f"\nTotal unique images: {unique_images}")
    
    # Analyze defect distribution
    print("\n" + "="*60)
    print("DEFECT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count defects per class
    defect_counts = {}
    for class_id in [1, 2, 3, 4]:
        class_df = df[df['ClassId'] == class_id]
        # Count non-null encoded pixels (actual defects)
        defect_count = class_df['EncodedPixels'].notna().sum()
        defect_counts[f'Type_{class_id}'] = defect_count
    
    # Count images with no defects
    # An image has no defect if it doesn't appear in the CSV or all its entries are NaN
    images_with_defects = df[df['EncodedPixels'].notna()]['ImageId'].unique()
    all_images = set(os.listdir(TRAIN_IMAGES))
    all_images = {img for img in all_images if img.endswith('.jpg')}
    images_without_defects = all_images - set(images_with_defects)
    defect_counts['No_Defect'] = len(images_without_defects)
    
    print("\nDefect counts:")
    for defect_type, count in defect_counts.items():
        print(f"  {defect_type}: {count}")
    
    total_samples = sum(defect_counts.values())
    print(f"\nTotal samples: {total_samples}")
    
    # Calculate imbalance ratio
    max_count = max(defect_counts.values())
    min_count = min(defect_counts.values())
    print(f"Imbalance ratio: {max_count/min_count:.2f}:1")
    
    return df, defect_counts


def visualize_distribution(defect_counts):
    """Visualize defect distribution"""
    ensure_dir(os.path.join(RESULTS_DIR, 'plots'))
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    categories = list(defect_counts.keys())
    counts = list(defect_counts.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = plt.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Defect Type', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.title('Steel Defect Distribution (Highly Imbalanced)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'plots', 'defect_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot: {save_path}")
    plt.close()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Defect Distribution (Percentage)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'plots', 'defect_distribution_pie.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pie chart: {save_path}")
    plt.close()


def visualize_samples(df):
    """Visualize sample images from each defect type"""
    ensure_dir(os.path.join(RESULTS_DIR, 'plots'))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Sample Steel Defect Images', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, class_id in enumerate([1, 2, 3, 4]):
        # Get a sample image for this class
        class_df = df[(df['ClassId'] == class_id) & (df['EncodedPixels'].notna())]
        if len(class_df) > 0:
            sample = class_df.iloc[0]
            img_path = os.path.join(TRAIN_IMAGES, sample['ImageId'])
            
            if os.path.exists(img_path):
                # Load and display original image
                img = Image.open(img_path)
                axes[0, idx].imshow(img, cmap='gray')
                axes[0, idx].set_title(f'Type {class_id} - Original', fontweight='bold')
                axes[0, idx].axis('off')
                
                # Decode and display mask
                mask = rle_decode(sample['EncodedPixels'])
                axes[1, idx].imshow(mask, cmap='Reds', alpha=0.8)
                axes[1, idx].set_title(f'Type {class_id} - Mask', fontweight='bold')
                axes[1, idx].axis('off')
    
    # Show a no-defect sample
    all_images = os.listdir(TRAIN_IMAGES)
    images_with_defects = df[df['EncodedPixels'].notna()]['ImageId'].unique()
    no_defect_images = [img for img in all_images if img not in images_with_defects and img.endswith('.jpg')]
    
    if no_defect_images:
        no_defect_img = no_defect_images[0]
        img_path = os.path.join(TRAIN_IMAGES, no_defect_img)
        img = Image.open(img_path)
        axes[0, 4].imshow(img, cmap='gray')
        axes[0, 4].set_title('No Defect - Original', fontweight='bold')
        axes[0, 4].axis('off')
        
        axes[1, 4].imshow(np.zeros((256, 1600)), cmap='gray')
        axes[1, 4].set_title('No Defect - Mask', fontweight='bold')
        axes[1, 4].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'sample_defects.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample images: {save_path}")
    plt.close()


def main():
    """Main function"""
    # Load and analyze data
    df, defect_counts = load_and_analyze_data()
    
    # Visualize distribution
    visualize_distribution(defect_counts)
    
    # Visualize samples
    visualize_samples(df)
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}/plots/")
    print("\nKey Findings:")
    print("  - Dataset is highly imbalanced")
    print("  - Type 2 (247 samples, 1.9%) and Type 4 (801 samples, 6.2%) are the rarest defects")
    print("  - GAN will focus on generating Type 2 and Type 4 samples")
    print("="*60)


if __name__ == "__main__":
    main()
