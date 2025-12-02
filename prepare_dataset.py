"""
Prepare dataset splits and organize data for training
"""
import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.helpers import ensure_dir, set_seed, rle_decode

def prepare_classification_data():
    """Prepare data for classification task"""
    print("="*60)
    print("PREPARING CLASSIFICATION DATASET")
    print("="*60)
    
    set_seed(RANDOM_SEED)
    
    # Load CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} entries from train.csv")
    
    # Create class labels for each image
    image_labels = {}
    
    # Process images with defects
    for class_id in [1, 2, 3, 4]:
        class_df = df[(df['ClassId'] == class_id) & (df['EncodedPixels'].notna())]
        for img_id in class_df['ImageId'].unique():
            if img_id not in image_labels:
                image_labels[img_id] = class_id - 1  # 0-indexed (0,1,2,3)
    
    # Process images without defects
    all_images = set(os.listdir(TRAIN_IMAGES))
    all_images = {img for img in all_images if img.endswith('.jpg')}
    images_with_defects = set(image_labels.keys())
    images_without_defects = all_images - images_with_defects
    
    for img_id in images_without_defects:
        image_labels[img_id] = 4  # No defect class (index 4)
    
    print(f"\nTotal images classified: {len(image_labels)}")
    
    # Count per class
    class_counts = {}
    for class_idx in range(5):
        count = sum(1 for label in image_labels.values() if label == class_idx)
        class_counts[CLASS_NAMES[class_idx]] = count
        print(f"  {CLASS_NAMES[class_idx]}: {count}")
    
    # Create DataFrame
    data = []
    for img_id, label in image_labels.items():
        data.append({
            'image_id': img_id,
            'class': label,
            'class_name': CLASS_NAMES[label]
        })
    
    full_df = pd.DataFrame(data)
    
    # Split data: train/val/test
    train_df, test_df = train_test_split(
        full_df, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=full_df['class']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), 
        random_state=RANDOM_SEED, stratify=train_df['class']
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(full_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(full_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(full_df)*100:.1f}%)")
    
    # Save to CSV
    ensure_dir(PROCESSED_DIR)
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    print(f"\nSaved splits to: {PROCESSED_DIR}")
    
    return train_df, val_df, test_df


def prepare_gan_data(train_df):
    """Prepare data for GAN training (rare classes only)"""
    print("\n" + "="*60)
    print("PREPARING GAN TRAINING DATASET")
    print("="*60)
    
    for class_idx in RARE_CLASSES:
        class_name = CLASS_NAMES[class_idx]
        print(f"\nPreparing {class_name} data for GAN training...")
        
        # Filter data for this class
        class_df = train_df[train_df['class'] == class_idx]
        print(f"  Found {len(class_df)} {class_name} images")
        
        # Create directory
        gan_data_dir = os.path.join(PROCESSED_DIR, f'gan_{class_name.lower()}')
        ensure_dir(gan_data_dir)
        
        # Copy images
        copied = 0
        for img_id in class_df['image_id'].values:
            src = os.path.join(TRAIN_IMAGES, img_id)
            dst = os.path.join(gan_data_dir, img_id)
            if os.path.exists(src):
                shutil.copy(src, dst)
                copied += 1
        
        print(f"  Copied {copied} images to {gan_data_dir}")
        
        # Save manifest
        class_df.to_csv(os.path.join(gan_data_dir, 'manifest.csv'), index=False)
    
    print("\nGAN data preparation complete!")


def main():
    """Main function"""
    # Prepare classification data
    train_df, val_df, test_df = prepare_classification_data()
    
    # Prepare GAN data
    prepare_gan_data(train_df)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"Classification splits saved to: {PROCESSED_DIR}")
    print(f"GAN training data saved to: {PROCESSED_DIR}/gan_*")
    print("="*60)


if __name__ == "__main__":
    main()
