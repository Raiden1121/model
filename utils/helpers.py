"""
Utility functions for Steel Defect GAN project
"""
import numpy as np
import torch
import os
from PIL import Image


def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decode Run-Length Encoded mask
    
    Args:
        mask_rle: string of RLE encoded mask
        shape: (height, width) of the mask
    
    Returns:
        numpy array of decoded mask
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def rle_encode(mask):
    """
    Encode mask to Run-Length Encoding
    
    Args:
        mask: numpy array of binary mask
    
    Returns:
        string of RLE encoded mask
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (Epoch {epoch}, Loss: {loss:.4f})")
    return epoch, loss


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize_image(tensor, mean, std):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: normalized image tensor
        mean: mean used for normalization
        std: std used for normalization
    
    Returns:
        denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


import pandas as pd

def get_class_weights(train_df):
    """
    Calculate class weights for handling imbalanced data
    
    Args:
        train_df: training dataframe with class labels
    
    Returns:
        tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['class'].values),
        y=train_df['class'].values
    )
    return torch.FloatTensor(class_weights)
