"""
Configuration file for Steel Defect GAN project
"""
import torch
import os

# ============= Paths =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'severstal-steel-defect-detection')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
GENERATED_DIR = os.path.join(BASE_DIR, 'data', 'generated')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ============= Data Split =============
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ============= Image Settings =============
GAN_IMAGE_SIZE = 64  # For GAN training (faster)
CLASSIFIER_IMAGE_SIZE = 224  # For ResNet18
IMAGE_CHANNELS = 3

# ============= Class Names =============
CLASS_NAMES = ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'No_Defect']
NUM_CLASSES = 5
RARE_CLASSES = [2, 3]  # Type_3 and Type_4 (0-indexed)

# ============= GAN Hyperparameters =============
GAN_LATENT_DIM = 100
GAN_BATCH_SIZE = 32
GAN_EPOCHS = 200
GAN_LR = 0.0002
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999

# ============= Classifier Hyperparameters =============
CLASSIFIER_BATCH_SIZE = 16
CLASSIFIER_EPOCHS = 50
CLASSIFIER_LR = 0.001
EARLY_STOPPING_PATIENCE = 10

# ============= Augmentation Settings =============
# Number of synthetic samples to generate per rare class
NUM_SYNTHETIC_SAMPLES = {
    'Type_3': 1000,
    'Type_4': 1000
}

# ============= Device Configuration =============
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")

# ============= ImageNet Normalization =============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
