# Steel Defect Classification with GAN Data Augmentation

A complete implementation of GAN-based data augmentation to improve steel defect classification on the **Severstal Steel Defect Detection** dataset.

## 🎯 Project Overview

This project demonstrates how **DCGAN (Deep Convolutional GAN)** can be used to generate synthetic images of rare defect types, improving classifier performance on imbalanced datasets.

### Key Features
- ✅ DCGAN for generating synthetic Type 3 & Type 4 defects
- ✅ ResNet18-based classifier with baseline and augmented training
- ✅ Comprehensive evaluation metrics (Accuracy, Recall, Precision, F1, Confusion Matrix)
- ✅ Visualization tools for GAN samples and performance comparison
- ✅ Quantitative proof of GAN effectiveness

## 📊 Dataset

**Severstal Steel Defect Detection** (Kaggle)
- 5 classes: Type 1, Type 2, Type 3, Type 4, No Defect
- Highly imbalanced: Type 3 and Type 4 are the rarest
- Image size: 256×1600 (resized for training)

## 🏗️ Architecture

### DCGAN Generator
```
Latent (100) → Linear → 1024×4×4 
→ ConvTranspose (512×8×8) 
→ ConvTranspose (256×16×16) 
→ ConvTranspose (128×32×32) 
→ ConvTranspose (3×64×64) → Tanh
```

### DCGAN Discriminator
```
Image (3×64×64) → Conv (128×32×32) 
→ Conv (256×16×16) 
→ Conv (512×8×8) 
→ Conv (1024×4×4) 
→ Flatten → Linear → Sigmoid
```

### Classifier
- Base: ResNet18 (pretrained on ImageNet)
- Modified FC layer: 512 → 5 classes
- Input: 3×224×224 RGB images

## 🚀 Quick Start

### 1. Installation

```bash
cd steel-defect-gan
pip install -r requirements.txt
```

### 2. Data Exploration

```bash
python data_explorer.py
```
- Analyzes class distribution
- Visualizes sample defects
- Generates plots in `results/plots/`

### 3. Data Preparation

```bash
python prepare_dataset.py
```
- Splits data into train/val/test (80/10/10)
- Organizes data for GAN training
- Creates CSV manifests

### 4. Train GAN Models

```bash
# Train GAN for Type 3 defects
python train_gan.py --defect_type 3 --epochs 200

# Train GAN for Type 4 defects
python train_gan.py --defect_type 4 --epochs 200
```
- Training takes 2-4 hours per GAN (with GPU)
- Checkpoints saved every 50 epochs
- Samples generated every 10 epochs

### 5. Generate Synthetic Data

```bash
# Generate 1000 Type 3 images
python generate_samples.py --defect_type 3 --num_samples 1000

# Generate 1000 Type 4 images
python generate_samples.py --defect_type 4 --num_samples 1000
```

### 6. Train Classifiers

```bash
# Baseline (real data only)
python train_classifier.py --mode baseline

# Augmented (real + GAN data)
python train_classifier.py --mode augmented
```

### 7. Evaluate & Compare

```bash
python evaluate.py
```
- Evaluates both models on test set
- Calculates all metrics
- Generates comparison table

### 8. Visualize Results

```bash
python visualize_results.py
```
- GAN sample grids
- Confusion matrices
- Performance comparison charts
- Per-class recall comparison

## 📁 Project Structure

```
steel-defect-gan/
├── config.py                  # Configuration & hyperparameters
├── requirements.txt           # Dependencies
├── data_explorer.py          # Data analysis
├── prepare_dataset.py        # Data preparation
├── train_gan.py              # GAN training
├── generate_samples.py       # Generate synthetic images
├── train_classifier.py       # Classifier training
├── evaluate.py               # Model evaluation
├── visualize_results.py      # Visualization
├── models/                   
│   ├── generator.py          # DCGAN Generator
│   ├── discriminator.py      # DCGAN Discriminator
│   └── classifier.py         # ResNet18 Classifier
├── utils/
│   └── helpers.py            # Utility functions
├── data/
│   ├── processed/            # Split datasets
│   └── generated/            # GAN-generated images
├── checkpoints/              # Saved models
│   ├── gan_type3/
│   ├── gan_type4/
│   ├── baseline/
│   └── augmented/
└── results/                  # Outputs
    ├── gan_samples/          # GAN samples during training
    ├── confusion_matrices/   # Confusion matrix plots
    ├── plots/                # Performance charts
    └── metrics/              # Evaluation metrics
```

## 📈 Expected Results

### Performance Improvement
- **Baseline**: Lower recall on Type 3/4 due to imbalance
- **Augmented**: +10-20% recall improvement on rare classes
- **Overall Accuracy**: May slightly decrease, but recall significantly improves

### GAN Quality
- Generated images visually resemble real defects
- Diversity in generated samples
- No mode collapse

## 🔧 Hyperparameters

### GAN Training
- Latent dimension: 100
- Batch size: 32
- Epochs: 200
- Learning rate: 0.0002
- Optimizer: Adam (β1=0.5, β2=0.999)

### Classifier Training
- Batch size: 16
- Epochs: 50
- Learning rate: 0.001
- Optimizer: Adam
- Early stopping: patience=10

## 📊 Metrics

- **Overall**: Accuracy, Macro Precision/Recall/F1
- **Per-class**: Precision, Recall, F1-Score for each defect type
- **Confusion Matrix**: Normalized by true labels
- **Special Focus**: Recall on Type 3 and Type 4 (rare classes)

## 🎥 Visualization Outputs

1. **Defect Distribution**: Bar chart showing class imbalance
2. **Sample Defects**: Grid of real defect images with masks
3. **GAN Samples**: 8×8 grids of generated images
4. **Confusion Matrices**: Heatmaps for baseline and augmented
5. **Performance Comparison**: Bar charts comparing metrics
6. **Per-Class Recall**: Highlighting improvement on rare classes

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA/MPS (recommended for training)
- 16GB+ RAM
- 50GB+ disk space (for dataset and models)

## 📝 Notes

- **Training Time**: GAN training ~2-4h per defect type (GPU), Classifier training ~1-2h per mode
- **Image Sizes**: GAN uses 64×64, Classifier uses 224×224
- **Random Seed**: Set to 42 for reproducibility
- **Class Weights**: Used in classifier to handle imbalance

## 🎓 Use Cases

- Steel manufacturing quality control
- Automated defect detection systems
- Research on GAN-based data augmentation
- Deep learning course projects
- Industrial AI applications

## 📚 Citation

Dataset: [Severstal Steel Defect Detection (Kaggle)](https://www.kaggle.com/c/severstal-steel-defect-detection)

## 📧 Support

For questions or issues, please check:
1. Configuration in `config.py`
2. Data preparation steps
3. GPU/CUDA availability
4. Dataset integrity

---

**Built for demonstrating GAN effectiveness in improving classifier performance on imbalanced industrial defect data.**
