"""
Visualize results: GAN samples, confusion matrices, performance comparisons
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.helpers import ensure_dir


def visualize_gan_samples():
    """Visualize GAN-generated samples"""
    print("Visualizing GAN-generated samples...")
    
    output_dir = os.path.join(RESULTS_DIR, 'gan_samples')
    
    for defect_type in [3, 4]:
        class_name = CLASS_NAMES[defect_type - 1]
        sample_dir = os.path.join(output_dir, f'type{defect_type}')
        
        # Find the latest epoch sample
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.startswith('epoch_') and f.endswith('.png')]
            if sample_files:
                latest_sample = sorted(sample_files)[-1]
                sample_path = os.path.join(sample_dir, latest_sample)
                
                # Display
                fig, ax = plt.subplots(figsize=(12, 12))
                img = Image.open(sample_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'GAN-Generated {class_name} Samples', 
                           fontsize=16, fontweight='bold', pad=20)
                
                save_path = os.path.join(RESULTS_DIR, 'plots', f'gan_samples_type{defect_type}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved Type {defect_type} samples: {save_path}")


def plot_confusion_matrix(cm, title, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Count'},
                linewidths=1, linecolor='gray')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_confusion_matrices():
    """Visualize confusion matrices for baseline and augmented models"""
    print("\\nVisualizing confusion matrices...")
    
    metrics_path = os.path.join(RESULTS_DIR, 'metrics', 'evaluation_results.pth')
    
    if not os.path.exists(metrics_path):
        print("  Warning: Evaluation results not found. Run evaluate.py first.")
        return
    
    results = torch.load(metrics_path, weights_only=False)
    cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices')
    ensure_dir(cm_dir)
    
    for mode in ['baseline', 'augmented']:
        if mode in results:
            cm = results[mode]['metrics']['confusion_matrix']
            title = f'Confusion Matrix - {mode.upper()} Model'
            save_path = os.path.join(cm_dir, f'confusion_matrix_{mode}.png')
            plot_confusion_matrix(cm, title, save_path)
            print(f"  Saved {mode} confusion matrix: {save_path}")


def plot_performance_comparison():
    """Plot performance comparison between baseline and augmented"""
    print("\\nPlotting performance comparison...")
    
    comparison_path = os.path.join(RESULTS_DIR, 'metrics', 'comparison.csv')
    
    if not os.path.exists(comparison_path):
        print("  Warning: Comparison results not found. Run evaluate.py first.")
        return
    
    df = pd.read_csv(comparison_path)
    
    # Extract values
    metrics = df['Metric'].values
    baseline = df['Baseline'].astype(float).values
    augmented = df['Augmented'].astype(float).values
    
    # Plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', 
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, augmented, width, label='Augmented (with GAN)', 
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs GAN-Augmented', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved performance comparison: {save_path}")


def plot_per_class_recall():
    """Plot per-class recall comparison"""
    print("\\nPlotting per-class recall comparison...")
    
    metrics_path = os.path.join(RESULTS_DIR, 'metrics', 'evaluation_results.pth')
    
    if not os.path.exists(metrics_path):
        print("  Warning: Evaluation results not found. Run evaluate.py first.")
        return
    
    results = torch.load(metrics_path, weights_only=False)
    
    if 'baseline' not in results or 'augmented' not in results:
        print("  Warning: Need both baseline and augmented results.")
        return
    
    baseline_recall = results['baseline']['metrics']['recall_per_class']
    augmented_recall = results['augmented']['metrics']['recall_per_class']
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_recall, width, label='Baseline', 
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, augmented_recall, width, label='Augmented (with GAN)', 
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight rare classes
    for idx in RARE_CLASSES:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color='gold', zorder=0)
    
    ax.set_xlabel('Defect Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall Score', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Recall: Baseline vs GAN-Augmented\\n(Highlighted: Rare Classes)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'plots', 'per_class_recall.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved per-class recall: {save_path}")


def main():
    """Generate all visualizations"""
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    ensure_dir(os.path.join(RESULTS_DIR, 'plots'))
    
    # GAN samples
    visualize_gan_samples()
    
    # Confusion matrices
    visualize_confusion_matrices()
    
    # Performance comparisons
    plot_performance_comparison()
    plot_per_class_recall()
    
    print(f"\\n{'='*70}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Visualizations saved to: {RESULTS_DIR}")
    print(f"{'='*70}\\n")


if __name__ == "__main__":
    main()
