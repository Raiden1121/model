"""
Evaluate and compare baseline vs augmented classifiers
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from models.classifier import SteelDefectClassifier
from train_classifier import SteelDefectDataset
from utils.helpers import ensure_dir


def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(preds, labels):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_per_class': precision_score(labels, preds, average=None, zero_division=0),
        'recall_per_class': recall_score(labels, preds, average=None, zero_division=0),
        'f1_per_class': f1_score(labels, preds, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds)
    }
    
    return metrics


def evaluate():
    """Evaluate both baseline and augmented models"""
    print(\"=\"*70)
    print(\"EVALUATING CLASSIFIERS\")
    print(\"=\"*70)
    
    # Test data
    test_transform = transforms.Compose([
        transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    test_dataset = SteelDefectDataset(
        os.path.join(PROCESSED_DIR, 'test.csv'),
        TRAIN_IMAGES,
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=CLASSIFIER_BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f\"\\nTest dataset size: {len(test_dataset)}\")
    
    # Results directory
    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    ensure_dir(metrics_dir)
    
    results = {}
    
    for mode in ['baseline', 'augmented']:
        print(f\"\\n{'='*70}\")
        print(f\"EVALUATING {mode.upper()} MODEL\")
        print(f\"{'='*70}\")
        
        # Load model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, mode, 'best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f\"Warning: Checkpoint not found: {checkpoint_path}\")
            print(f\"Skipping {mode} evaluation\")
            continue
        
        model = SteelDefectClassifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f\"Loaded model from: {checkpoint_path}\")
        
        # Evaluate
        preds, labels = evaluate_model(model, test_loader, DEVICE)
        metrics = calculate_metrics(preds, labels)
        
        # Store results
        results[mode] = {
            'predictions': preds,
            'labels': labels,
            'metrics': metrics
        }
        
        # Print results
        print(f\"\\n{mode.upper()} Results:\")
        print(\"-\" * 50)
        print(f\"Overall Accuracy: {metrics['accuracy']:.4f}\")
        print(f\"Macro Precision: {metrics['precision_macro']:.4f}\")
        print(f\"Macro Recall: {metrics['recall_macro']:.4f}\")
        print(f\"Macro F1-Score: {metrics['f1_macro']:.4f}\")
        
        print(f\"\\nPer-Class Metrics:\")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f\"  {class_name}:\")
            print(f\"    Precision: {metrics['precision_per_class'][i]:.4f}\")
            print(f\"    Recall: {metrics['recall_per_class'][i]:.4f}\")
            print(f\"    F1-Score: {metrics['f1_per_class'][i]:.4f}\")
        
        # Save detailed results
        with open(os.path.join(metrics_dir, f'{mode}_metrics.txt'), 'w') as f:
            f.write(f\"{mode.upper()} MODEL EVALUATION\\n\")
            f.write(\"=\" * 50 + \"\\n\\n\")
            f.write(f\"Overall Accuracy: {metrics['accuracy']:.4f}\\n\")
            f.write(f\"Macro Precision: {metrics['precision_macro']:.4f}\\n\")
            f.write(f\"Macro Recall: {metrics['recall_macro']:.4f}\\n\")
            f.write(f\"Macro F1-Score: {metrics['f1_macro']:.4f}\\n\\n\")
            f.write(\"Per-Class Metrics:\\n\")
            for i, class_name in enumerate(CLASS_NAMES):
                f.write(f\"\\n{class_name}:\\n\")
                f.write(f\"  Precision: {metrics['precision_per_class'][i]:.4f}\\n\")
                f.write(f\"  Recall: {metrics['recall_per_class'][i]:.4f}\\n\")
                f.write(f\"  F1-Score: {metrics['f1_per_class'][i]:.4f}\\n\")
    
    # Compare results
    if 'baseline' in results and 'augmented' in results:
        print(f\"\\n{'='*70}\")
        print(\"COMPARISON: BASELINE vs AUGMENTED\")
        print(f\"{'='*70}\")
        
        comparison_data = []
        
        for metric_name in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            baseline_val = results['baseline']['metrics'][metric_name]
            augmented_val = results['augmented']['metrics'][metric_name]
            improvement = augmented_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
            
            comparison_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Baseline': f\"{baseline_val:.4f}\",
                'Augmented': f\"{augmented_val:.4f}\",
                'Improvement': f\"{improvement:+.4f} ({improvement_pct:+.1f}%)\"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(\"\\n\" + comparison_df.to_string(index=False))
        
        # Per-class comparison
        print(f\"\\nPer-Class Recall Comparison:\")
        print(\"-\" * 70)
        
        for i, class_name in enumerate(CLASS_NAMES):
            baseline_recall = results['baseline']['metrics']['recall_per_class'][i]
            augmented_recall = results['augmented']['metrics']['recall_per_class'][i]
            improvement = augmented_recall - baseline_recall
            improvement_pct = (improvement / baseline_recall) * 100 if baseline_recall > 0 else 0
            
            print(f\"{class_name:15s} | Baseline: {baseline_recall:.4f} | \"
                  f\"Augmented: {augmented_recall:.4f} | \"
                  f\"Δ: {improvement:+.4f} ({improvement_pct:+.1f}%)\")
        
        # Save comparison
        comparison_df.to_csv(os.path.join(metrics_dir, 'comparison.csv'), index=False)
    
    # Save all results
    torch.save(results, os.path.join(metrics_dir, 'evaluation_results.pth'))
    
    print(f\"\\n{'='*70}\")
    print(\"EVALUATION COMPLETE!\")
    print(f\"{'='*70}\")
    print(f\"Results saved to: {metrics_dir}\")
    print(f\"{'='*70}\\n\")


def main():
    evaluate()


if __name__ == \"__main__\":
    main()
