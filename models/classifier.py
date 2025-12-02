"""
ResNet18-based classifier for steel defect classification
"""
import torch
import torch.nn as nn
from torchvision import models


class SteelDefectClassifier(nn.Module):
    """
    Modified ResNet18 for 5-class steel defect classification
    Classes: Type_1, Type_2, Type_3, Type_4, No_Defect
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(SteelDefectClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input images (batch_size, 3, 224, 224)
        
        Returns:
            logits (batch_size, num_classes)
        """
        return self.resnet(x)


def test_classifier():
    """Test the classifier"""
    print("Testing Steel Defect Classifier...")
    
    batch_size = 4
    num_classes = 5
    
    # Create classifier
    model = SteelDefectClassifier(num_classes=num_classes, pretrained=False)
    print(f"Classifier parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create random images (224x224 for ResNet)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum (should be ~1.0): {probabilities[0].sum():.4f}")
    print("Classifier test passed! ✓")


if __name__ == "__main__":
    test_classifier()
