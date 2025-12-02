"""
DCGAN Discriminator - Classify real vs fake steel defect images
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator with Convolutional Layers
    Takes 64x64 RGB images and outputs probability [0, 1]
    """
    
    def __init__(self, image_channels=3):
        super(Discriminator, self).__init__()
        
        self.image_channels = image_channels
        
        # Convolutional layers
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 32 x 32
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 16 x 16
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 8 x 8
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 1024 x 4 x 4
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with normal distribution"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img):
        """
        Forward pass
        
        Args:
            img: input images (batch_size, image_channels, 64, 64)
        
        Returns:
            Probability of being real (batch_size, 1)
        """
        features = self.main(img)
        output = self.classifier(features)
        return output


def test_discriminator():
    """Test the discriminator"""
    print("Testing Discriminator...")
    
    batch_size = 8
    
    # Create discriminator
    disc = Discriminator(image_channels=3)
    print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # Create random images
    fake_images = torch.randn(batch_size, 3, 64, 64)
    
    # Classify images
    with torch.no_grad():
        predictions = disc(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print("Discriminator test passed! ✓")


if __name__ == "__main__":
    test_discriminator()
