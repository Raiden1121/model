"""
DCGAN Generator - Generate synthetic steel defect images
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator with Transpose Convolutions
    Generates 64x64 RGB images from random noise
    """
    
    def __init__(self, latent_dim=100, image_channels=3):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        # Initial projection layer
        # Input: latent_dim (100) -> Output: 1024 * 4 * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transpose convolutional layers
        self.main = nn.Sequential(
            # State: 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 8 x 8
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 16 x 16
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 32 x 32
            
            nn.ConvTranspose2d(128, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: image_channels x 64 x 64
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
    
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z: latent noise vector (batch_size, latent_dim)
        
        Returns:
            Generated images (batch_size, image_channels, 64, 64)
        """
        # Project and reshape
        x = self.project(z)
        x = x.view(-1, 1024, 4, 4)
        
        # Generate image
        x = self.main(x)
        
        return x


def test_generator():
    """Test the generator"""
    print("Testing Generator...")
    
    latent_dim = 100
    batch_size = 8
    
    # Create generator
    gen = Generator(latent_dim=latent_dim, image_channels=3)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    # Generate random noise
    z = torch.randn(batch_size, latent_dim)
    
    # Generate images
    with torch.no_grad():
        fake_images = gen(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {fake_images.shape}")
    print(f"Output range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    print("Generator test passed! ✓")


if __name__ == "__main__":
    test_generator()
