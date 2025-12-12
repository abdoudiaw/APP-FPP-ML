"""
2D UNet model for processing SOLPS data from APP-FPP project.
This model is designed to learn 2D spatial patterns in plasma simulation data.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2D -> BatchNorm -> ReLU) x 2
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of intermediate channels
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: ConvTranspose -> Concatenate -> DoubleConv
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        bilinear (bool): Use bilinear upsampling instead of transposed convolution
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D UNet architecture for SOLPS data processing.
    
    This model uses an encoder-decoder architecture with skip connections
    to learn 2D spatial patterns in plasma simulation data.
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output channels
        bilinear (bool): Use bilinear upsampling instead of transposed convolution
        base_channels (int): Base number of channels (will be multiplied at each level)
    """
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_channels=64):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, x):
        """
        Forward pass through the UNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_num_parameters(self):
        """
        Get the total number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = UNet2D(n_channels=1, n_classes=1, base_channels=64)
    print(f"Model architecture:\n{model}")
    print(f"\nTotal trainable parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 1, height, width)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
