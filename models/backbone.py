"""
Backbone Networks for PMP-DACIS
ResNet-18 and MobileNetV2 implementations with pruning support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Optional, Tuple


class PrunableConv2d(nn.Conv2d):
    """Conv2d layer with channel pruning support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_mask = None
        
    def set_mask(self, mask: torch.Tensor):
        """Set pruning mask for output channels"""
        self.channel_mask = mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.channel_mask is not None:
            out = out * self.channel_mask.view(1, -1, 1, 1)
        return out


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone with channel pruning support
    
    Architecture:
        conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool
        
    Output: 512-dim feature vector
    """
    
    def __init__(self, pretrained: bool = True, num_classes: Optional[int] = None):
        super().__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Feature extraction layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        self.avgpool = resnet.avgpool
        
        # Feature dimension
        self.feature_dim = 512
        
        # Optional classifier
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(512, num_classes)
            
        # Track prunable layers
        self.prunable_layers = self._get_prunable_layers()
        
        # Channel masks for pruning
        self.channel_masks: Dict[str, torch.Tensor] = {}
        
    def _get_prunable_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """Get all conv layers that can be pruned"""
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                layers.append((name, module))
        return layers
    
    def set_channel_mask(self, layer_name: str, mask: torch.Tensor):
        """Set pruning mask for a specific layer"""
        self.channel_masks[layer_name] = mask
        
    def get_channel_counts(self) -> Dict[str, int]:
        """Get number of channels per layer"""
        counts = {}
        for name, module in self.prunable_layers:
            counts[name] = module.out_channels
        return counts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            features: Feature tensor [B, 512]
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        if self.classifier is not None:
            return self.classifier(features)
        
        return features
    
    def extract_features(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        Extract features with optional intermediate outputs
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_intermediate: Whether to return features from each layer
            
        Returns:
            features: Final features [B, 512]
            intermediate: Dict of intermediate features (if return_intermediate=True)
        """
        intermediate = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        intermediate['stem'] = x
        
        x = self.layer1(x)
        intermediate['layer1'] = x
        
        x = self.layer2(x)
        intermediate['layer2'] = x
        
        x = self.layer3(x)
        intermediate['layer3'] = x
        
        x = self.layer4(x)
        intermediate['layer4'] = x
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_intermediate:
            return features, intermediate
        return features


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone with channel pruning support
    
    Optimized for edge deployment with inverted residuals
    Output: 1280-dim feature vector (or 512 after projection)
    """
    
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Feature extraction layers
        self.features = mobilenet.features
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.feature_dim = feature_dim
        
        # Track prunable layers
        self.prunable_layers = self._get_prunable_layers()
        
    def _get_prunable_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """Get all conv layers that can be pruned"""
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append((name, module))
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            features: Feature tensor [B, feature_dim]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return x
    
    def extract_features(self, x: torch.Tensor, return_intermediate: bool = False):
        """Extract features with optional intermediate outputs"""
        intermediate = {}
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [3, 6, 13, 17]:  # Key bottleneck outputs
                intermediate[f'block_{idx}'] = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.projection(x)
        
        if return_intermediate:
            return features, intermediate
        return features


def create_backbone(name: str = "resnet18", pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create backbone networks
    
    Args:
        name: Backbone name ("resnet18" or "mobilenetv2")
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for backbone
        
    Returns:
        backbone: Backbone network
    """
    backbones = {
        "resnet18": ResNet18Backbone,
        "mobilenetv2": MobileNetV2Backbone,
    }
    
    if name.lower() not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(backbones.keys())}")
    
    return backbones[name.lower()](pretrained=pretrained, **kwargs)


class FeatureExtractor(nn.Module):
    """
    Wrapper for feature extraction with support for different backbones
    Provides unified interface for PMP-DACIS framework
    """
    
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained)
        self.feature_dim = self.backbone.feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get activations from all layers for DACIS scoring"""
        _, intermediate = self.backbone.extract_features(x, return_intermediate=True)
        return intermediate
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_size: Tuple[int, int] = (224, 224)) -> int:
        """Estimate FLOPs for given input size"""
        # Simplified FLOPs estimation
        total_flops = 0
        x = torch.randn(1, 3, *input_size)
        
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 * K^2 * C_in * C_out * H_out * W_out
                h_out = input_size[0] // (2 ** (int(name.split('.')[0][-1]) if name[0] == 'l' else 1))
                w_out = h_out
                flops = 2 * module.kernel_size[0] * module.kernel_size[1] * \
                        module.in_channels * module.out_channels * h_out * w_out
                total_flops += flops
                
        return total_flops


if __name__ == "__main__":
    # Test backbones
    print("Testing ResNet-18 backbone...")
    resnet = create_backbone("resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = resnet(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Feature dim: {resnet.feature_dim}")
    
    print("\nTesting MobileNetV2 backbone...")
    mobilenet = create_backbone("mobilenetv2", pretrained=False)
    out = mobilenet(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Feature dim: {mobilenet.feature_dim}")
    
    print("\nTesting FeatureExtractor wrapper...")
    extractor = FeatureExtractor("resnet18", pretrained=False)
    print(f"  Total parameters: {extractor.count_parameters():,}")
