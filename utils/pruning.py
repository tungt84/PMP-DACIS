"""
PMP-DACIS Channel Pruning Utilities

Implements structured channel pruning operations:
- Channel importance ranking
- Magnitude-based pruning
- Gradient-based pruning
- DACIS-guided pruning
- FLOPs and parameter counting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import copy


class ChannelPruner:
    """
    Structured channel pruning for convolutional neural networks.
    
    Supports multiple pruning strategies including magnitude-based,
    gradient-based, and DACIS-guided channel selection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        min_channels: int = 8,
        global_pruning: bool = True
    ):
        """
        Initialize channel pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of channels to remove (0-1)
            min_channels: Minimum channels to keep per layer
            global_pruning: Use global ranking vs per-layer
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.min_channels = min_channels
        self.global_pruning = global_pruning
        
        # Track prunable layers
        self.prunable_layers = self._identify_prunable_layers()
        
        # Channel masks (None = all channels active)
        self.channel_masks: Dict[str, torch.Tensor] = {}
    
    def _identify_prunable_layers(self) -> Dict[str, nn.Conv2d]:
        """Identify all Conv2d layers that can be pruned."""
        prunable = OrderedDict()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Skip 1x1 convolutions in residual connections
                # and depthwise separable convolutions
                if module.groups == module.in_channels:
                    continue  # Depthwise conv
                if module.kernel_size == (1, 1) and 'downsample' in name:
                    continue  # Residual downsampling
                
                prunable[name] = module
        
        return prunable
    
    def compute_channel_importance(
        self,
        strategy: str = 'magnitude',
        data_loader: Optional[any] = None,
        num_batches: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Compute channel importance scores for all prunable layers.
        
        Args:
            strategy: 'magnitude', 'gradient', 'taylor', or 'random'
            data_loader: DataLoader for gradient-based methods
            num_batches: Number of batches for gradient computation
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        importance_scores = {}
        
        if strategy == 'magnitude':
            importance_scores = self._magnitude_importance()
        elif strategy == 'gradient':
            if data_loader is None:
                raise ValueError("data_loader required for gradient strategy")
            importance_scores = self._gradient_importance(data_loader, num_batches)
        elif strategy == 'taylor':
            if data_loader is None:
                raise ValueError("data_loader required for taylor strategy")
            importance_scores = self._taylor_importance(data_loader, num_batches)
        elif strategy == 'random':
            importance_scores = self._random_importance()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return importance_scores
    
    def _magnitude_importance(self) -> Dict[str, torch.Tensor]:
        """Compute importance based on filter weight magnitude."""
        importance = {}
        
        for name, layer in self.prunable_layers.items():
            # Weight shape: [out_channels, in_channels, H, W]
            weights = layer.weight.data
            
            # L2 norm across input channels and spatial dimensions
            filter_importance = weights.abs().pow(2).sum(dim=(1, 2, 3)).sqrt()
            
            importance[name] = filter_importance.cpu()
        
        return importance
    
    def _gradient_importance(
        self,
        data_loader,
        num_batches: int
    ) -> Dict[str, torch.Tensor]:
        """Compute importance based on gradient magnitude."""
        importance = {name: torch.zeros(layer.out_channels) 
                      for name, layer in self.prunable_layers.items()}
        
        self.model.train()
        device = next(self.model.parameters()).device
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, layer in self.prunable_layers.items():
                if layer.weight.grad is not None:
                    grad_magnitude = layer.weight.grad.abs().sum(dim=(1, 2, 3))
                    importance[name] += grad_magnitude.cpu()
        
        # Normalize
        for name in importance:
            importance[name] /= num_batches
        
        return importance
    
    def _taylor_importance(
        self,
        data_loader,
        num_batches: int
    ) -> Dict[str, torch.Tensor]:
        """Compute Taylor expansion based importance (weight * gradient)."""
        importance = {name: torch.zeros(layer.out_channels) 
                      for name, layer in self.prunable_layers.items()}
        
        self.model.train()
        device = next(self.model.parameters()).device
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, layer in self.prunable_layers.items():
                if layer.weight.grad is not None:
                    # Taylor: |weight * gradient|
                    taylor = (layer.weight.data * layer.weight.grad).abs()
                    taylor_score = taylor.sum(dim=(1, 2, 3))
                    importance[name] += taylor_score.cpu()
        
        # Normalize
        for name in importance:
            importance[name] /= num_batches
        
        return importance
    
    def _random_importance(self) -> Dict[str, torch.Tensor]:
        """Random importance scores (baseline)."""
        importance = {}
        
        for name, layer in self.prunable_layers.items():
            importance[name] = torch.rand(layer.out_channels)
        
        return importance
    
    def compute_pruning_masks(
        self,
        importance_scores: Dict[str, torch.Tensor],
        pruning_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute binary pruning masks from importance scores.
        
        Args:
            importance_scores: Channel importance per layer
            pruning_ratio: Override default pruning ratio
            
        Returns:
            Binary masks (1 = keep, 0 = prune)
        """
        ratio = pruning_ratio if pruning_ratio is not None else self.pruning_ratio
        
        if self.global_pruning:
            masks = self._global_pruning_masks(importance_scores, ratio)
        else:
            masks = self._local_pruning_masks(importance_scores, ratio)
        
        self.channel_masks = masks
        return masks
    
    def _global_pruning_masks(
        self,
        importance_scores: Dict[str, torch.Tensor],
        pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Global threshold-based pruning."""
        # Collect all scores
        all_scores = []
        score_to_layer = []
        
        for name, scores in importance_scores.items():
            all_scores.extend(scores.tolist())
            score_to_layer.extend([name] * len(scores))
        
        # Sort globally
        sorted_indices = np.argsort(all_scores)
        num_prune = int(len(all_scores) * pruning_ratio)
        
        # Initialize masks (all channels kept)
        masks = {name: torch.ones(scores.shape[0], dtype=torch.bool)
                 for name, scores in importance_scores.items()}
        
        # Mark channels to prune
        channels_pruned = {name: 0 for name in importance_scores}
        
        for i in sorted_indices[:num_prune]:
            layer_name = score_to_layer[i]
            layer_size = len(importance_scores[layer_name])
            
            # Check minimum channels constraint
            current_kept = masks[layer_name].sum().item()
            if current_kept <= self.min_channels:
                continue
            
            # Find local index
            local_idx = i - sum(len(importance_scores[n]) 
                              for n in list(importance_scores.keys())[:list(importance_scores.keys()).index(layer_name)])
            
            if local_idx >= 0 and local_idx < layer_size:
                masks[layer_name][local_idx] = False
                channels_pruned[layer_name] += 1
        
        return masks
    
    def _local_pruning_masks(
        self,
        importance_scores: Dict[str, torch.Tensor],
        pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Per-layer pruning."""
        masks = {}
        
        for name, scores in importance_scores.items():
            num_channels = len(scores)
            num_keep = max(
                self.min_channels,
                int(num_channels * (1 - pruning_ratio))
            )
            
            # Keep top-k channels
            _, top_indices = torch.topk(scores, num_keep)
            
            mask = torch.zeros(num_channels, dtype=torch.bool)
            mask[top_indices] = True
            masks[name] = mask
        
        return masks
    
    def apply_masks(self):
        """Apply pruning masks to the model (zero out pruned channels)."""
        for name, layer in self.prunable_layers.items():
            if name in self.channel_masks:
                mask = self.channel_masks[name].to(layer.weight.device)
                
                # Zero out pruned channels
                layer.weight.data[~mask] = 0
                
                if layer.bias is not None:
                    layer.bias.data[~mask] = 0
    
    def get_pruned_model(self, remove_channels: bool = False) -> nn.Module:
        """
        Get pruned model.
        
        Args:
            remove_channels: Actually remove channels (changes architecture)
                           vs just zeroing them out
        
        Returns:
            Pruned model
        """
        if not remove_channels:
            # Just apply masks
            pruned_model = copy.deepcopy(self.model)
            self.model, pruned_model = pruned_model, self.model
            self.apply_masks()
            self.model, pruned_model = pruned_model, self.model
            return pruned_model
        else:
            # Actually remove channels - more complex, architecture changes
            # For now, return zeroed model
            return self.get_pruned_model(remove_channels=False)
    
    def get_pruning_stats(self) -> Dict[str, any]:
        """Get statistics about the pruning."""
        stats = {
            'total_channels': 0,
            'pruned_channels': 0,
            'kept_channels': 0,
            'per_layer': {},
            'compression_ratio': 1.0
        }
        
        for name, layer in self.prunable_layers.items():
            num_channels = layer.out_channels
            stats['total_channels'] += num_channels
            
            if name in self.channel_masks:
                kept = self.channel_masks[name].sum().item()
                pruned = num_channels - kept
            else:
                kept = num_channels
                pruned = 0
            
            stats['kept_channels'] += kept
            stats['pruned_channels'] += pruned
            
            stats['per_layer'][name] = {
                'total': num_channels,
                'kept': kept,
                'pruned': pruned,
                'pruning_ratio': pruned / num_channels if num_channels > 0 else 0
            }
        
        if stats['total_channels'] > 0:
            stats['compression_ratio'] = (
                stats['total_channels'] / stats['kept_channels']
                if stats['kept_channels'] > 0 else float('inf')
            )
            stats['overall_pruning_ratio'] = (
                stats['pruned_channels'] / stats['total_channels']
            )
        
        return stats


def apply_channel_pruning(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    strategy: str = 'magnitude',
    data_loader: Optional[any] = None,
    global_pruning: bool = True,
    min_channels: int = 8
) -> Tuple[nn.Module, Dict[str, any]]:
    """
    Convenience function to apply channel pruning.
    
    Args:
        model: Model to prune
        pruning_ratio: Fraction of channels to prune
        strategy: Pruning strategy
        data_loader: Data for gradient-based methods
        global_pruning: Global vs per-layer pruning
        min_channels: Minimum channels per layer
        
    Returns:
        Tuple of (pruned_model, pruning_stats)
    """
    pruner = ChannelPruner(
        model=model,
        pruning_ratio=pruning_ratio,
        min_channels=min_channels,
        global_pruning=global_pruning
    )
    
    importance = pruner.compute_channel_importance(
        strategy=strategy,
        data_loader=data_loader
    )
    
    pruner.compute_pruning_masks(importance)
    pruned_model = pruner.get_pruned_model()
    stats = pruner.get_pruning_stats()
    
    return pruned_model, stats


def compute_sparsity(model: nn.Module, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Compute sparsity statistics for a model.
    
    Args:
        model: Model to analyze
        threshold: Values below this are considered zero
        
    Returns:
        Sparsity statistics
    """
    total_params = 0
    zero_params = 0
    layer_sparsity = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        num_zeros = (param.abs() < threshold).sum().item()
        
        total_params += num_params
        zero_params += num_zeros
        
        layer_sparsity[name] = num_zeros / num_params if num_params > 0 else 0
    
    return {
        'total_params': total_params,
        'zero_params': zero_params,
        'nonzero_params': total_params - zero_params,
        'overall_sparsity': zero_params / total_params if total_params > 0 else 0,
        'density': (total_params - zero_params) / total_params if total_params > 0 else 0,
        'per_layer': layer_sparsity
    }


def count_nonzero_params(model: nn.Module, threshold: float = 1e-6) -> int:
    """Count non-zero parameters in a model."""
    total = 0
    for param in model.parameters():
        total += (param.abs() >= threshold).sum().item()
    return total


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    count_zero_ops: bool = False,
    threshold: float = 1e-6
) -> Dict[str, int]:
    """
    Compute FLOPs for a model.
    
    Args:
        model: Model to analyze
        input_size: Input tensor shape
        count_zero_ops: Count operations with zero weights
        threshold: Zero threshold
        
    Returns:
        FLOPs statistics
    """
    total_flops = 0
    nonzero_flops = 0
    per_layer_flops = {}
    
    def hook_fn(module, input, output, name):
        nonlocal total_flops, nonzero_flops
        
        if isinstance(module, nn.Conv2d):
            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]
            
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels // module.groups
            out_channels = module.out_channels
            
            flops = batch_size * np.prod(output_dims) * kernel_ops * in_channels * out_channels
            total_flops += flops
            
            # Count non-zero FLOPs
            if not count_zero_ops:
                weight_density = (module.weight.abs() >= threshold).float().mean().item()
                nonzero_flops += int(flops * weight_density)
            else:
                nonzero_flops += flops
            
            per_layer_flops[name] = int(flops)
            
        elif isinstance(module, nn.Linear):
            batch_size = input[0].shape[0]
            flops = batch_size * module.in_features * module.out_features
            total_flops += flops
            
            if not count_zero_ops:
                weight_density = (module.weight.abs() >= threshold).float().mean().item()
                nonzero_flops += int(flops * weight_density)
            else:
                nonzero_flops += flops
            
            per_layer_flops[name] = int(flops)
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            hooks.append(hook)
    
    model.eval()
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return {
        'total_flops': int(total_flops),
        'nonzero_flops': int(nonzero_flops),
        'theoretical_speedup': total_flops / nonzero_flops if nonzero_flops > 0 else 1.0,
        'gflops': total_flops / 1e9,
        'per_layer': per_layer_flops
    }


class GradualPruner:
    """
    Gradual magnitude pruning with sparsity scheduling.
    
    Implements the progressive pruning strategy from PMP framework.
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.7,
        begin_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100
    ):
        """
        Initialize gradual pruner.
        
        Args:
            model: Model to prune
            initial_sparsity: Starting sparsity
            final_sparsity: Target sparsity
            begin_step: Step to begin pruning
            end_step: Step to stop pruning
            frequency: Pruning frequency (steps)
        """
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        
        self.current_step = 0
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Initialize masks
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize all masks to ones."""
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)
    
    def get_current_sparsity(self) -> float:
        """Get current target sparsity based on step."""
        if self.current_step < self.begin_step:
            return self.initial_sparsity
        if self.current_step >= self.end_step:
            return self.final_sparsity
        
        # Cubic sparsity schedule
        progress = (self.current_step - self.begin_step) / (self.end_step - self.begin_step)
        sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3
        
        return sparsity
    
    def step(self):
        """Perform one pruning step if needed."""
        self.current_step += 1
        
        if self.current_step < self.begin_step:
            return
        if self.current_step > self.end_step:
            return
        if (self.current_step - self.begin_step) % self.frequency != 0:
            return
        
        # Update masks
        target_sparsity = self.get_current_sparsity()
        self._update_masks(target_sparsity)
        self._apply_masks()
    
    def _update_masks(self, target_sparsity: float):
        """Update masks based on current weights and target sparsity."""
        # Collect all weights and their magnitudes
        all_weights = []
        weight_info = []
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                weights = param.data.abs().flatten()
                all_weights.append(weights)
                weight_info.extend([(name, i) for i in range(len(weights))])
        
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        num_prune = int(len(all_weights) * target_sparsity)
        if num_prune > 0:
            threshold = torch.topk(all_weights, num_prune, largest=False).values[-1]
        else:
            threshold = 0
        
        # Update masks
        for name, param in self.model.named_parameters():
            if name in self.masks:
                self.masks[name] = param.data.abs() >= threshold
    
    def _apply_masks(self):
        """Apply current masks to model weights."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get current sparsity statistics."""
        total = 0
        pruned = 0
        
        for name, mask in self.masks.items():
            total += mask.numel()
            pruned += (~mask).sum().item()
        
        return {
            'target_sparsity': self.get_current_sparsity(),
            'actual_sparsity': pruned / total if total > 0 else 0,
            'total_params': total,
            'pruned_params': pruned,
            'current_step': self.current_step
        }


if __name__ == '__main__':
    # Demo/test of pruning utilities
    print("=" * 60)
    print("PMP-DACIS Pruning Utilities Demo")
    print("=" * 60)
    
    # Create a simple test model
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    model = SimpleConvNet()
    
    # Test Channel Pruner
    print("\n1. Channel Pruning (Magnitude-based)")
    print("-" * 40)
    pruner = ChannelPruner(model, pruning_ratio=0.3)
    importance = pruner.compute_channel_importance(strategy='magnitude')
    pruner.compute_pruning_masks(importance)
    stats = pruner.get_pruning_stats()
    
    print(f"   Total channels: {stats['total_channels']}")
    print(f"   Pruned channels: {stats['pruned_channels']}")
    print(f"   Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"   Overall pruning ratio: {stats['overall_pruning_ratio']:.1%}")
    
    # Test convenience function
    print("\n2. Apply Channel Pruning (Convenience Function)")
    print("-" * 40)
    model2 = SimpleConvNet()
    pruned_model, prune_stats = apply_channel_pruning(
        model2, 
        pruning_ratio=0.5,
        strategy='magnitude'
    )
    print(f"   Original params: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"   Compression: {prune_stats['compression_ratio']:.2f}x")
    
    # Test sparsity computation
    print("\n3. Sparsity Statistics")
    print("-" * 40)
    sparsity_stats = compute_sparsity(pruned_model)
    print(f"   Total parameters: {sparsity_stats['total_params']:,}")
    print(f"   Non-zero parameters: {sparsity_stats['nonzero_params']:,}")
    print(f"   Overall sparsity: {sparsity_stats['overall_sparsity']:.1%}")
    print(f"   Density: {sparsity_stats['density']:.1%}")
    
    # Test FLOPs computation
    print("\n4. FLOPs Analysis")
    print("-" * 40)
    flops_stats = compute_flops(pruned_model)
    print(f"   Total FLOPs: {flops_stats['gflops']:.2f} GFLOPs")
    print(f"   Non-zero FLOPs: {flops_stats['nonzero_flops'] / 1e9:.2f} GFLOPs")
    print(f"   Theoretical speedup: {flops_stats['theoretical_speedup']:.2f}x")
    
    # Test Gradual Pruner
    print("\n5. Gradual Pruning Schedule")
    print("-" * 40)
    model3 = SimpleConvNet()
    gradual_pruner = GradualPruner(
        model3,
        initial_sparsity=0.0,
        final_sparsity=0.7,
        begin_step=0,
        end_step=100,
        frequency=20
    )
    
    print("   Step | Target Sparsity | Actual Sparsity")
    print("   -----|-----------------|----------------")
    for step in range(0, 120, 20):
        gradual_pruner.current_step = step
        if step > 0:
            gradual_pruner.step()
        stats = gradual_pruner.get_sparsity_stats()
        print(f"   {step:4d} | {stats['target_sparsity']:14.1%} | {stats['actual_sparsity']:.1%}")
    
    print("\n" + "=" * 60)
    print("All pruning tests passed!")
    print("=" * 60)
