"""
Disease-Aware Channel Importance Scoring (DACIS)
Combines gradient norm, feature variance, and Fisher's discriminant
for disease-aware channel pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class DACSIScorer:
    """
    Disease-Aware Channel Importance Scoring (DACIS)
    
    DACIS = λ₁·G + λ₂·V + λ₃·D
    
    Where:
        G: Gradient norm contribution
        V: Feature variance contribution  
        D: Fisher's discriminant (disease discriminability)
    """
    
    def __init__(
        self,
        lambda_gradient: float = 0.3,
        lambda_variance: float = 0.2,
        lambda_fisher: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize DACIS scorer
        
        Args:
            lambda_gradient: Weight for gradient norm (λ₁)
            lambda_variance: Weight for feature variance (λ₂)
            lambda_fisher: Weight for Fisher's discriminant (λ₃)
            device: Computation device
        """
        assert abs(lambda_gradient + lambda_variance + lambda_fisher - 1.0) < 1e-6, \
            "Weights must sum to 1"
        
        self.lambda_g = lambda_gradient
        self.lambda_v = lambda_variance
        self.lambda_d = lambda_fisher
        self.device = device
        
        # Storage for computed scores
        self.gradient_scores: Dict[str, torch.Tensor] = {}
        self.variance_scores: Dict[str, torch.Tensor] = {}
        self.fisher_scores: Dict[str, torch.Tensor] = {}
        self.dacis_scores: Dict[str, torch.Tensor] = {}
        
    def compute_gradient_norm(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient norm contribution G for each channel
        
        G_ℓ^(c) = (1/|D|) Σ ||∂L/∂W_ℓ^(c)||_F
        
        Args:
            model: Neural network model
            dataloader: Data loader for computing gradients
            criterion: Loss function
            
        Returns:
            gradient_scores: Dict mapping layer names to gradient scores
        """
        model.train()
        gradient_accum = {}
        count = 0
        
        for batch in dataloader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Accumulate gradients for each conv layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.weight.grad is not None:
                        # Gradient norm per output channel
                        grad = module.weight.grad
                        # Shape: [C_out, C_in, K, K] -> compute norm over [C_in, K, K]
                        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1)
                        
                        if name not in gradient_accum:
                            gradient_accum[name] = grad_norm.clone()
                        else:
                            gradient_accum[name] += grad_norm
                            
            count += 1
            
        # Average gradients
        for name in gradient_accum:
            gradient_accum[name] /= count
            # Normalize to [0, 1]
            gradient_accum[name] = self._normalize(gradient_accum[name])
            
        self.gradient_scores = gradient_accum
        return gradient_accum
    
    def compute_feature_variance(
        self,
        model: nn.Module,
        dataloader,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature variance contribution V for each channel
        
        V_ℓ^(c) = Var_{x∈D}[GAP(a_ℓ^(c)(x))]
        
        Args:
            model: Neural network model
            dataloader: Data loader
            layer_names: Specific layers to compute (None = all conv layers)
            
        Returns:
            variance_scores: Dict mapping layer names to variance scores
        """
        model.eval()
        activations = {}
        hooks = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                # Global average pooling
                gap = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
                activations[name].append(gap.detach().cpu())
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if layer_names is None or name in layer_names:
                    hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        # Collect activations
        with torch.no_grad():
            for batch in dataloader:
                images, _ = batch
                images = images.to(self.device)
                model(images)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute variance for each channel
        variance_scores = {}
        for name, acts in activations.items():
            acts_tensor = torch.cat(acts, dim=0)  # [N, C]
            variance = torch.var(acts_tensor, dim=0)  # [C]
            variance_scores[name] = self._normalize(variance)
            
        self.variance_scores = variance_scores
        return variance_scores
    
    def compute_fisher_discriminant(
        self,
        model: nn.Module,
        dataloader,
        num_classes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher's Linear Discriminant score D for each channel
        
        D_ℓ^(c) = S_B / S_W (between-class / within-class scatter)
        
        Args:
            model: Neural network model
            dataloader: Data loader with class labels
            num_classes: Number of classes
            
        Returns:
            fisher_scores: Dict mapping layer names to Fisher scores
        """
        model.eval()
        class_activations = {c: {} for c in range(num_classes)}
        hooks = []
        current_labels = None
        
        def get_activation_hook(name):
            def hook(module, input, output):
                gap = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
                for i, label in enumerate(current_labels):
                    label = label.item()
                    if name not in class_activations[label]:
                        class_activations[label][name] = []
                    class_activations[label][name].append(gap[i:i+1].detach().cpu())
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        # Collect activations per class
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                current_labels = labels
                model(images)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute Fisher's discriminant for each layer
        fisher_scores = {}
        layer_names = list(class_activations[0].keys())
        
        for layer_name in layer_names:
            # Compute class means
            class_means = []
            class_counts = []
            all_activations = []
            
            for c in range(num_classes):
                if layer_name in class_activations[c] and len(class_activations[c][layer_name]) > 0:
                    acts = torch.cat(class_activations[c][layer_name], dim=0)
                    class_means.append(acts.mean(dim=0))
                    class_counts.append(acts.size(0))
                    all_activations.append(acts)
            
            if len(class_means) < 2:
                continue
                
            class_means = torch.stack(class_means)  # [K, C]
            global_mean = torch.cat(all_activations, dim=0).mean(dim=0)  # [C]
            
            # Between-class scatter: S_B = Σ n_k (μ_k - μ)²
            between_class = torch.zeros(class_means.size(1))
            for k, (mean, count) in enumerate(zip(class_means, class_counts)):
                between_class += count * (mean - global_mean) ** 2
            
            # Within-class scatter: S_W = Σ Σ (x - μ_k)²
            within_class = torch.zeros(class_means.size(1))
            for k, (mean, acts) in enumerate(zip(class_means, 
                [torch.cat(class_activations[c][layer_name], dim=0) 
                 for c in range(num_classes) if layer_name in class_activations[c]])):
                within_class += ((acts - mean) ** 2).sum(dim=0)
            
            # Fisher's criterion: D = S_B / S_W
            fisher = between_class / (within_class + 1e-8)
            fisher_scores[layer_name] = self._normalize(fisher)
        
        self.fisher_scores = fisher_scores
        return fisher_scores
    
    def compute_dacis_scores(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        num_classes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete DACIS scores for all channels
        
        DACIS = λ₁·G + λ₂·V + λ₃·D
        
        Args:
            model: Neural network model
            dataloader: Data loader
            criterion: Loss function
            num_classes: Number of classes
            
        Returns:
            dacis_scores: Dict mapping layer names to DACIS scores
        """
        print("Computing gradient norm scores...")
        self.compute_gradient_norm(model, dataloader, criterion)
        
        print("Computing feature variance scores...")
        self.compute_feature_variance(model, dataloader)
        
        print("Computing Fisher's discriminant scores...")
        self.compute_fisher_discriminant(model, dataloader, num_classes)
        
        # Combine scores
        print("Computing final DACIS scores...")
        dacis_scores = {}
        
        for layer_name in self.gradient_scores.keys():
            if layer_name in self.variance_scores and layer_name in self.fisher_scores:
                G = self.gradient_scores[layer_name]
                V = self.variance_scores[layer_name]
                D = self.fisher_scores[layer_name]
                
                dacis = self.lambda_g * G + self.lambda_v * V + self.lambda_d * D
                dacis_scores[layer_name] = dacis
        
        self.dacis_scores = dacis_scores
        return dacis_scores
    
    def get_pruning_mask(
        self,
        layer_name: str,
        prune_ratio: float,
        tau_base: float = 0.5,
        layer_idx: int = 0,
        total_layers: int = 1,
        alpha: float = 0.5,
        task_complexity: float = 0.5,
        beta: float = 2.0
    ) -> torch.Tensor:
        """
        Get pruning mask for a layer using layer-adaptive threshold
        
        τ_ℓ = τ_base · (1 + α · ℓ/L) · exp(-β · C_task)
        
        Args:
            layer_name: Name of the layer
            prune_ratio: Target pruning ratio
            tau_base: Base pruning threshold
            layer_idx: Current layer index
            total_layers: Total number of layers
            alpha: Layer depth factor
            task_complexity: Task complexity score
            beta: Task complexity sensitivity
            
        Returns:
            mask: Binary mask (1 = keep, 0 = prune)
        """
        if layer_name not in self.dacis_scores:
            raise ValueError(f"No DACIS scores for layer {layer_name}")
        
        scores = self.dacis_scores[layer_name]
        num_channels = scores.size(0)
        
        # Layer-adaptive threshold
        tau = tau_base * (1 + alpha * layer_idx / total_layers) * \
              np.exp(-beta * task_complexity)
        
        # Number of channels to keep
        num_keep = max(8, int(num_channels * (1 - prune_ratio)))
        
        # Keep channels with highest DACIS scores
        _, indices = torch.topk(scores, num_keep)
        mask = torch.zeros(num_channels, dtype=torch.bool)
        mask[indices] = True
        
        return mask
    
    def refine_with_meta_gradients(
        self,
        meta_gradients: Dict[str, torch.Tensor],
        gamma: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Refine DACIS scores with meta-gradients (Stage 3)
        
        DACIS_refined = DACIS · (1 + γ · ||G_meta||₂)
        
        Args:
            meta_gradients: Meta-gradients from Stage 2
            gamma: Meta-gradient weight
            
        Returns:
            refined_scores: Refined DACIS scores
        """
        refined_scores = {}
        
        for layer_name, dacis in self.dacis_scores.items():
            if layer_name in meta_gradients:
                meta_grad = meta_gradients[layer_name]
                meta_grad_norm = torch.norm(meta_grad, dim=-1) if meta_grad.dim() > 1 else meta_grad.abs()
                meta_grad_norm = self._normalize(meta_grad_norm)
                
                refined = dacis * (1 + gamma * meta_grad_norm)
                refined_scores[layer_name] = refined
            else:
                refined_scores[layer_name] = dacis
                
        return refined_scores
    
    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range"""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)


class TaskComplexityEstimator:
    """
    Estimate task complexity based on prototype similarity
    
    C_task = 1 - (1/C(N,2)) Σ_{i<j} cos(z̄_i, z̄_j)
    """
    
    @staticmethod
    def compute(prototypes: torch.Tensor) -> float:
        """
        Compute task complexity from class prototypes
        
        Args:
            prototypes: Class prototypes [N, D]
            
        Returns:
            complexity: Task complexity score [0, 1]
        """
        N = prototypes.size(0)
        if N < 2:
            return 0.5
        
        # Normalize prototypes
        prototypes = F.normalize(prototypes, dim=1)
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(prototypes, prototypes.t())
        
        # Average upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        avg_similarity = similarity_matrix[mask].mean().item()
        
        # Complexity = 1 - avg_similarity
        complexity = 1 - avg_similarity
        
        return complexity


if __name__ == "__main__":
    # Test DACIS scorer
    print("Testing DACIS Scorer...")
    
    scorer = DACSIScorer(
        lambda_gradient=0.3,
        lambda_variance=0.2,
        lambda_fisher=0.5,
        device="cpu"
    )
    
    print(f"  λ_G: {scorer.lambda_g}")
    print(f"  λ_V: {scorer.lambda_v}")
    print(f"  λ_D: {scorer.lambda_d}")
    
    # Test task complexity
    print("\nTesting Task Complexity Estimator...")
    prototypes = torch.randn(5, 512)
    complexity = TaskComplexityEstimator.compute(prototypes)
    print(f"  Task complexity: {complexity:.4f}")
