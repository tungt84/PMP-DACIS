"""
Prune-then-Meta-Learn-then-Prune (PMP) Framework
Three-stage training pipeline integrating pruning with meta-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
import copy
from tqdm import tqdm

from .backbone import create_backbone, FeatureExtractor
from .dacis import DACSIScorer, TaskComplexityEstimator


class PrototypicalHead(nn.Module):
    """Prototypical network head for few-shot classification"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        n_way: int,
        k_shot: int
    ) -> torch.Tensor:
        """
        Compute prototypical classification logits
        
        Args:
            support_features: [N*K, D] support set features
            query_features: [N*Q, D] query set features
            n_way: Number of classes
            k_shot: Number of support samples per class
            
        Returns:
            logits: [N*Q, N] classification logits
        """
        # Compute class prototypes
        support_features = support_features.view(n_way, k_shot, -1)
        prototypes = support_features.mean(dim=1)  # [N, D]
        
        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)  # [N*Q, N]
        
        # Return negative distances as logits
        return -distances


class MAMLHead(nn.Module):
    """MAML-style adaptation head"""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
    
    def adapt(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ) -> 'MAMLHead':
        """
        Adapt classifier on support set
        
        Args:
            support_features: Support set features
            support_labels: Support set labels
            inner_lr: Inner loop learning rate
            inner_steps: Number of adaptation steps
            
        Returns:
            adapted_head: Adapted classifier
        """
        adapted_head = copy.deepcopy(self)
        
        for _ in range(inner_steps):
            logits = adapted_head(support_features)
            loss = F.cross_entropy(logits, support_labels)
            
            grads = torch.autograd.grad(loss, adapted_head.parameters())
            
            for param, grad in zip(adapted_head.parameters(), grads):
                param.data -= inner_lr * grad
                
        return adapted_head


class PMPFramework(nn.Module):
    """
    Prune-then-Meta-Learn-then-Prune Framework
    
    Three-stage training:
        Stage 1: Initial pruning based on DACIS scores
        Stage 2: Episodic meta-learning on pruned network
        Stage 3: Refinement pruning using meta-gradients
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_way: int = 5,
        k_shot: int = 5,
        feature_dim: int = 512,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.device = device
        
        # Feature extractor
        self.backbone = create_backbone(backbone_name, pretrained=True)
        self.feature_dim = self.backbone.feature_dim
        
        # Classification heads
        self.proto_head = PrototypicalHead(self.feature_dim)
        self.maml_head = MAMLHead(self.feature_dim, n_way)
        
        # DACIS scorer
        self.dacis_scorer = DACSIScorer(device=device)
        
        # Pruning masks
        self.pruning_masks: Dict[str, torch.Tensor] = {}
        
        # Meta-gradient accumulator
        self.meta_gradients: Dict[str, torch.Tensor] = {}
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using backbone"""
        return self.backbone(x)
    
    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        support_labels: torch.Tensor,
        mode: str = "proto"
    ) -> torch.Tensor:
        """
        Forward pass for few-shot classification
        
        Args:
            support_images: [N*K, 3, H, W] support set
            query_images: [N*Q, 3, H, W] query set
            support_labels: [N*K] support labels
            mode: "proto" or "maml"
            
        Returns:
            logits: Classification logits for query set
        """
        # Extract features
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)
        
        if mode == "proto":
            return self.proto_head(
                support_features, query_features,
                self.n_way, self.k_shot
            )
        else:
            # MAML-style adaptation
            adapted_head = self.maml_head.adapt(
                support_features, support_labels
            )
            return adapted_head(query_features)
    
    # =========================================
    # Stage 1: Initial Pruning
    # =========================================
    
    def stage1_initial_pruning(
        self,
        dataloader,
        num_classes: int,
        prune_ratio: float = 0.4,
        finetune_epochs: int = 5,
        lr: float = 0.001
    ):
        """
        Stage 1: Conservative initial pruning based on DACIS scores
        
        Args:
            dataloader: Training data loader
            num_classes: Number of base classes
            prune_ratio: Initial pruning ratio (default 40%)
            finetune_epochs: Fine-tuning epochs after pruning
            lr: Learning rate for fine-tuning
        """
        print("=" * 50)
        print("Stage 1: Initial Pruning")
        print("=" * 50)
        
        # Compute DACIS scores
        criterion = nn.CrossEntropyLoss()
        self.dacis_scorer.compute_dacis_scores(
            self.backbone, dataloader, criterion, num_classes
        )
        
        # Generate pruning masks
        total_params_before = self._count_parameters()
        
        layer_names = list(self.dacis_scorer.dacis_scores.keys())
        for idx, layer_name in enumerate(layer_names):
            mask = self.dacis_scorer.get_pruning_mask(
                layer_name,
                prune_ratio=prune_ratio,
                layer_idx=idx,
                total_layers=len(layer_names)
            )
            self.pruning_masks[layer_name] = mask
        
        # Apply pruning
        self._apply_pruning_masks()
        
        total_params_after = self._count_active_parameters()
        print(f"  Parameters: {total_params_before:,} -> {total_params_after:,}")
        print(f"  Compression: {100*(1 - total_params_after/total_params_before):.1f}%")
        
        # Fine-tune
        if finetune_epochs > 0:
            print(f"  Fine-tuning for {finetune_epochs} epochs...")
            self._finetune(dataloader, finetune_epochs, lr)
    
    # =========================================
    # Stage 2: Meta-Learning
    # =========================================
    
    def stage2_meta_learning(
        self,
        episode_generator,
        num_episodes: int = 60000,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        meta_batch_size: int = 4
    ):
        """
        Stage 2: Episodic meta-training with gradient accumulation
        
        Args:
            episode_generator: Generator for N-way K-shot episodes
            num_episodes: Number of training episodes
            inner_lr: Inner loop learning rate (α)
            outer_lr: Outer loop learning rate (β)
            inner_steps: Inner loop adaptation steps
            meta_batch_size: Number of tasks per meta-batch
        """
        print("=" * 50)
        print("Stage 2: Meta-Learning")
        print("=" * 50)
        
        optimizer = Adam(self.parameters(), lr=outer_lr)
        scheduler = CosineAnnealingLR(optimizer, num_episodes // meta_batch_size)
        
        # Initialize meta-gradient accumulator
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                self.meta_gradients[name] = torch.zeros_like(param)
        
        episode_iter = iter(episode_generator)
        pbar = tqdm(range(num_episodes // meta_batch_size), desc="Meta-training")
        
        for iteration in pbar:
            meta_loss = 0.0
            
            for _ in range(meta_batch_size):
                try:
                    episode = next(episode_iter)
                except StopIteration:
                    episode_iter = iter(episode_generator)
                    episode = next(episode_iter)
                
                support_images, support_labels, query_images, query_labels = episode
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                # Inner loop: adapt on support set
                adapted_params = self._inner_loop_adapt(
                    support_images, support_labels,
                    inner_lr, inner_steps
                )
                
                # Outer loop: evaluate on query set
                query_features = self._forward_with_params(query_images, adapted_params)
                logits = self.maml_head(query_features)
                loss = F.cross_entropy(logits, query_labels)
                
                meta_loss += loss
            
            # Meta-update
            optimizer.zero_grad()
            meta_loss.backward()
            
            # Accumulate meta-gradients for Stage 3
            for name, param in self.backbone.named_parameters():
                if param.grad is not None and name in self.meta_gradients:
                    self.meta_gradients[name] += param.grad.abs()
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            if iteration % 100 == 0:
                pbar.set_postfix({"loss": meta_loss.item() / meta_batch_size})
    
    def _inner_loop_adapt(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        inner_lr: float,
        inner_steps: int
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation"""
        # Clone parameters
        adapted_params = {
            name: param.clone() 
            for name, param in self.backbone.named_parameters()
        }
        
        for _ in range(inner_steps):
            features = self._forward_with_params(support_images, adapted_params)
            logits = self.maml_head(features)
            loss = F.cross_entropy(logits, support_labels)
            
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            adapted_params = {
                name: param - inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with given parameters"""
        # For simplicity, use functional forward with params
        # In practice, implement proper functional backbone
        return self.backbone(x)
    
    # =========================================
    # Stage 3: Refinement Pruning
    # =========================================
    
    def stage3_refinement_pruning(
        self,
        dataloader,
        additional_prune_ratio: float = 0.38,
        gamma: float = 0.1,
        finetune_epochs: int = 10,
        lr: float = 0.0001
    ):
        """
        Stage 3: Refined pruning using meta-gradients
        
        DACIS_refined = DACIS · (1 + γ · ||G_meta||)
        
        Args:
            dataloader: Training data loader
            additional_prune_ratio: Additional pruning ratio
            gamma: Meta-gradient weight
            finetune_epochs: Fine-tuning epochs after pruning
            lr: Learning rate for fine-tuning
        """
        print("=" * 50)
        print("Stage 3: Refinement Pruning")
        print("=" * 50)
        
        # Normalize meta-gradients
        meta_grad_norms = {}
        for name, grad in self.meta_gradients.items():
            if 'conv' in name.lower() or 'weight' in name.lower():
                # Compute per-channel gradient norms
                if grad.dim() >= 2:
                    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1)
                    meta_grad_norms[name] = grad_norm
        
        # Refine DACIS scores
        refined_scores = self.dacis_scorer.refine_with_meta_gradients(
            meta_grad_norms, gamma
        )
        
        # Update pruning masks with refined scores
        total_params_before = self._count_active_parameters()
        
        for layer_name, scores in refined_scores.items():
            current_mask = self.pruning_masks.get(layer_name, 
                torch.ones(scores.size(0), dtype=torch.bool))
            
            # Additional pruning on remaining channels
            active_channels = current_mask.sum().item()
            channels_to_keep = max(8, int(active_channels * (1 - additional_prune_ratio)))
            
            # Get top channels by refined score
            masked_scores = scores.clone()
            masked_scores[~current_mask] = float('-inf')
            _, top_indices = torch.topk(masked_scores, channels_to_keep)
            
            new_mask = torch.zeros_like(current_mask)
            new_mask[top_indices] = True
            self.pruning_masks[layer_name] = new_mask
        
        # Apply refined pruning
        self._apply_pruning_masks()
        
        total_params_after = self._count_active_parameters()
        print(f"  Parameters: {total_params_before:,} -> {total_params_after:,}")
        print(f"  Additional compression: {100*(1 - total_params_after/total_params_before):.1f}%")
        
        # Final fine-tuning
        if finetune_epochs > 0:
            print(f"  Fine-tuning for {finetune_epochs} epochs...")
            self._finetune(dataloader, finetune_epochs, lr)
    
    # =========================================
    # Utility Methods
    # =========================================
    
    def _apply_pruning_masks(self):
        """Apply pruning masks to model weights"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.pruning_masks:
                mask = self.pruning_masks[name]
                # Zero out pruned channels
                with torch.no_grad():
                    module.weight.data[~mask] = 0
                    if module.bias is not None:
                        module.bias.data[~mask] = 0
    
    def _finetune(self, dataloader, epochs: int, lr: float):
        """Fine-tune model after pruning"""
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                features = self.extract_features(images)
                logits = self.maml_head(features)
                loss = criterion(logits, labels)
                loss.backward()
                
                # Maintain sparsity
                self._apply_pruning_masks()
                
                optimizer.step()
                total_loss += loss.item()
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def _count_active_parameters(self) -> int:
        """Count non-zero parameters"""
        total = 0
        for p in self.parameters():
            total += (p != 0).sum().item()
        return total
    
    def get_compression_ratio(self) -> float:
        """Get current compression ratio"""
        total = self._count_parameters()
        active = self._count_active_parameters()
        return 1 - active / total
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'backbone_state': self.backbone.state_dict(),
            'proto_head_state': self.proto_head.state_dict(),
            'maml_head_state': self.maml_head.state_dict(),
            'pruning_masks': self.pruning_masks,
            'dacis_scores': self.dacis_scorer.dacis_scores,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state'])
        self.proto_head.load_state_dict(checkpoint['proto_head_state'])
        self.maml_head.load_state_dict(checkpoint['maml_head_state'])
        self.pruning_masks = checkpoint['pruning_masks']
        self.dacis_scorer.dacis_scores = checkpoint['dacis_scores']


def create_pmp_model(config: dict) -> PMPFramework:
    """
    Factory function to create PMP model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: PMPFramework instance
    """
    return PMPFramework(
        backbone_name=config['model']['backbone'],
        n_way=config['fsl']['n_way'],
        k_shot=config['fsl']['k_shot'],
        feature_dim=config['model'].get('num_channels', 512),
        device=config['training']['device']
    )


if __name__ == "__main__":
    # Test PMP Framework
    print("Testing PMP Framework...")
    
    model = PMPFramework(
        backbone_name="resnet18",
        n_way=5,
        k_shot=5,
        device="cpu"
    )
    
    print(f"  Total parameters: {model._count_parameters():,}")
    print(f"  Feature dimension: {model.feature_dim}")
    
    # Test forward pass
    support = torch.randn(25, 3, 224, 224)  # 5-way 5-shot
    query = torch.randn(75, 3, 224, 224)    # 15 queries per class
    labels = torch.arange(5).repeat_interleave(5)
    
    logits = model(support, query, labels, mode="proto")
    print(f"  Output shape: {logits.shape}")
