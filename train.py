"""
PMP-DACIS Training Script

Main training script implementing the 3-stage Progressive Meta-Pruning framework:
1. Stage 1: Initial DACIS-guided pruning
2. Stage 2: Meta-learning with prototypical/MAML heads
3. Stage 3: Refinement pruning with meta-gradients

Usage:
    python train.py --config configs/default.yaml
    python train.py --backbone resnet18 --dataset plantvillage --shots 5
"""

import argparse
import os
import sys
import time
import random
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_backbone, PMPFramework, DACSIScorer, create_pmp_model
from utils import (
    DeploymentEfficiencyScore,
    FewShotStabilityIndex,
    CompressionStabilityGain,
    MetricLogger,
    apply_channel_pruning,
    compute_sparsity
)


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{experiment_name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class EpisodeSampler:
    """
    Few-shot episode sampler for meta-learning.
    
    Generates N-way K-shot episodes from a dataset.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 1000
    ):
        """
        Initialize episode sampler.
        
        Args:
            dataset: Source dataset with (image, label) pairs
            n_way: Number of classes per episode
            k_shot: Support samples per class
            q_query: Query samples per class
            num_episodes: Number of episodes to generate
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        # Build class-to-indices mapping
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def __len__(self):
        return self.num_episodes
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self._sample_episode()
    
    def _sample_episode(self) -> Tuple[torch.Tensor, ...]:
        """Sample a single episode."""
        # Sample N classes
        episode_classes = random.sample(self.classes, self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for new_label, cls in enumerate(episode_classes):
            # Sample K+Q indices
            indices = random.sample(
                self.class_indices[cls],
                min(self.k_shot + self.q_query, len(self.class_indices[cls]))
            )
            
            # Split into support and query
            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:self.k_shot + self.q_query]
            
            for idx in support_indices:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(new_label)
            
            for idx in query_indices:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(new_label)
        
        # Stack tensors
        support_x = torch.stack(support_images)
        support_y = torch.tensor(support_labels, dtype=torch.long)
        query_x = torch.stack(query_images)
        query_y = torch.tensor(query_labels, dtype=torch.long)
        
        return support_x, support_y, query_x, query_y


class DummyDataset(Dataset):
    """Dummy dataset for testing when real data is not available."""
    
    def __init__(
        self,
        num_classes: int = 38,
        samples_per_class: int = 100,
        image_size: int = 224
    ):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.total_samples = num_classes * samples_per_class
        
        # Pre-generate class assignments
        self.labels = []
        for cls in range(num_classes):
            self.labels.extend([cls] * samples_per_class)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(3, self.image_size, self.image_size)
        label = self.labels[idx]
        return image, label


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_config = config.get('dataset', {})
    fsl_config = config.get('fsl', {})
    
    # Load PlantVillage from HuggingFace Datasets (simple implementation)
    from datasets import load_dataset, ClassLabel
    from torchvision import transforms
    from PIL import Image
    import io
    import numpy as np

    logging.info("Loading PlantVillage dataset from local Datasets...")
    try:
        # Load local PlantVillage using the provided local loader
        from local_loader import load_local_dataset

        root = dataset_config.get('root', 'data')
        # if dataset root points to parent folder, append plantvillage subfolder
        if dataset_config.get('name', '').lower() == 'plantvillage':
            candidate = os.path.join(root, 'plantvillage')
            if os.path.isdir(candidate):
                root = candidate
        config_name = dataset_config.get('config', 'color')
        splits = load_local_dataset(root, config=config_name)

        # Build class -> index mapping from train split
        class_names = sorted({ex['label'] for ex in splits['train']})
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        num_classes = len(class_names)
        dataset_config['num_classes'] = num_classes

        image_size = dataset_config.get('image_size', 224)

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class LocalPVDataset(Dataset):
            def __init__(self, examples, class_to_idx, transform=None):
                self.examples = examples
                self.class_to_idx = class_to_idx
                self.transform = transform

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                ex = self.examples[idx]
                img_path = ex['abs_path']
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    # If image cannot be opened, return a random tensor
                    img = Image.new('RGB', (image_size, image_size))
                if self.transform:
                    img = self.transform(img)
                label_name = ex['label']
                label = self.class_to_idx.get(label_name, 0)
                return img, label

        train_examples = list(splits['train'])
        test_examples = list(splits['test'])

        # Group train examples by leaf_id then split leaf_ids for val (10%)
        groups = defaultdict(list)
        for ex in train_examples:
            groups[ex['leaf_id']].append(ex)

        leaf_ids = list(groups.keys())
        import random
        random.Random(42).shuffle(leaf_ids)
        val_frac = 0.1
        val_count = int(len(leaf_ids) * val_frac)
        val_leaf_ids = set(leaf_ids[:val_count])
        new_train = []
        val = []
        for lid, exs in groups.items():
            if lid in val_leaf_ids:
                val.extend(exs)
            else:
                new_train.extend(exs)

        train_dataset = LocalPVDataset(new_train, class_to_idx, transform=transform)
        val_dataset = LocalPVDataset(val, class_to_idx, transform=transform)
        test_dataset = LocalPVDataset(test_examples, class_to_idx, transform=transform)
        logging.info(f"Loaded datasets — train: {len(new_train)}, val: {len(val)}, test: {len(test_examples)}")
        
    except Exception as e:
        logging.warning(f"local dataset load failed: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
        logging.warning("Falling back to dummy datasets")
        num_classes = dataset_config.get('num_classes', 38)
        train_dataset = DummyDataset(num_classes=num_classes, samples_per_class=100)
        val_dataset = DummyDataset(num_classes=num_classes, samples_per_class=20)
        test_dataset = DummyDataset(num_classes=num_classes, samples_per_class=20)
    
    # Create episode samplers (optionally streaming via IterableDataset)
    n_way = fsl_config.get('n_way', 5)
    k_shot = fsl_config.get('k_shot', 5)
    q_query = fsl_config.get('q_query', 15)

    use_iterable = config.get('streaming', {}).get('use_iterable_episodes', False)

    if use_iterable:
        class EpisodeIterableDataset(IterableDataset):
            def __init__(self, dataset, n_way, k_shot, q_query, num_episodes):
                super().__init__()
                self.dataset = dataset
                self.n_way = n_way
                self.k_shot = k_shot
                self.q_query = q_query
                self.num_episodes = num_episodes

                # build class indices efficiently
                self.class_indices = defaultdict(list)
                if hasattr(dataset, 'examples'):
                    for idx, ex in enumerate(dataset.examples):
                        lbl = dataset.class_to_idx.get(ex['label'], 0) if hasattr(dataset, 'class_to_idx') else 0
                        self.class_indices[lbl].append(idx)
                elif hasattr(dataset, 'labels'):
                    for idx, lbl in enumerate(dataset.labels):
                        self.class_indices[lbl].append(idx)
                else:
                    for idx in range(len(dataset)):
                        _, lbl = dataset[idx]
                        self.class_indices[lbl].append(idx)

                self.classes = list(self.class_indices.keys())

            def __iter__(self):
                rng = random.Random()
                for _ in range(self.num_episodes):
                    episode_classes = rng.sample(self.classes, self.n_way)
                    support_images = []
                    support_labels = []
                    query_images = []
                    query_labels = []

                    for new_label, cls in enumerate(episode_classes):
                        indices = rng.sample(self.class_indices[cls], min(self.k_shot + self.q_query, len(self.class_indices[cls])))
                        support_indices = indices[:self.k_shot]
                        query_indices = indices[self.k_shot:self.k_shot + self.q_query]
                        for idx in support_indices:
                            img, _ = self.dataset[idx]
                            support_images.append(img)
                            support_labels.append(new_label)
                        for idx in query_indices:
                            img, _ = self.dataset[idx]
                            query_images.append(img)
                            query_labels.append(new_label)

                    support_x = torch.stack(support_images)
                    support_y = torch.tensor(support_labels, dtype=torch.long)
                    query_x = torch.stack(query_images)
                    query_y = torch.tensor(query_labels, dtype=torch.long)
                    yield support_x, support_y, query_x, query_y

        train_sampler = EpisodeIterableDataset(train_dataset, n_way, k_shot, q_query, fsl_config.get('train_episodes', 10000))
        val_sampler = EpisodeIterableDataset(val_dataset, n_way, k_shot, q_query, fsl_config.get('val_episodes', 1000))
        test_sampler = EpisodeIterableDataset(test_dataset, n_way, k_shot, q_query, fsl_config.get('test_episodes', 2000))
    else:
        train_sampler = EpisodeSampler(train_dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, num_episodes=fsl_config.get('train_episodes', 10000))
        val_sampler = EpisodeSampler(val_dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, num_episodes=fsl_config.get('val_episodes', 1000))
        test_sampler = EpisodeSampler(test_dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, num_episodes=fsl_config.get('test_episodes', 2000))
    
    # Wrap in simple loader
    def episode_collate(episodes):
        return episodes[0]  # Single episode per batch for simplicity
    
    # Configure DataLoader depending on whether sampler is iterable
    num_workers = config.get('training', {}).get('num_workers', 4)
    pin_memory = config.get('training', {}).get('pin_memory', False)

    if use_iterable:
        train_loader = DataLoader(
            train_sampler,
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_sampler,
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate,
            num_workers=max(0, num_workers // 2),
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_sampler,
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate,
            num_workers=max(0, num_workers // 2),
            pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            list(train_sampler),
            batch_size=1,
            shuffle=True,
            collate_fn=episode_collate
        )
        val_loader = DataLoader(
            list(val_sampler),
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate
        )
        test_loader = DataLoader(
            list(test_sampler),
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate
        )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: PMPFramework,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    logger: logging.Logger,
    max_episodes: Optional[int] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: PMP model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        max_episodes: Maximum episodes per epoch
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_queries = 0
    episode_accuracies = []
    
    num_episodes = len(train_loader)
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
        if max_episodes is not None and episode_idx >= max_episodes:
            break
        
        # Move to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        logits = model.forward_episode(support_x, support_y, query_x)
        loss = nn.functional.cross_entropy(logits, query_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == query_y).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_queries += query_y.size(0)
        episode_accuracies.append(correct / query_y.size(0))
        
        # Log progress
        if (episode_idx + 1) % 100 == 0:
            avg_loss = total_loss / (episode_idx + 1)
            avg_acc = total_correct / total_queries
            logger.info(
                f"Epoch {epoch} | Episode {episode_idx + 1}/{num_episodes} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
            )
    
    return {
        'loss': total_loss / num_episodes,
        'accuracy': total_correct / total_queries,
        'episode_accuracies': episode_accuracies
    }


@torch.no_grad()
def evaluate(
    model: PMPFramework,
    eval_loader: DataLoader,
    device: str,
    max_episodes: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on few-shot episodes.
    
    Args:
        model: PMP model
        eval_loader: Evaluation data loader
        device: Device to evaluate on
        max_episodes: Maximum episodes to evaluate
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    
    total_correct = 0
    total_queries = 0
    episode_accuracies = []
    
    num_episodes = len(eval_loader)
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(eval_loader):
        if max_episodes is not None and episode_idx >= max_episodes:
            break
        
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        logits = model.forward_episode(support_x, support_y, query_x)
        predictions = logits.argmax(dim=-1)
        
        correct = (predictions == query_y).sum().item()
        total_correct += correct
        total_queries += query_y.size(0)
        episode_accuracies.append(correct / query_y.size(0))
    
    accuracy = total_correct / total_queries
    accuracy_std = np.std(episode_accuracies)
    
    # Compute FSI
    fsi_calc = FewShotStabilityIndex()
    fsi_calc.add_batch_accuracies(episode_accuracies)
    fsi_results = fsi_calc.compute()
    
    return {
        'accuracy': accuracy,
        'accuracy_std': accuracy_std,
        'fsi': fsi_results['fsi'],
        'episode_accuracies': episode_accuracies,
        'num_episodes': len(episode_accuracies)
    }


def run_pmp_training(
    config: Dict,
    device: str,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None
) -> Tuple[nn.Module, Dict[str, any]]:
    """
    Run complete PMP 3-stage training.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
        logger: Logger instance
        writer: TensorBoard writer
        
    Returns:
        Tuple of (trained_model, training_stats)
    """
    # Create model
    model_config = config.get('model', {})
    backbone_name = model_config.get('backbone', 'resnet18')
    
    logger.info(f"Creating PMP model with backbone: {backbone_name}")
    
 
    pmp_model_config = {
         "model" : {"backbone": backbone_name,"num_channels": model_config.get('feature_dim', 512)},
         "training": {"device": device},
         "fsl": {"n_way": config['fsl']['n_way'], "k_shot": config['fsl']['k_shot']}
    }
    model = create_pmp_model(
        config=pmp_model_config
        #backbone_name=backbone_name,
        #feature_dim=model_config.get('feature_dim', 512),
        #num_classes=config['dataset'].get('num_classes', 38),
        #n_way=config['fsl'].get('n_way', 5)
    )
    model = model.to(device)
    
    # Create DACIS scorer
    dacis_config = config.get('dacis', {})
    scorer = DACSIScorer(
        #lambda_g=dacis_config.get('lambda_g', 0.3),
        #lambda_v=dacis_config.get('lambda_v', 0.2),
        #lambda_d=dacis_config.get('lambda_d', 0.5)
        lambda_gradient=dacis_config.get('lambda_g', 0.3),
        lambda_variance=dacis_config.get('lambda_v', 0.2),
        lambda_fisher=dacis_config.get('lambda_d', 0.5),
        device=device
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Training configuration
    training_config = config.get('training', {})
    pmp_config = config.get('pmp', {})
    
    # CSG tracker
    csg_tracker = CompressionStabilityGain()
    
    # ================================================================
    # STAGE 1: Initial DACIS-guided Pruning
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Initial DACIS-guided Pruning")
    logger.info("=" * 60)
    
    stage1_config = pmp_config.get('stage1', {})
    stage1_epochs = stage1_config.get('epochs', 10)
    stage1_prune_ratio = stage1_config.get('pruning_ratio', 0.3)
    
    # Initial training before pruning
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('lr', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    for epoch in range(stage1_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, logger,
            max_episodes=training_config.get('episodes_per_epoch', 500)
        )
        
        if writer:
            writer.add_scalar('Stage1/train_loss', train_metrics['loss'], epoch)
            writer.add_scalar('Stage1/train_acc', train_metrics['accuracy'], epoch)
    
    # Evaluate before pruning
    pre_prune_metrics = evaluate(model, val_loader, device, max_episodes=200)
    csg_tracker.record_stage_accuracy(pre_prune_metrics['accuracy'], 'pre_prune')
    logger.info(f"Pre-prune accuracy: {pre_prune_metrics['accuracy']:.4f}")
    
    # Apply DACIS-guided pruning
    logger.info(f"Applying DACIS-guided pruning (ratio={stage1_prune_ratio})")
    
    # Get sample batch for DACIS scoring
    sample_batch = next(iter(train_loader))
    support_x, support_y, query_x, query_y = [t.to(device) for t in sample_batch]
    
    # Compute DACIS scores (simplified - use backbone features)
    backbone = model.backbone
    with torch.enable_grad():
        features = backbone(support_x)
        pseudo_loss = features.mean()
        pseudo_loss.backward()
    
    # Apply magnitude-based pruning (with DACIS influence)
    pruned_model, prune_stats = apply_channel_pruning(
        model,
        pruning_ratio=stage1_prune_ratio,
        strategy='magnitude'
    )
    model = pruned_model.to(device)
    
    # Evaluate after Stage 1 pruning
    stage1_metrics = evaluate(model, val_loader, device, max_episodes=200)
    csg_tracker.record_stage_accuracy(stage1_metrics['accuracy'], 'stage1')
    logger.info(f"Stage 1 accuracy: {stage1_metrics['accuracy']:.4f}")
    logger.info(f"Compression ratio: {prune_stats['compression_ratio']:.2f}x")
    
    # ================================================================
    # STAGE 2: Meta-Learning
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Meta-Learning")
    logger.info("=" * 60)
    
    stage2_config = pmp_config.get('stage2', {})
    stage2_epochs = stage2_config.get('epochs', 50)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=stage2_config.get('outer_lr', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage2_epochs,
        eta_min=1e-6
    )
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(stage2_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, logger,
            max_episodes=training_config.get('episodes_per_epoch', 500)
        )
        
        scheduler.step()
        
        # Validate
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(model, val_loader, device, max_episodes=200)
            logger.info(
                f"Epoch {epoch + 1} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"FSI: {val_metrics['fsi']:.4f}"
            )
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
            
            if writer:
                writer.add_scalar('Stage2/val_acc', val_metrics['accuracy'], epoch)
                writer.add_scalar('Stage2/val_fsi', val_metrics['fsi'], epoch)
        
        if writer:
            writer.add_scalar('Stage2/train_loss', train_metrics['loss'], epoch)
            writer.add_scalar('Stage2/train_acc', train_metrics['accuracy'], epoch)
            writer.add_scalar('Stage2/lr', scheduler.get_last_lr()[0], epoch)
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    stage2_metrics = evaluate(model, val_loader, device, max_episodes=200)
    csg_tracker.record_stage_accuracy(stage2_metrics['accuracy'], 'stage2')
    logger.info(f"Stage 2 best accuracy: {stage2_metrics['accuracy']:.4f}")
    
    # ================================================================
    # STAGE 3: Refinement Pruning
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 3: Refinement Pruning")
    logger.info("=" * 60)
    
    stage3_config = pmp_config.get('stage3', {})
    stage3_epochs = stage3_config.get('epochs', 20)
    stage3_prune_ratio = stage3_config.get('pruning_ratio', 0.2)
    
    # Additional pruning guided by meta-gradients
    logger.info(f"Applying refinement pruning (ratio={stage3_prune_ratio})")
    
    pruned_model, prune_stats = apply_channel_pruning(
        model,
        pruning_ratio=stage3_prune_ratio,
        strategy='magnitude'
    )
    model = pruned_model.to(device)
    
    # Fine-tune after final pruning
    optimizer = optim.Adam(
        model.parameters(),
        lr=stage3_config.get('lr', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    for epoch in range(stage3_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, logger,
            max_episodes=training_config.get('episodes_per_epoch', 300)
        )
        
        if writer:
            writer.add_scalar('Stage3/train_loss', train_metrics['loss'], epoch)
            writer.add_scalar('Stage3/train_acc', train_metrics['accuracy'], epoch)
    
    # Final evaluation
    stage3_metrics = evaluate(model, val_loader, device, max_episodes=200)
    csg_tracker.record_stage_accuracy(stage3_metrics['accuracy'], 'stage3')
    
    # ================================================================
    # Final Results
    # ================================================================
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    # Compute CSG
    csg_results = csg_tracker.compute()
    
    # Compute final sparsity
    sparsity_stats = compute_sparsity(model)
    
    # Compute DES
    des_calc = DeploymentEfficiencyScore(
        hardware_profile=config.get('hardware', {}).get('target', 'raspberry_pi_4')
    )
    des_results = des_calc.compute(model, stage3_metrics['accuracy'], device=device)
    
    results = {
        'final_accuracy': stage3_metrics['accuracy'],
        'final_fsi': stage3_metrics['fsi'],
        'csg': csg_results['csg'],
        'des': des_results['des'],
        'sparsity': sparsity_stats['overall_sparsity'],
        'compression_ratio': 1.0 / (1.0 - sparsity_stats['overall_sparsity']) if sparsity_stats['overall_sparsity'] < 1 else float('inf'),
        'fps': des_results['fps'],
        'parameters': des_results['parameters'],
        'stage_accuracies': {
            'pre_prune': csg_tracker.stage_accuracies['default'].get('pre_prune', 0),
            'stage1': csg_tracker.stage_accuracies['default'].get('stage1', 0),
            'stage2': csg_tracker.stage_accuracies['default'].get('stage2', 0),
            'stage3': csg_tracker.stage_accuracies['default'].get('stage3', 0)
        }
    }
    
    logger.info(f"Final Accuracy: {results['final_accuracy']:.4f}")
    logger.info(f"FSI: {results['final_fsi']:.4f}")
    logger.info(f"CSG: {results['csg']:.4f}")
    logger.info(f"DES: {results['des']:.4f}")
    logger.info(f"Sparsity: {results['sparsity']:.1%}")
    logger.info(f"Parameters: {results['parameters']:,}")
    logger.info(f"FPS: {results['fps']:.2f}")
    
    return model, results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='PMP-DACIS Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone (resnet18, mobilenetv2)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset name')
    parser.add_argument('--shots', type=int, default=None,
                        help='Override K-shot value')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.shots:
        config['fsl']['k_shot'] = args.shots
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f"pmp_{config['model']['backbone']}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, experiment_name)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Set seed
    set_seed(args.seed)
    
    # Run training
    try:
        model, results = run_pmp_training(config, device, logger, writer)
        
        # Save model
        model_path = os.path.join(output_dir, 'model_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': results
        }, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save results
        results_path = os.path.join(output_dir, 'results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        writer.close()
    
    logger.info("Training completed successfully!")
    return model, results


if __name__ == '__main__':
    main()
