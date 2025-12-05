"""
PMP-DACIS Evaluation Script

Evaluate trained models on few-shot plant disease classification.

Usage:
    python evaluate.py --checkpoint outputs/experiment/model_final.pth
    python evaluate.py --checkpoint model.pth --dataset plantvillage --shots 5
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_backbone, PMPFramework, create_pmp_model
from utils import (
    DeploymentEfficiencyScore,
    FewShotStabilityIndex,
    CompressionStabilityGain,
    compute_sparsity,
    compute_flops,
    count_nonzero_params
)


def setup_logging(output_dir: str = None) -> logging.Logger:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'evaluate.log')
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy dataset for testing when real data is not available."""
    
    def __init__(
        self,
        num_classes: int = 38,
        samples_per_class: int = 50,
        image_size: int = 224,
        seed: int = 42
    ):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.total_samples = num_classes * samples_per_class
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Pre-generate class assignments
        self.labels = []
        for cls in range(num_classes):
            self.labels.extend([cls] * samples_per_class)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        label = self.labels[idx]
        return image, label


class EpisodeSampler:
    """Few-shot episode sampler for evaluation."""
    
    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 1000,
        seed: int = 42
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.seed = seed
        
        # Build class-to-indices mapping
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        
        # Set seed
        np.random.seed(seed)
    
    def __len__(self):
        return self.num_episodes
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self._sample_episode()
    
    def _sample_episode(self) -> Tuple[torch.Tensor, ...]:
        """Sample a single episode."""
        episode_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for new_label, cls in enumerate(episode_classes):
            available = self.class_indices[cls]
            indices = np.random.choice(
                available,
                min(self.k_shot + self.q_query, len(available)),
                replace=False
            )
            
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
        
        return support_x, support_y, query_x, query_y


def load_model(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model
    model = create_pmp_model(
        backbone_name=model_config.get('backbone', 'resnet18'),
        feature_dim=model_config.get('feature_dim', 512),
        num_classes=config.get('dataset', {}).get('num_classes', 38),
        n_way=config.get('fsl', {}).get('n_way', 5)
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_fewshot(
    model: nn.Module,
    test_loader,
    device: str,
    num_episodes: int = 1000,
    logger: logging.Logger = None
) -> Dict[str, any]:
    """
    Evaluate model on few-shot classification.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        num_episodes: Number of episodes to evaluate
        logger: Logger instance
        
    Returns:
        Evaluation results
    """
    model.eval()
    
    episode_accuracies = []
    episode_times = []
    
    with torch.no_grad():
        for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(test_loader):
            if episode_idx >= num_episodes:
                break
            
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
            
            # Time the inference
            start_time = time.time()
            
            if hasattr(model, 'forward_episode'):
                logits = model.forward_episode(support_x, support_y, query_x)
            else:
                logits = model(query_x)
            
            episode_time = time.time() - start_time
            
            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == query_y).float().mean().item()
            
            episode_accuracies.append(accuracy)
            episode_times.append(episode_time)
            
            # Log progress
            if logger and (episode_idx + 1) % 200 == 0:
                current_mean = np.mean(episode_accuracies)
                current_std = np.std(episode_accuracies)
                logger.info(
                    f"Episode {episode_idx + 1}/{num_episodes} | "
                    f"Acc: {current_mean:.4f} ± {current_std:.4f}"
                )
    
    # Compute statistics
    accuracies = np.array(episode_accuracies)
    times = np.array(episode_times)
    
    # Confidence interval (95%)
    n = len(accuracies)
    ci_95 = 1.96 * np.std(accuracies) / np.sqrt(n)
    
    # FSI
    fsi_calc = FewShotStabilityIndex()
    fsi_calc.add_batch_accuracies(episode_accuracies)
    fsi_results = fsi_calc.compute()
    
    results = {
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'ci_95': float(ci_95),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
            'median': float(np.median(accuracies))
        },
        'fsi': float(fsi_results['fsi']),
        'timing': {
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'episodes_per_second': float(1.0 / np.mean(times))
        },
        'num_episodes': n,
        'episode_accuracies': accuracies.tolist()
    }
    
    return results


def evaluate_efficiency(
    model: nn.Module,
    device: str,
    hardware_profile: str = 'raspberry_pi_4',
    accuracy: float = 0.0
) -> Dict[str, any]:
    """
    Evaluate model efficiency metrics.
    
    Args:
        model: Model to evaluate
        device: Device to run on
        hardware_profile: Target hardware
        accuracy: Model accuracy for DES computation
        
    Returns:
        Efficiency metrics
    """
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero_params = count_nonzero_params(model)
    
    # Compute sparsity
    sparsity_stats = compute_sparsity(model)
    
    # Compute FLOPs
    flops_stats = compute_flops(model, input_size=(1, 3, 224, 224))
    
    # Compute DES
    des_calc = DeploymentEfficiencyScore(hardware_profile=hardware_profile)
    des_results = des_calc.compute(model, accuracy, device=device)
    
    results = {
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'nonzero': nonzero_params,
            'compression_ratio': total_params / nonzero_params if nonzero_params > 0 else 1.0
        },
        'sparsity': {
            'overall': sparsity_stats['overall_sparsity'],
            'density': sparsity_stats['density']
        },
        'flops': {
            'total_gflops': flops_stats['gflops'],
            'nonzero_gflops': flops_stats['nonzero_flops'] / 1e9,
            'theoretical_speedup': flops_stats['theoretical_speedup']
        },
        'des': {
            'score': des_results['des'],
            'fps': des_results['fps'],
            'energy_j': des_results['energy_j'],
            'hardware': hardware_profile
        }
    }
    
    return results


def evaluate_per_shot(
    model: nn.Module,
    test_dataset: Dataset,
    device: str,
    n_way: int = 5,
    shots: List[int] = [1, 5, 10],
    episodes_per_shot: int = 500,
    logger: logging.Logger = None
) -> Dict[str, Dict]:
    """
    Evaluate model across different shot settings.
    
    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        device: Device to run on
        n_way: Number of ways
        shots: List of shot values to evaluate
        episodes_per_shot: Episodes per shot setting
        logger: Logger instance
        
    Returns:
        Results per shot setting
    """
    results = {}
    
    for k_shot in shots:
        if logger:
            logger.info(f"\nEvaluating {n_way}-way {k_shot}-shot...")
        
        sampler = EpisodeSampler(
            test_dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=15,
            num_episodes=episodes_per_shot
        )
        
        loader = list(sampler)
        
        shot_results = evaluate_fewshot(
            model, loader, device,
            num_episodes=episodes_per_shot,
            logger=logger
        )
        
        results[f'{k_shot}-shot'] = {
            'accuracy': shot_results['accuracy']['mean'],
            'std': shot_results['accuracy']['std'],
            'ci_95': shot_results['accuracy']['ci_95'],
            'fsi': shot_results['fsi']
        }
        
        if logger:
            logger.info(
                f"{k_shot}-shot: {shot_results['accuracy']['mean']:.4f} "
                f"± {shot_results['accuracy']['ci_95']:.4f} | "
                f"FSI: {shot_results['fsi']:.4f}"
            )
    
    return results


def generate_report(
    model: nn.Module,
    fewshot_results: Dict,
    efficiency_results: Dict,
    per_shot_results: Dict = None,
    output_path: str = None
) -> str:
    """
    Generate evaluation report.
    
    Args:
        model: Evaluated model
        fewshot_results: Few-shot evaluation results
        efficiency_results: Efficiency metrics
        per_shot_results: Per-shot evaluation results
        output_path: Path to save report
        
    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PMP-DACIS EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Few-shot Performance
    lines.append("-" * 70)
    lines.append("FEW-SHOT CLASSIFICATION PERFORMANCE")
    lines.append("-" * 70)
    acc = fewshot_results['accuracy']
    lines.append(f"Accuracy:     {acc['mean']:.4f} ± {acc['std']:.4f}")
    lines.append(f"95% CI:       [{acc['mean']-acc['ci_95']:.4f}, {acc['mean']+acc['ci_95']:.4f}]")
    lines.append(f"FSI:          {fewshot_results['fsi']:.4f}")
    lines.append(f"Episodes:     {fewshot_results['num_episodes']}")
    lines.append("")
    
    # Per-shot results
    if per_shot_results:
        lines.append("-" * 70)
        lines.append("PER-SHOT PERFORMANCE")
        lines.append("-" * 70)
        for shot, results in per_shot_results.items():
            lines.append(f"{shot}: {results['accuracy']:.4f} ± {results['ci_95']:.4f} (FSI: {results['fsi']:.4f})")
        lines.append("")
    
    # Efficiency Metrics
    lines.append("-" * 70)
    lines.append("EFFICIENCY METRICS")
    lines.append("-" * 70)
    params = efficiency_results['parameters']
    lines.append(f"Total Parameters:    {params['total']:,}")
    lines.append(f"Non-zero Parameters: {params['nonzero']:,}")
    lines.append(f"Compression Ratio:   {params['compression_ratio']:.2f}x")
    lines.append("")
    
    sparsity = efficiency_results['sparsity']
    lines.append(f"Overall Sparsity:    {sparsity['overall']:.1%}")
    lines.append(f"Model Density:       {sparsity['density']:.1%}")
    lines.append("")
    
    flops = efficiency_results['flops']
    lines.append(f"Total GFLOPs:        {flops['total_gflops']:.2f}")
    lines.append(f"Effective GFLOPs:    {flops['nonzero_gflops']:.2f}")
    lines.append(f"Theoretical Speedup: {flops['theoretical_speedup']:.2f}x")
    lines.append("")
    
    des = efficiency_results['des']
    lines.append(f"DES Score:           {des['score']:.4f}")
    lines.append(f"FPS:                 {des['fps']:.2f}")
    lines.append(f"Energy per Inf:      {des['energy_j']:.4f} J")
    lines.append(f"Target Hardware:     {des['hardware']}")
    lines.append("")
    
    # Timing
    lines.append("-" * 70)
    lines.append("TIMING")
    lines.append("-" * 70)
    timing = fewshot_results['timing']
    lines.append(f"Mean Episode Time:   {timing['mean_ms']:.2f} ms")
    lines.append(f"Episodes/Second:     {timing['episodes_per_second']:.2f}")
    lines.append("")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='PMP-DACIS Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (uses dummy if not specified)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset')
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 5, 10],
                        help='Shot values to evaluate')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for main evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu)')
    parser.add_argument('--hardware', type=str, default='raspberry_pi_4',
                        help='Target hardware for DES')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Setup
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Loading model from: {args.checkpoint}")
    logger.info(f"Device: {device}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    logger.info(f"Model loaded successfully")
    
    # Create test dataset
    logger.info("Creating test dataset...")
    num_classes = config.get('dataset', {}).get('num_classes', 38)
    test_dataset = DummyDataset(
        num_classes=num_classes,
        samples_per_class=50,
        seed=args.seed
    )
    
    # Main evaluation
    logger.info(f"\nRunning main evaluation ({args.episodes} episodes)...")
    main_sampler = EpisodeSampler(
        test_dataset,
        n_way=args.n_way,
        k_shot=args.shots[1] if len(args.shots) > 1 else args.shots[0],
        q_query=15,
        num_episodes=args.episodes,
        seed=args.seed
    )
    main_loader = list(main_sampler)
    
    fewshot_results = evaluate_fewshot(
        model, main_loader, device,
        num_episodes=args.episodes,
        logger=logger
    )
    
    logger.info(f"\nMain Results:")
    logger.info(f"Accuracy: {fewshot_results['accuracy']['mean']:.4f} ± {fewshot_results['accuracy']['ci_95']:.4f}")
    logger.info(f"FSI: {fewshot_results['fsi']:.4f}")
    
    # Per-shot evaluation
    logger.info(f"\nRunning per-shot evaluation...")
    per_shot_results = evaluate_per_shot(
        model, test_dataset, device,
        n_way=args.n_way,
        shots=args.shots,
        episodes_per_shot=min(500, args.episodes // 2),
        logger=logger
    )
    
    # Efficiency evaluation
    logger.info(f"\nEvaluating efficiency metrics...")
    efficiency_results = evaluate_efficiency(
        model, device,
        hardware_profile=args.hardware,
        accuracy=fewshot_results['accuracy']['mean']
    )
    
    logger.info(f"Parameters: {efficiency_results['parameters']['total']:,}")
    logger.info(f"Sparsity: {efficiency_results['sparsity']['overall']:.1%}")
    logger.info(f"DES: {efficiency_results['des']['score']:.4f}")
    
    # Generate report
    report = generate_report(
        model,
        fewshot_results,
        efficiency_results,
        per_shot_results,
        output_path=os.path.join(args.output_dir, 'report.txt')
    )
    print("\n" + report)
    
    # Save detailed results
    all_results = {
        'fewshot': {k: v for k, v in fewshot_results.items() if k != 'episode_accuracies'},
        'efficiency': efficiency_results,
        'per_shot': per_shot_results,
        'config': {
            'checkpoint': args.checkpoint,
            'n_way': args.n_way,
            'shots': args.shots,
            'episodes': args.episodes,
            'hardware': args.hardware,
            'seed': args.seed
        }
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save episode accuracies separately
    accuracies_path = os.path.join(args.output_dir, 'episode_accuracies.npy')
    np.save(accuracies_path, np.array(fewshot_results['episode_accuracies']))
    logger.info(f"Episode accuracies saved to: {accuracies_path}")
    
    logger.info("\nEvaluation completed!")
    
    return all_results


if __name__ == '__main__':
    main()
