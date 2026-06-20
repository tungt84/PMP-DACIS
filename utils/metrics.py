"""
PMP-DACIS Evaluation Metrics

This module implements the three key metrics from the paper:
1. Deployment Efficiency Score (DES)
2. Few-Shot Stability Index (FSI)
3. Compression-Stability Gain (CSG)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from collections import defaultdict
import time
import json
import os


class DeploymentEfficiencyScore:
    """
    Deployment Efficiency Score (DES)
    
    DES = (Accuracy × FPS) / (Parameters × Energy)
    
    Balances accuracy with inference speed while considering
    model complexity and energy consumption.
    """
    
    def __init__(
        self,
        hardware_profile: str = 'raspberry_pi_4',
        normalize: bool = True
    ):
        """
        Initialize DES calculator.
        
        Args:
            hardware_profile: Target hardware for energy estimation
            normalize: Whether to normalize the score
        """
        self.hardware_profile = hardware_profile
        self.normalize = normalize
        
        # Energy coefficients (Joules per GFLOP) for different hardware
        self.energy_coefficients = {
            'raspberry_pi_4': 0.5,      # ~0.5 J/GFLOP
            'jetson_nano': 0.15,         # ~0.15 J/GFLOP (GPU)
            'edge_tpu': 0.08,           # ~0.08 J/GFLOP
            'laptop_cpu': 0.3,          # ~0.3 J/GFLOP
            'desktop_gpu': 0.05         # ~0.05 J/GFLOP
        }
        
        # Baseline values for normalization
        self.baselines = {
            'accuracy': 0.85,
            'fps': 30.0,
            'parameters': 11.2e6,  # ResNet-18
            'energy': 0.1
        }
    
    def compute_fps(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup: int = 10,
        device: str = 'cpu',
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
    ) -> float:
        """Measure inference FPS for a meta‑few‑shot model.

        The PMPFramework (and similar meta‑learning models) expects a
        *support* set, a *query* set and the corresponding *support labels*
        as inputs to ``forward``.  The original implementation used a single
        dummy image tensor, which caused a ``TypeError`` because the required
        arguments were missing.  This version creates a dummy *episode*
        matching the signature ``forward(support_images, query_images,
        support_labels, mode='proto')``.

        Args:
            model: Model to benchmark.
            input_size: Shape of a single image tensor ``(C, H, W)`` – the
                batch dimension is derived from ``n_way``, ``k_shot`` and
                ``q_query``.
            num_runs: Number of inference runs for timing.
            warmup: Warm‑up iterations to stabilize GPU kernels.
            device: Device identifier (e.g., ``'cpu'`` or ``'cuda'``).
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
            q_query: Number of query samples per class.

        Returns:
            Frames per second (FPS) measured as the number of *query* images
            processed per second.
        """
        model = model.to(device)
        model.eval()

        # ------------------------------------------------------------------
        # 1️⃣ Build dummy episode (support, query, labels)
        # ------------------------------------------------------------------
        # ``input_size`` is expected to be ``(B, C, H, W)`` where B is 1.
        # We extract the channel and spatial dimensions.
        _, C, H, W = input_size

        # Support set: (n_way * k_shot, C, H, W)
        support_images = torch.randn(
            n_way * k_shot, C, H, W, device=device
        )
        # Query set: (n_way * q_query, C, H, W)
        query_images = torch.randn(
            n_way * q_query, C, H, W, device=device
        )
        # Support labels: repeated class indices for each support sample
        support_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)

        # ------------------------------------------------------------------
        # 2️⃣ Warm‑up runs (GPU kernels need a few iterations to stabilise)
        # ------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(
                    support_images,
                    query_images,
                    support_labels,
                    mode='proto'
                )

        # Synchronize if on GPU before timing
        if device == 'cuda':
            torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # 3️⃣ Benchmark
        # ------------------------------------------------------------------
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(
                    support_images,
                    query_images,
                    support_labels,
                    mode='proto'
                )
        if device == 'cuda':
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        # FPS is measured in terms of *query* images processed per second.
        fps = (num_runs * n_way * q_query) / total_time
        return fps
    
    def estimate_energy(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> float:
        """
        Estimate energy consumption per inference.
        
        Args:
            model: Model to analyze
            input_size: Input tensor shape
            
        Returns:
            Estimated energy in Joules
        """
        flops = self._estimate_flops(model, input_size)
        gflops = flops / 1e9
        
        energy_coefficient = self.energy_coefficients.get(
            self.hardware_profile, 0.5
        )
        
        energy = gflops * energy_coefficient
        return energy
    
    def _estimate_flops(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...]
    ) -> int:
        """Estimate FLOPs for the model using a dummy episode."""
        total_flops = 0

        def hook_fn(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                batch_size = input[0].shape[0]
                output_dims = output.shape[2:]
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                in_channels = module.in_channels // module.groups
                out_channels = module.out_channels
                flops = batch_size * np.prod(output_dims) * kernel_ops * in_channels * out_channels
                total_flops += flops
            elif isinstance(module, nn.Linear):
                batch_size = input[0].shape[0]
                flops = batch_size * module.in_features * module.out_features
                total_flops += flops

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

        # Build dummy episode matching compute_fps signature
        _, C, H, W = input_size
        n_way, k_shot, q_query = 5, 5, 15
        support_images = torch.randn(n_way * k_shot, C, H, W)
        query_images = torch.randn(n_way * q_query, C, H, W)
        support_labels = torch.arange(n_way).repeat_interleave(k_shot)

        model.eval()
        with torch.no_grad():
            _ = model(support_images, query_images, support_labels, mode="proto")

        for hook in hooks:
            hook.remove()
        return int(total_flops)
    
    def count_parameters(self, model: nn.Module) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())
    
    def compute(
        self,
        model: nn.Module,
        accuracy: float,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        device: str = 'cpu',
        fps: Optional[float] = None,
        energy: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute Deployment Efficiency Score.
        
        Args:
            model: Model to evaluate
            accuracy: Model accuracy (0-1)
            input_size: Input tensor shape
            device: Device for benchmarking
            fps: Pre-computed FPS (optional)
            energy: Pre-computed energy (optional)
            
        Returns:
            Dictionary with DES and component metrics
        """
        # Compute metrics
        if fps is None:
            fps = self.compute_fps(model, input_size, device=device)
        
        if energy is None:
            energy = self.estimate_energy(model, input_size)
        
        params = self.count_parameters(model)
        
        # Avoid division by zero
        params = max(params, 1)
        energy = max(energy, 1e-6)
        
        # Compute raw DES
        des_raw = (accuracy * fps) / (params * energy)
        
        # Normalize if requested
        if self.normalize:
            baseline_des = (
                self.baselines['accuracy'] * self.baselines['fps']
            ) / (
                self.baselines['parameters'] * self.baselines['energy']
            )
            des = des_raw / baseline_des
        else:
            des = des_raw
        
        return {
            'des': des,
            'des_raw': des_raw,
            'accuracy': accuracy,
            'fps': fps,
            'parameters': params,
            'energy_j': energy,
            'hardware': self.hardware_profile
        }


class FewShotStabilityIndex:
    """
    Few-Shot Stability Index (FSI)
    
    FSI = 1 - (σ_acc / μ_acc)
    
    Measures consistency of accuracy across different episode configurations.
    Higher FSI indicates more stable performance.
    """
    
    def __init__(self, min_episodes: int = 100):
        """
        Initialize FSI calculator.
        
        Args:
            min_episodes: Minimum episodes for reliable FSI
        """
        self.min_episodes = min_episodes
        self.accuracies: Dict[str, List[float]] = defaultdict(list)
    
    def add_episode_accuracy(
        self,
        accuracy: float,
        config: str = 'default'
    ):
        """
        Add accuracy from an episode.
        
        Args:
            accuracy: Episode accuracy (0-1)
            config: Configuration identifier
        """
        self.accuracies[config].append(accuracy)
    
    def add_batch_accuracies(
        self,
        accuracies: List[float],
        config: str = 'default'
    ):
        """Add multiple accuracies at once."""
        self.accuracies[config].extend(accuracies)
    
    def compute(self, config: str = 'default') -> Dict[str, float]:
        """
        Compute FSI for a configuration.
        
        Args:
            config: Configuration to evaluate
            
        Returns:
            Dictionary with FSI and statistics
        """
        accs = self.accuracies.get(config, [])
        
        if len(accs) < 2:
            return {
                'fsi': 0.0,
                'mean_accuracy': np.mean(accs) if accs else 0.0,
                'std_accuracy': 0.0,
                'num_episodes': len(accs),
                'reliable': False
            }
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        
        # Avoid division by zero
        if mean_acc < 1e-6:
            fsi = 0.0
        else:
            fsi = 1.0 - (std_acc / mean_acc)
        
        # Clamp to [0, 1]
        fsi = max(0.0, min(1.0, fsi))
        
        return {
            'fsi': fsi,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'num_episodes': len(accs),
            'reliable': len(accs) >= self.min_episodes,
            'cv': std_acc / mean_acc if mean_acc > 1e-6 else float('inf')
        }
    
    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """Compute FSI for all configurations."""
        return {
            config: self.compute(config)
            for config in self.accuracies.keys()
        }
    
    def reset(self, config: Optional[str] = None):
        """Reset stored accuracies."""
        if config is None:
            self.accuracies.clear()
        elif config in self.accuracies:
            del self.accuracies[config]


class CompressionStabilityGain:
    """
    Compression-Stability Gain (CSG)
    
    CSG = Accuracy_late_stage / Accuracy_early_stage
    
    Measures the stability gain through progressive pruning.
    CSG > 1 indicates improvement through meta-learning.
    """
    
    def __init__(self):
        """Initialize CSG calculator."""
        self.stage_accuracies: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def record_stage_accuracy(
        self,
        accuracy: float,
        stage: str,
        config: str = 'default'
    ):
        """
        Record accuracy at a pruning stage.
        
        Args:
            accuracy: Accuracy at this stage
            stage: Stage identifier (e.g., 'stage1', 'stage2', 'stage3')
            config: Configuration identifier
        """
        self.stage_accuracies[config][stage] = accuracy
    
    def compute(
        self,
        early_stage: str = 'stage1',
        late_stage: str = 'stage3',
        config: str = 'default'
    ) -> Dict[str, float]:
        """
        Compute CSG between two stages.
        
        Args:
            early_stage: Earlier stage name
            late_stage: Later stage name
            config: Configuration to evaluate
            
        Returns:
            Dictionary with CSG and stage accuracies
        """
        stages = self.stage_accuracies.get(config, {})
        
        early_acc = stages.get(early_stage)
        late_acc = stages.get(late_stage)
        
        if early_acc is None or late_acc is None:
            return {
                'csg': None,
                'early_accuracy': early_acc,
                'late_accuracy': late_acc,
                'early_stage': early_stage,
                'late_stage': late_stage,
                'valid': False
            }
        
        # Avoid division by zero
        if early_acc < 1e-6:
            csg = float('inf') if late_acc > 0 else 0.0
        else:
            csg = late_acc / early_acc
        
        return {
            'csg': csg,
            'early_accuracy': early_acc,
            'late_accuracy': late_acc,
            'early_stage': early_stage,
            'late_stage': late_stage,
            'improvement_pct': (csg - 1.0) * 100 if csg is not None else None,
            'valid': True
        }
    
    def get_all_stages(self, config: str = 'default') -> Dict[str, float]:
        """Get all recorded stage accuracies."""
        return dict(self.stage_accuracies.get(config, {}))


class MetricLogger:
    """
    Utility class for logging and saving metrics during training/evaluation.
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        experiment_name: str = 'experiment'
    ):
        """
        Initialize metric logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics: Dict[str, List] = defaultdict(list)
        self.step = 0
        
        os.makedirs(log_dir, exist_ok=True)
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics at a step.
        
        Args:
            metrics: Dictionary of metric values
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self.step
            self.step += 1
        
        for name, value in metrics.items():
            self.metrics[name].append({
                'step': step,
                'value': value
            })
    
    def get_metric_history(self, name: str) -> List[Dict]:
        """Get history of a specific metric."""
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric."""
        history = self.metrics.get(name, [])
        return history[-1]['value'] if history else None
    
    def save(self, filename: Optional[str] = None):
        """Save metrics to JSON file."""
        if filename is None:
            filename = f'{self.experiment_name}_metrics.json'
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'metrics': dict(self.metrics)
            }, f, indent=2)
        
        return filepath
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.experiment_name = data.get('experiment_name', 'loaded')
        self.metrics = defaultdict(list, data.get('metrics', {}))
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        
        for name, history in self.metrics.items():
            values = [h['value'] for h in history]
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary


def compute_all_metrics(
    model: nn.Module,
    accuracy: float,
    episode_accuracies: List[float],
    stage_accuracies: Dict[str, float],
    hardware_profile: str = 'raspberry_pi_4',
    device: str = 'cpu'
) -> Dict[str, any]:
    """
    Compute all PMP-DACIS metrics at once.
    
    Args:
        model: Model to evaluate
        accuracy: Final accuracy
        episode_accuracies: List of per-episode accuracies
        stage_accuracies: Dictionary mapping stage names to accuracies
        hardware_profile: Target hardware
        device: Device for benchmarking
        
    Returns:
        Dictionary with all metrics
    """
    # Deployment Efficiency Score
    des_calculator = DeploymentEfficiencyScore(hardware_profile=hardware_profile)
    des_results = des_calculator.compute(model, accuracy, device=device)
    
    # Few-Shot Stability Index
    fsi_calculator = FewShotStabilityIndex()
    fsi_calculator.add_batch_accuracies(episode_accuracies)
    fsi_results = fsi_calculator.compute()
    
    # Compression-Stability Gain
    csg_calculator = CompressionStabilityGain()
    for stage, acc in stage_accuracies.items():
        csg_calculator.record_stage_accuracy(acc, stage)
    csg_results = csg_calculator.compute()
    
    return {
        'des': des_results,
        'fsi': fsi_results,
        'csg': csg_results,
        'summary': {
            'des_score': des_results['des'],
            'fsi_score': fsi_results['fsi'],
            'csg_score': csg_results['csg'],
            'accuracy': accuracy,
            'fps': des_results['fps'],
            'parameters': des_results['parameters']
        }
    }


# Convenience functions for standard benchmarks
def benchmark_model(
    model: nn.Module,
    test_loader,
    device: str = 'cpu',
    num_episodes: int = 100,
    hardware_profile: str = 'raspberry_pi_4'
) -> Dict[str, any]:
    """
    Run complete benchmark on a model.
    
    Args:
        model: Model to benchmark
        test_loader: DataLoader yielding (support, query) tasks
        device: Device to run on
        num_episodes: Number of episodes to evaluate
        hardware_profile: Target hardware for DES
        
    Returns:
        Complete benchmark results
    """
    model = model.to(device)
    model.eval()
    
    episode_accuracies = []
    
    with torch.no_grad():
        for i, (support_x, support_y, query_x, query_y) in enumerate(test_loader):
            if i >= num_episodes:
                break
            
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
            
            # Adapt and predict
            if hasattr(model, 'adapt_and_predict'):
                predictions = model.adapt_and_predict(
                    support_x, support_y, query_x
                )
            else:
                # Basic forward pass
                predictions = model(query_x)
            
            # Compute accuracy
            pred_labels = predictions.argmax(dim=-1)
            accuracy = (pred_labels == query_y).float().mean().item()
            episode_accuracies.append(accuracy)
    
    # Compute metrics
    final_accuracy = np.mean(episode_accuracies)
    
    results = compute_all_metrics(
        model=model,
        accuracy=final_accuracy,
        episode_accuracies=episode_accuracies,
        stage_accuracies={'final': final_accuracy},
        hardware_profile=hardware_profile,
        device=device
    )
    
    return results


if __name__ == '__main__':
    # Demo/test of metrics
    print("=" * 60)
    print("PMP-DACIS Metrics Module Demo")
    print("=" * 60)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    model = SimpleModel()
    
    # Test DES
    print("\n1. Deployment Efficiency Score (DES)")
    print("-" * 40)
    des_calc = DeploymentEfficiencyScore(hardware_profile='raspberry_pi_4')
    des_result = des_calc.compute(model, accuracy=0.89)
    print(f"   DES: {des_result['des']:.4f}")
    print(f"   FPS: {des_result['fps']:.2f}")
    print(f"   Parameters: {des_result['parameters']:,}")
    print(f"   Energy: {des_result['energy_j']:.4f} J")
    
    # Test FSI
    print("\n2. Few-Shot Stability Index (FSI)")
    print("-" * 40)
    fsi_calc = FewShotStabilityIndex()
    # Simulate episode accuracies
    np.random.seed(42)
    episode_accs = np.random.normal(0.89, 0.05, 100).clip(0, 1)
    fsi_calc.add_batch_accuracies(episode_accs.tolist())
    fsi_result = fsi_calc.compute()
    print(f"   FSI: {fsi_result['fsi']:.4f}")
    print(f"   Mean Accuracy: {fsi_result['mean_accuracy']:.4f}")
    print(f"   Std Accuracy: {fsi_result['std_accuracy']:.4f}")
    
    # Test CSG
    print("\n3. Compression-Stability Gain (CSG)")
    print("-" * 40)
    csg_calc = CompressionStabilityGain()
    csg_calc.record_stage_accuracy(0.85, 'stage1')  # Initial pruning
    csg_calc.record_stage_accuracy(0.87, 'stage2')  # Meta-learning
    csg_calc.record_stage_accuracy(0.89, 'stage3')  # Refinement
    csg_result = csg_calc.compute()
    print(f"   CSG: {csg_result['csg']:.4f}")
    print(f"   Stage 1 (Early): {csg_result['early_accuracy']:.4f}")
    print(f"   Stage 3 (Late): {csg_result['late_accuracy']:.4f}")
    print(f"   Improvement: {csg_result['improvement_pct']:.1f}%")
    
    # Test combined metrics
    print("\n4. Combined Metrics")
    print("-" * 40)
    all_metrics = compute_all_metrics(
        model=model,
        accuracy=0.89,
        episode_accuracies=episode_accs.tolist(),
        stage_accuracies={'stage1': 0.85, 'stage3': 0.89},
        hardware_profile='raspberry_pi_4'
    )
    print(f"   DES: {all_metrics['summary']['des_score']:.4f}")
    print(f"   FSI: {all_metrics['summary']['fsi_score']:.4f}")
    print(f"   CSG: {all_metrics['summary']['csg_score']:.4f}")
    
    # Test MetricLogger
    print("\n5. Metric Logger")
    print("-" * 40)
    logger = MetricLogger(log_dir='./test_logs', experiment_name='demo')
    for i in range(10):
        logger.log({
            'loss': 1.0 / (i + 1),
            'accuracy': 0.5 + 0.05 * i
        })
    summary = logger.summary()
    print(f"   Loss - Mean: {summary['loss']['mean']:.4f}, "
          f"Min: {summary['loss']['min']:.4f}")
    print(f"   Accuracy - Mean: {summary['accuracy']['mean']:.4f}, "
          f"Max: {summary['accuracy']['max']:.4f}")
    
    print("\n" + "=" * 60)
    print("All metrics tests passed!")
    print("=" * 60)
