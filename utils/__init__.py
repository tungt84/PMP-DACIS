"""
Utilities package for PMP-DACIS
"""

from .metrics import (
    DeploymentEfficiencyScore,
    FewShotStabilityIndex,
    CompressionStabilityGain,
    compute_all_metrics,
    MetricLogger
)

from .pruning import (
    ChannelPruner,
    apply_channel_pruning,
    compute_sparsity,
    count_nonzero_params,
    compute_flops
)

__all__ = [
    # Metrics
    'DeploymentEfficiencyScore',
    'FewShotStabilityIndex',
    'CompressionStabilityGain',
    'compute_all_metrics',
    'MetricLogger',
    
    # Pruning
    'ChannelPruner',
    'apply_channel_pruning',
    'compute_sparsity',
    'count_nonzero_params',
    'compute_flops',
]
