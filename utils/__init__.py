"""
LAIGCD Utils
"""

from .data import (
    AIGCDataset,
    SimpleDataset,
    get_train_dataloader,
    get_val_dataloader,
    get_train_transforms,
    get_val_transforms
)

from .train import (
    MetricLogger,
    train_one_epoch,
    validate,
    print_metrics,
    save_checkpoint,
    load_checkpoint
)

from .metrics import (
    compute_metrics,
    compute_per_generator_metrics,
    print_metrics as print_metrics_dict,
    print_per_generator_metrics,
    format_metrics_table,
    get_optimal_threshold
)

from .viz import (
    plot_confusion_matrix,
    plot_roc_curve,
    visualize_prototype_attention,
    plot_training_curves,
    plot_prototypes_embeddings
)

__all__ = [
    # Data
    'AIGCDataset',
    'SimpleDataset',
    'get_train_dataloader',
    'get_val_dataloader',
    'get_train_transforms',
    'get_val_transforms',

    # Train
    'MetricLogger',
    'train_one_epoch',
    'validate',
    'print_metrics',
    'save_checkpoint',
    'load_checkpoint',

    # Metrics
    'compute_metrics',
    'compute_per_generator_metrics',
    'print_metrics_dict',
    'print_per_generator_metrics',
    'format_metrics_table',
    'get_optimal_threshold',

    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'visualize_prototype_attention',
    'plot_training_curves',
    'plot_prototypes_embeddings',
]
