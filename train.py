"""
Unified training script for PES multi-task classification.

Usage:
    python train.py --config config1
    python train.py --config config5 --epochs 50  # override config value
"""

import os
import sys
import json
import argparse
import importlib
import importlib.util
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model_builder import build_model
from dataset_builder import create_dataloaders
from Task_Setting.task_catalog import get_display_names
from engine.trainer import train_epoch, validate_epoch
from visualization.plots import plot_training_curves, plot_confusion_matrices, plot_auc_curves


def load_config(config_name):
    """Load a config module by name (e.g. 'config1' or 'trainingconfig1')."""
    def _load_from_file(file_path, module_tag):
        spec = importlib.util.spec_from_file_location(module_tag, file_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Cannot load config from file: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    config_root = os.path.join(PROJECT_ROOT, 'config')

    # Dotted names (e.g. trainingconfigs.config1) are treated as package modules.
    if '.' in config_name:
        return importlib.import_module(f'config.{config_name}')

    # Prefer direct file loading for top-level config names to avoid package/file collisions
    # such as config/config2.py and config/config2/.
    direct_file = os.path.join(config_root, f'{config_name}.py')
    if os.path.isfile(direct_file):
        return _load_from_file(direct_file, f'config_file_{config_name}')

    training_file = os.path.join(config_root, 'trainingconfigs', f'{config_name}.py')
    if os.path.isfile(training_file):
        return _load_from_file(training_file, f'training_config_file_{config_name}')

    # Fallback to import-based loading.
    try:
        return importlib.import_module(f'config.{config_name}')
    except ModuleNotFoundError as e:
        try:
            return importlib.import_module(f'config.trainingconfigs.{config_name}')
        except ModuleNotFoundError:
            raise e


def build_criteria(cfg, class_weights, device):
    """Build loss functions for each task based on config."""
    loss_type = getattr(cfg, 'LOSS_TYPE', 'weighted_ce')
    criteria = {}

    if loss_type == 'cb_focal':
        from Loss_function.cb_focal import CBFocalLoss
        # Need per-task class counts from split info
        for key in cfg.TASKS:
            w = class_weights[key]
            # Convert weights back to approximate counts for CB-Focal
            # Use weights as proxy (higher weight = fewer samples)
            inv_w = 1.0 / w.clamp(min=1e-6)
            counts = (inv_w / inv_w.sum() * 100).numpy()  # approximate
            beta = getattr(cfg, 'CB_FOCAL_BETA', 0.999)
            gamma = getattr(cfg, 'CB_FOCAL_GAMMA', 1.5)
            criteria[key] = CBFocalLoss(counts, beta=beta, gamma=gamma).to(device)
    else:
        # Weighted cross-entropy (default)
        from Loss_function.weighted_ce import create_weighted_ce
        for key in cfg.TASKS:
            criteria[key] = create_weighted_ce(class_weights[key], device)

    return criteria


def main():
    parser = argparse.ArgumentParser(description='PES Multi-Task Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Config name (e.g. config1 or trainingconfig1)')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Override device')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    config_name = args.config

    # Apply overrides
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.device is not None:
        cfg.DEVICE = args.device
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size

    device = getattr(cfg, 'DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Config: {config_name}")
    print(f"Device: {device}")
    print(f"Tasks: {cfg.TASKS}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', config_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config snapshot
    config_snapshot = {k: v for k, v in vars(cfg).items()
                       if not k.startswith('_') and not callable(v)}
    # Convert non-serializable values
    for k, v in config_snapshot.items():
        if isinstance(v, type):
            config_snapshot[k] = str(v)
    with open(os.path.join(output_dir, 'config_snapshot.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Build model
    model, preprocess = build_model(cfg, device=device)

    # Create dataloaders
    train_loader, val_loader, class_weights, split_info = create_dataloaders(cfg, preprocess)

    # Save split info
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2, default=str)

    # Build loss functions
    criteria = build_criteria(cfg, class_weights, device)

    # Optimizer and scheduler
    lr = getattr(cfg, 'LEARNING_RATE', 1e-4)
    weight_decay = getattr(cfg, 'WEIGHT_DECAY', 0.01)
    optimizer = AdamW(model.get_trainable_params(), lr=lr, weight_decay=weight_decay)

    num_epochs = getattr(cfg, 'NUM_EPOCHS', 30)
    eta_min_ratio = getattr(cfg, 'ETA_MIN_RATIO', 0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * eta_min_ratio)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_f1': [], 'val_auc': [],
        # Best checkpoint is selected by avg_auc.
        'selection_metric': 'avg_auc',
        'best_selected_metric': -1.0,
        'best_accuracy': 0.0, 'best_f1': 0.0, 'best_auc': 0.0,
        'best_epoch': 0,
        # Peak values across all epochs (independent from selection criterion).
        'peak_accuracy': 0.0, 'peak_f1': 0.0, 'peak_auc': 0.0,
    }

    task_keys = cfg.TASKS
    display_names = get_display_names(task_keys)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Tasks: {', '.join(display_names)}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_losses = train_epoch(model, train_loader, criteria, optimizer, device, task_keys)
        history['train_loss'].append(train_losses['total'])

        # Validate
        val_results = validate_epoch(model, val_loader, criteria, device, task_keys)
        history['val_loss'].append(val_results['avg_loss'])
        history['val_accuracy'].append(val_results['avg_accuracy'])
        history['val_f1'].append(val_results['avg_f1'])
        history['val_auc'].append(val_results['avg_auc'])

        # Step scheduler
        scheduler.step()

        # Log
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val   Loss: {val_results['avg_loss']:.4f} | "
              f"Acc: {val_results['avg_accuracy']:.4f} | "
              f"F1: {val_results['avg_f1']:.4f} | "
              f"AUC: {val_results['avg_auc']:.4f}")

        for key, name in zip(task_keys, display_names):
            r = val_results[key]
            print(f"    {name}: Acc={r['accuracy']:.3f}, F1={r['f1']:.3f}, AUC={r['auc']:.3f}")

        history['peak_accuracy'] = max(history['peak_accuracy'], val_results['avg_accuracy'])
        history['peak_f1'] = max(history['peak_f1'], val_results['avg_f1'])
        history['peak_auc'] = max(history['peak_auc'], val_results['avg_auc'])

        # Save best model by avg AUC.
        if val_results['avg_auc'] > history['best_selected_metric']:
            history['best_selected_metric'] = val_results['avg_auc']
            history['best_accuracy'] = val_results['avg_accuracy']
            history['best_f1'] = val_results['avg_f1']
            history['best_auc'] = val_results['avg_auc']
            history['best_epoch'] = epoch
            print(f"  New best checkpoint by AUC at epoch {epoch} (AUC={val_results['avg_auc']:.4f})")
            model.save_model(os.path.join(output_dir, 'best_model.pt'))

            # Save best validation results
            best_results = {key: val_results[key] for key in task_keys}
            best_results['avg_accuracy'] = val_results['avg_accuracy']
            best_results['avg_f1'] = val_results['avg_f1']
            best_results['avg_auc'] = val_results['avg_auc']
            best_results['selection_metric'] = 'avg_auc'
            best_results['selected_epoch'] = epoch
            with open(os.path.join(output_dir, 'best_results.json'), 'w') as f:
                json.dump(best_results, f, indent=2)

    # Save final history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(
        os.path.join(output_dir, 'training_history.json'),
        os.path.join(output_dir, 'training_curves.png'),
        show=False,
    )
    plot_confusion_matrices(
        os.path.join(output_dir, 'best_results.json'),
        task_keys, display_names,
        os.path.join(output_dir, 'confusion_matrices.png'),
        show=False,
    )
    plot_auc_curves(
        os.path.join(output_dir, 'best_results.json'),
        task_keys, display_names,
        os.path.join(output_dir, 'auc_curves.png'),
        show=False,
    )

    print(f"\nTraining complete!")
    print(f"Best epoch (by AUC): {history['best_epoch']}")
    print(f"Best AUC (selection metric): {history['best_auc']:.4f}")
    print(f"Accuracy @ Best AUC: {history['best_accuracy']:.4f}")
    print(f"F1 @ Best AUC: {history['best_f1']:.4f}")
    print(f"Peak accuracy (all epochs): {history['peak_accuracy']:.4f}")
    print(f"Peak F1 (all epochs): {history['peak_f1']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
