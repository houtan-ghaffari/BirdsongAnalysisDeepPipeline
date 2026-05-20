__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import gc
from datetime import datetime
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, Callable
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine.utils import infinite_batch_iterator
from data.factory import get_dataloaders
from models.builder import get_model
from engine.optim import get_optimizer, RiseRunDecay
from engine.evaluator import test_segmentation_sliding_window


def supervised_train_step(model: torch.nn.Module,
                          inputs: torch.Tensor,
                          targets: torch.Tensor,
                          optimizer: torch.optim.Optimizer,
                          scheduler: Any = None,
                          device: str = 'cuda',
                          clip_norm: Optional[float] = None,
                          accumulation_steps: int = 1,
                          is_accumulating: bool = False):

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(inputs)

    logits = logits.float().flatten(0, 1)  # B, T, C -> B*T, C
    targets = targets.flatten()  # B, T -> B*T,
    valid_frames = targets != -1  # to batch, we padded shorter inputs with -1, and we ignore them now
    logits = logits[valid_frames]
    targets = targets[valid_frames]

    loss = F.cross_entropy(logits, targets, ignore_index=-1)

    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    if not is_accumulating:
        if clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item()


def run_supervised_experiment(args):
    """Executes the Supervised finetuning pipeline for a specific bird."""

    print(f"\n[INFO] Starting Supervised Task for Bird: {args.species}-{args.bird}")

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/Supervised')
    log_dir.mkdir(parents=True, exist_ok=True)
    history_save_path = f'supervised_history_{args.species}_{args.bird}_{time_stamp}.csv'
    state_save_path = f'supervised_state_{args.species}_{args.bird}_{time_stamp}.pt'
    if args.pretrained_path is None:
        state_save_path = 'random_init_' + state_save_path
        history_save_path = 'random_init_' + history_save_path
    elif 'mae_' in args.pretrained_path.lower():
        state_save_path = 'mae_' + state_save_path
        history_save_path = 'mae_' + history_save_path
    elif 'osc_' in args.pretrained_path.lower():
        state_save_path = 'osc_' + state_save_path
        history_save_path = 'osc_' + history_save_path
    else:
        raise ValueError(f"Unknown pretrained path type: {args.pretrained_path}")
    history_save_path = log_dir / history_save_path
    state_save_path = log_dir / state_save_path
    temp_save_path = f'temp_state_{time_stamp}.pt'
    print(f"\n[INFO] Model will be saved at: {state_save_path}")

    train_loader, val_loader, test_loader, data_transforms, num_classes = get_dataloaders(args)
    args.num_classes = num_classes
    infinite_train_loader = infinite_batch_iterator(train_loader)
    track_validation = True if val_loader else False
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.optimization_steps,
                             min_lr=args.min_lr)

    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    pbar = tqdm(total=args.optimization_steps, desc="Supervised Training", colour='#87ceeb')
    history = defaultdict(list)
    loss = 0
    best_score = 0
    val_msg = ""

    for step in range(total_forward_passes):
        x, y = next(infinite_train_loader)
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        with torch.no_grad():
            x = data_transforms(x, augment=args.augment)
            t = min(x.shape[1], y.shape[1])  # 1 frame difference happens due to rounding error in data class
            x, y = x[:, :t], y[:, :t]

        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        step_loss = supervised_train_step(
            model=model, inputs=x, targets=y, optimizer=optimizer, scheduler=scheduler, device=args.device,
            clip_norm=args.grad_clip_norm, accumulation_steps=args.grad_accumulation_steps,
            is_accumulating=is_accumulating
        )

        loss += step_loss

        if not is_accumulating:
            pbar.update(1)
            loss = loss / args.grad_accumulation_steps

            if not track_validation:
                history['train_loss'].append(loss)

            elif pbar.n % args.val_frequency == 0 or pbar.n == args.optimization_steps:
                history['train_loss'].append(loss)
                val_msg = ""
                model.eval()
                val_metrics = test_segmentation_sliding_window(model, val_loader, data_transforms, args.device)
                model.train()
                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)
                    val_msg += f" | val_{k}: {v:.2f}"

                if best_score <= val_metrics[args.metric_to_track]:
                    best_score = val_metrics[args.metric_to_track]
                    torch.save(model.state_dict(), temp_save_path)

            pbar.set_description(f"train loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}" + val_msg)
            loss = 0

    pbar.close()
    pd.DataFrame(history).to_csv(history_save_path, index=False)

    if Path(temp_save_path).exists():
        model.load_state_dict(torch.load(temp_save_path))
        Path(temp_save_path).unlink()

    print("\n[INFO] Running Final Evaluation on Test Set...")
    model.eval()
    test_metrics = test_segmentation_sliding_window(model, test_loader, data_transforms, args.device)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}")

    torch.save({'state_dict': model.state_dict(), 'history': history, 'test_metrics': test_metrics, 'args': vars(args)},
               state_save_path)
