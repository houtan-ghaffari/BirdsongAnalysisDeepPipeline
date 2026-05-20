__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from typing import Optional, Any
import argparse
from tqdm import tqdm

from data.factory import get_dataloaders
from models.builder import get_model
from engine.optim import get_optimizer, RiseRunDecay, EMA_Scheduler, ema_update
from models.ssl import ClusterLoss
from engine import infinite_batch_iterator


# *********************
# Masked Auto-Encoder
# *********************

def mae_train_step(model: torch.nn.Module,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any = None,
                   device: str = 'cuda',
                   clip_norm: Optional[float] = None,
                   accumulation_steps: int = 1,
                   is_accumulating: bool = False):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        outputs = model(inputs)

    loss = (targets - outputs.float()).pow(2.).mean()
    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    if not is_accumulating:
        if clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item()


def run_ssl_mae(args: argparse.Namespace) -> None:
    """
    Sets up and executes the Masked Auto-Encoder Self-Supervised pretraining pipeline.
    """
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/SSL')
    log_dir.mkdir(parents=True, exist_ok=True)
    history_save_path = log_dir / f'MAE_history_{args.species}_{time_stamp}.csv'
    state_save_path = f'MAE_state_{args.species}_{time_stamp}.pt'

    train_loader, data_transforms = get_dataloaders(args)
    infinite_loader = infinite_batch_iterator(train_loader)
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.optimization_steps,
                             min_lr=args.min_lr)

    save_freq = args.save_interval if args.save_interval else args.optimization_steps
    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps

    print(f'#parameters: {sum([p.numel() for p in model.parameters()]):_}')

    pbar = tqdm(total=args.optimization_steps, desc="MAE Pretraining", colour='#87ceeb')
    history = {'reconstruction_loss': []}
    loss = 0
    for step in range(total_forward_passes):
        x = next(infinite_loader)
        x = x.to(args.device, non_blocking=True)
        with torch.no_grad():
            x, y = data_transforms(x)

        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        step_loss = mae_train_step(
            model=model, inputs=x, targets=y, optimizer=optimizer, scheduler=scheduler, device=args.device,
            clip_norm=args.grad_clip_norm, accumulation_steps=args.grad_accumulation_steps,
            is_accumulating=is_accumulating
        )

        loss += step_loss

        if not is_accumulating:
            pbar.update(1)
            loss = loss / args.grad_accumulation_steps
            history['reconstruction_loss'].append(loss)
            if pbar.n % save_freq == 0 or pbar.n == args.optimization_steps:
                current_save_path = log_dir / f'step({pbar.n})_{state_save_path}'
                torch.save({'state_dict': model.state_dict(), 'history': history, 'args': vars(args)},
                           current_save_path)
                print(f"\n[INFO] SSL Model saved at: {current_save_path}")
            pbar.set_description(f"loss={loss:3.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            loss = 0

    pbar.close()
    pd.DataFrame(history).to_csv(history_save_path, index=False)


# ***************************
# Online Syllable Clustering
# ***************************

def osc_train_step(student: torch.nn.Module,
                   teacher: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   concatenated_views: torch.Tensor,
                   scheduler: Any,
                   cluster_loss: torch.nn.Module,
                   ema_scheduler: Any,
                   device='cuda',
                   clip_norm: Optional[float] = None,
                   accumulation_steps: int = 1,
                   is_accumulating: bool = False):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        student_logits = student(concatenated_views)  # B, T, D -> B*T, C
        with torch.no_grad():
            teacher_logits = teacher(concatenated_views)
            t1, t2 = teacher_logits.chunk(2)
            teacher_logits = torch.cat([t2, t1], dim=0)  # cross-view

    loss, xe_loss, gini_loss = cluster_loss(student_logits.float(), teacher_logits.float())

    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    if not is_accumulating:
        if clip_norm is not None: torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        decay = ema_scheduler.step()
        ema_update(teacher, student, decay=decay)

    return loss.item(), xe_loss.item(), -gini_loss


def run_ssl_osc(args: argparse.Namespace) -> None:
    """
    Sets up and executes the Online Syllable Clustring Self-Supervised pretraining pipeline.
    """
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/SSL')
    log_dir.mkdir(parents=True, exist_ok=True)
    history_save_path = log_dir / f'OSC_history_{args.species}_{time_stamp}.csv'
    state_save_path = f'OSC_state_{args.species}_{time_stamp}.pt'

    train_loader, data_transforms = get_dataloaders(args)
    infinite_train_loader = infinite_batch_iterator(train_loader)
    student, teacher = get_model(args)
    cluster_loss = ClusterLoss()
    ema_scheduler = EMA_Scheduler(decay_start=args.ema_decay_start, decay_end=args.ema_decay_end,
                                  ema_warmup_steps=args.ema_warmup_steps)
    optimizer = get_optimizer(student, args)
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.optimization_steps,
                             min_lr=args.min_lr)

    save_freq = args.save_interval if args.save_interval else args.optimization_steps
    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    pbar = tqdm(total=args.optimization_steps, desc="OSC Pretraining", colour='#87ceeb')
    history = {'loss': [], 'xe_loss': [], 'gini_impurity': []}
    loss, xe_loss, gini_impurity = 0, 0, 0

    for step in range(total_forward_passes):

        x = next(infinite_train_loader)
        x = x.to(args.device, non_blocking=True)
        with torch.no_grad():
            x = torch.cat([data_transforms(x), data_transforms(x)], dim=0)  # 2 views concatenated

        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        step_loss, step_xe_loss, step_gini = osc_train_step(
            student=student, teacher=teacher, optimizer=optimizer, concatenated_views=x, scheduler=scheduler,
            cluster_loss=cluster_loss, ema_scheduler=ema_scheduler, device=args.device, clip_norm=args.grad_clip_norm,
            accumulation_steps=args.grad_accumulation_steps, is_accumulating=is_accumulating
        )

        loss += step_loss
        xe_loss += step_xe_loss
        gini_impurity += step_gini

        if not is_accumulating:
            pbar.update(1)
            loss = loss / args.grad_accumulation_steps
            xe_loss = xe_loss / args.grad_accumulation_steps
            gini_impurity = gini_impurity / args.grad_accumulation_steps
            history['loss'].append(loss)
            history['xe_loss'].append(xe_loss)
            history['gini_impurity'].append(gini_impurity)

            if pbar.n % save_freq == 0 or pbar.n == args.optimization_steps:
                current_save_path = log_dir / f'step({pbar.n})_{state_save_path}'
                torch.save({'state_dict': student.state_dict(), 'history': history, 'args': vars(args)},
                           current_save_path)
                print(f"\n[INFO] SSL Model saved at: {current_save_path}")

            pbar.set_description(f"loss={loss:3.4f} | xe_loss={xe_loss:3.4f} | gini_impurity={gini_impurity:3.6f} | "
                                 f"LR: {scheduler.get_last_lr()[0]:.2e}")
            loss, xe_loss, gini_impurity = 0, 0, 0

    pbar.close()
    pd.DataFrame(history).to_csv(history_save_path, index=False)
