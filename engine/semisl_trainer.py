"""Semi-Supervised Post-Training Script """

__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.factory import get_dataloaders
from engine import infinite_batch_iterator
from models.builder import get_model
from engine.optim import get_optimizer, RiseRunDecay, EMA_Scheduler, ema_update
from engine.evaluator import test_segmentation_sliding_window


def semi_supervised_train_step(student: torch.nn.Module,
                               teacher: torch.nn.Module,
                               student_unlabeled_inputs: torch.Tensor,
                               teacher_unlabeled_inputs: torch.Tensor,
                               inputs: torch.Tensor,
                               targets: torch.Tensor,
                               optimizer: torch.optim.Optimizer,
                               scheduler: Any = None,
                               ema_scheduler: Any = None,
                               consistency_loss_weight: float = 1.0,
                               confidence_margin: float = 0.95,
                               device: str = 'cuda',
                               clip_norm: Optional[float] = None,
                               accumulation_steps: int = 1,
                               is_accumulating: bool = False
                               ):

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = student(inputs)
        student_unlabeled_logits = student(student_unlabeled_inputs)
        with torch.no_grad():
            teacher_unlabeled_logits = teacher(teacher_unlabeled_inputs)

    # supervised loss
    logits = logits.float().flatten(0, 1)  # B, T, C -> B*T, C
    targets = targets.flatten()  # B, T -> B*T,
    valid_frames = targets != -1
    logits = logits[valid_frames]
    targets = targets[valid_frames]
    loss = F.cross_entropy(logits, targets, ignore_index=-1)

    total_loss = loss

    # semi-supervised loss
    consistency_loss_val = 0.0  # default if no frames pass the confidence threshold
    student_unlabeled_logits = student_unlabeled_logits.float().flatten(0, 1)  # B*T, C
    teacher_unlabeled_logits = teacher_unlabeled_logits.float().flatten(0, 1)  # B*T, C

    # Convert teacher logits to probabilities
    teacher_probs = teacher_unlabeled_logits.softmax(dim=1)

    # we only use confident predictions
    max_probs, _ = teacher_probs.max(dim=1)  # maximum probability for each frame; B*T,

    # a boolean mask for frames that exceed the margin
    confident_mask = max_probs > confidence_margin

    # proceed if at least one frame in the batch passed the threshold
    if torch.any(confident_mask):
        # filter both student and teacher tensors to only keep confident frames
        confident_student_logits = student_unlabeled_logits[confident_mask]  # N, C
        confident_teacher_probs = teacher_probs[confident_mask]  # N, C

        # calculate dynamic prevalence of each class
        class_prevalence = confident_teacher_probs.mean(dim=0).detach()

        # Inverse frequency weighting: rare classes get higher weights
        class_weights = 1.0 / (class_prevalence + 1e-8)

        # normalize the weights for stable training
        active_classes_sum = class_weights.sum()
        if active_classes_sum > 0:
            num_active_classes = class_weights.shape[0]
            class_weights = (class_weights / active_classes_sum) * num_active_classes

        # soft cross-entropy with dynamic class weights on confident frames
        consistency_loss = F.cross_entropy(confident_student_logits, confident_teacher_probs, weight=class_weights)

        if not torch.isnan(consistency_loss):
            total_loss = total_loss + (consistency_loss_weight * consistency_loss)
            consistency_loss_val = consistency_loss.item()

    normalized_loss = total_loss / accumulation_steps
    normalized_loss.backward()

    if not is_accumulating:
        if clip_norm: torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        decay = ema_scheduler.step()
        ema_update(teacher, student, decay=decay)

    return loss.item(), consistency_loss_val


def run_semisupervised_experiment(args):
    """Executes the Semi-Supervised (Mean Teacher) post-training pipeline."""
    print(f"\n[INFO] Starting Semi-Supervised Task for Bird: {args.bird}")

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path('logs/SemiSupervised')
    log_dir.mkdir(parents=True, exist_ok=True)
    history_save_path = f'semi_supervised_history_{args.species}_{args.bird}_{time_stamp}.csv'
    state_save_path = f'semi_supervised_state_{args.species}_{args.bird}_{time_stamp}.pt'
    assert 'supervised' in args.pretrained_path.lower(), args.pretrained_path
    history_save_path = log_dir / history_save_path
    state_save_path = log_dir / state_save_path
    temp_save_path = f'temp_state_{time_stamp}.pt'
    print(f"\n[INFO] Model will be saved at: {state_save_path}")

    train_loader, val_loader, test_loader, data_transforms, unlabeled_loader, num_classes = get_dataloaders(args)
    args.num_classes = num_classes
    infinite_train_loader = infinite_batch_iterator(train_loader)
    infinite_unlabeled_loader = infinite_batch_iterator(unlabeled_loader)
    track_validation = True if val_loader else False

    student, teacher = get_model(args)
    optimizer = get_optimizer(student, args)
    scheduler = RiseRunDecay(optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.optimization_steps,
                             min_lr=args.min_lr)
    ema_scheduler = EMA_Scheduler(decay_start=args.ema_decay_start, decay_end=args.ema_decay_end,
                                  ema_warmup_steps=args.ema_warmup_steps)

    total_forward_passes = args.optimization_steps * args.grad_accumulation_steps
    pbar = tqdm(total=args.optimization_steps, desc="Semi-Supervised Training", colour='#87ceeb')
    history = defaultdict(list)
    loss, consistency_loss = 0, 0
    best_score = 0
    val_msg = ""

    for step in range(total_forward_passes):
        inputs, targets = next(infinite_train_loader)
        unlabeled_inputs, _ = next(infinite_unlabeled_loader)

        inputs = inputs.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        unlabeled_inputs = unlabeled_inputs.to(args.device, non_blocking=True)

        with torch.no_grad():
            inputs = data_transforms(inputs, augment=args.augment)
            student_unlabeled_inputs = data_transforms(unlabeled_inputs, augment=True)
            teacher_unlabeled_inputs = data_transforms(unlabeled_inputs, augment=False)
            t = min(inputs.shape[1], targets.shape[1])
            inputs, targets = inputs[:, :t], targets[:, :t]

        is_accumulating = (step + 1) % args.grad_accumulation_steps != 0

        step_loss, step_consistency_loss = semi_supervised_train_step(
            student=student,
            teacher=teacher,
            student_unlabeled_inputs=student_unlabeled_inputs,
            teacher_unlabeled_inputs=teacher_unlabeled_inputs,
            inputs=inputs,
            targets=targets,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_scheduler=ema_scheduler,
            consistency_loss_weight=args.consistency_loss_weight,
            confidence_margin=args.confidence_margin,
            device=args.device,
            clip_norm=args.grad_clip_norm,
            accumulation_steps=args.grad_accumulation_steps,
            is_accumulating=is_accumulating
        )

        loss += step_loss
        consistency_loss += step_consistency_loss

        if not is_accumulating:
            pbar.update(1)
            loss = loss / args.grad_accumulation_steps
            consistency_loss = consistency_loss / args.grad_accumulation_steps

            if not track_validation:
                history['train_loss'].append(loss)
                history['consistency_loss'].append(consistency_loss)

            elif pbar.n % args.val_frequency == 0 or pbar.n == args.optimization_steps:
                history['train_loss'].append(loss)
                history['consistency_loss'].append(consistency_loss)
                val_msg = ""

                student.eval()
                val_metrics = test_segmentation_sliding_window(student, val_loader, data_transforms, args.device)
                student.train()

                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)
                    val_msg += f" | val_{k}: {v:.2f}"

                if best_score <= val_metrics[args.metric_to_track]:
                    best_score = val_metrics[args.metric_to_track]
                    torch.save(student.state_dict(), temp_save_path)

            pbar.set_description(f"train loss: {loss:.4f} | cons loss: {consistency_loss:.4f} | "
                                 f"LR: {scheduler.get_last_lr()[0]:.2e}" + val_msg)
            loss, consistency_loss = 0, 0

    pbar.close()
    pd.DataFrame(history).to_csv(history_save_path, index=False)

    if Path(temp_save_path).exists():
        student.load_state_dict(torch.load(temp_save_path))
        Path(temp_save_path).unlink()

    print("\n[INFO] Running Final Evaluation on Test Set...")
    student.eval()
    test_metrics = test_segmentation_sliding_window(student, test_loader, data_transforms, args.device)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}")

    torch.save({'state_dict': student.state_dict(), 'history': history, 'test_metrics': test_metrics,
                'args': vars(args)}, state_save_path)
