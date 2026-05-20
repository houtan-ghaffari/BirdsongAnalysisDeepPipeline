__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import math
import numpy as np
import torch
from typing import List, Optional


class RiseRunDecay(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with linear warmup, constant phase, and cosine decay."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: Optional[int] = None,
                 constant_steps: Optional[int] = None,
                 total_steps: Optional[int] = None,
                 min_lr: float = 1e-6):

        self.warmup_steps = warmup_steps or 0
        self.constant_steps = self.warmup_steps + (constant_steps or 0)
        self.total_steps = total_steps
        self.decay_interval = max(0, total_steps - self.constant_steps)
        self.min_lr = min_lr

        self.lr_scales = [param_group.get('lr_scale', 1.0) for param_group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lrs = []
        current_iteration = self.last_epoch
        if current_iteration <= self.warmup_steps and self.warmup_steps > 0:
            factor = current_iteration / self.warmup_steps
        elif current_iteration <= self.constant_steps:
            factor = 1.0
        else:
            if self.decay_interval == 0:
                factor = 0.0
            else:
                decay_iteration = current_iteration - self.constant_steps
                factor = 0.5 * (1 + math.cos(math.pi * decay_iteration / self.decay_interval))

        for lr, lr_scale in zip(self.base_lrs, self.lr_scales):
            scaled_min_lr = self.min_lr * lr_scale
            scaled_current_lr = lr * factor * lr_scale
            lrs.append(max(scaled_min_lr, scaled_current_lr))
        return lrs


class EMA_Scheduler:
    """Scheduler to gradually increase the Exponential Moving Average (EMA) decay factor."""

    def __init__(self, decay_start: float = 0.995, decay_end: float = 0.99999, ema_warmup_steps: int = 5000):
        self.decays = np.linspace(decay_start, decay_end, ema_warmup_steps, dtype=np.float32).tolist() + [decay_end]
        self.max_iter = ema_warmup_steps
        self.counter = 0

    def step(self) -> float:
        w = self.decays[min(self.counter, self.max_iter)]
        self.counter += 1
        return float(w)


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, buffers: bool = True, decay: float = 0.999) -> None:
    """Updates the exponential moving average of the given model's parameters and buffers."""
    for p_avg, p in zip(ema_model.parameters(), model.parameters()):
        p_avg.data = decay * p_avg.data + (1. - decay) * p.data
    if buffers:
        for (n, b_avg), (n2, b) in zip(ema_model.named_buffers(), model.named_buffers()):
            if n.split('.')[-1] == 'num_batches_tracked':
                b_avg.data = b.data
            else:
                b_avg.data = decay * b_avg.data + (1. - decay) * b.data


def get_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_decay_params = []
    encoder_no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = param.ndim == 1 or "bias" in name or "norm" in name

        if 'encoder' in name:
            if no_decay:
                encoder_no_decay_params.append(param)
            else:
                encoder_decay_params.append(param)
        else:
            if no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)

    encoder_lr = args.lr * args.encoder_lr_scale

    optim_groups = [
        {'params': encoder_decay_params, 'lr': encoder_lr, 'weight_decay': args.weight_decay},
        {'params': encoder_no_decay_params, 'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': head_decay_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': head_no_decay_params, 'lr': args.lr, 'weight_decay': 0.0}
    ]

    return torch.optim.AdamW(optim_groups, betas=(args.adam_beta1, args.adam_beta2))
