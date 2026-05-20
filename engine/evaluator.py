__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics


@torch.inference_mode()
def test_segmentation_full_length(model, dataloader, data_transforms, device='cuda'):
    model.eval()
    y_true, y_pred = [], []

    for inputs, targets in tqdm(dataloader):
        inputs = data_transforms(inputs.to(device), augment=False)
        shortest_time_frame_size = min(inputs.shape[1], targets.shape[1])
        inputs, targets = inputs[:, :shortest_time_frame_size], targets[:, :shortest_time_frame_size]
        print(inputs.shape, targets.shape)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(inputs)  # B, T, C

        predictions = logits.float().cpu().argmax(dim=2)  # B, T

        predictions = predictions.flatten()  # B, T -> B*T
        targets = targets.flatten()  # B, T -> B*T,
        valid_frames = targets != -1  # to batch, we padded shorter inputs with -1, and we ignore them now
        predictions = predictions[valid_frames]  # n, C
        targets = targets[valid_frames]  # n,

        y_true.append(targets.long())
        y_pred.append(predictions.long())

    y_true = torch.cat(y_true).numpy()  # N,
    y_pred = torch.cat(y_pred).numpy()  # N,

    acc = round(np.mean(y_true == y_pred) * 100, ndigits=2)
    f1 = round(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)
    precision = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)
    recall = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)

    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}


@torch.no_grad()
def test_segmentation_sliding_window(model,
                                     dataloader,
                                     data_transforms,
                                     device='cuda',
                                     chunk_size=15000,  # ~30 seconds => [(sr = 32 kHz) / (hop=64)] = 500 fps
                                     stride=7500,  # 50% overlap (15 seconds)
                                     ):
    """
    Evaluates segmentation using an overlapping sliding window inference.
    This prevents cuDNN sequence-length crashes on massive uncropped files and uses a Hamming window to prevent
    edge artifacts at chunk boundaries.
    """

    model.eval()
    y_true, y_pred = [], []

    for inputs, targets in dataloader:
        inputs = data_transforms(inputs.to(device), augment=False)
        shortest_time_frame_size = min(inputs.shape[1], targets.shape[1])
        inputs = inputs[:, :shortest_time_frame_size]
        targets = targets[:, :shortest_time_frame_size]

        final_logits = None
        weight_sum = None

        window_weight = torch.hamming_window(chunk_size).to(device)
        window_weight = window_weight.view(1, -1, 1)  # reshape for broadcasting: (1, T, 1)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            for start in range(0, inputs.shape[1], stride):
                end = min(start + chunk_size, inputs.shape[1])
                current_chunk_size = end - start

                chunk_logits = model(inputs[:, start:end])

                # initialize accumulation tensors on the first pass
                if final_logits is None:
                    batch_size, _, num_classes = chunk_logits.shape
                    final_logits = torch.zeros((batch_size, inputs.shape[1], num_classes), device=device)
                    weight_sum = torch.zeros((batch_size, inputs.shape[1], 1), device=device)

                # slice the weight window in case the very last chunk is smaller than chunk_size
                current_weight = window_weight[:, :current_chunk_size, :]

                # accumulate the weighted logits
                final_logits[:, start:end, :] += chunk_logits.float() * current_weight
                weight_sum[:, start:end, :] += current_weight

        # average the overlapping predictions based on accumulated weights
        logits = final_logits / (weight_sum + 1e-8)

        predictions = logits.argmax(dim=2).cpu()  # B, T

        # filter out padded areas (-1)
        predictions = predictions.flatten()  # B, T -> B*T
        targets = targets.flatten()  # B, T -> B*T
        valid_frames = targets != -1
        predictions = predictions[valid_frames]
        targets = targets[valid_frames]
        y_true.append(targets.long())
        y_pred.append(predictions.long())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = round(np.mean(y_true == y_pred) * 100, ndigits=2)
    f1 = round(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)
    precision = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)
    recall = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, ndigits=2)

    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
