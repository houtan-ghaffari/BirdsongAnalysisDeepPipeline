__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import torch

from .ssl import MAENet, OSCNet
from .rmr import RMRSyllableClassifier


def load_pretrained_state(model: torch.nn.Module, state_path: str):
    try:
        state_dict = torch.load(state_path, map_location='cpu', weights_only=False)['state_dict']
        model_dict = model.state_dict()

        filtered_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append(k)

        msg = model.load_state_dict(filtered_dict, strict=False)

        if skipped_keys:
            print(f"[INFO] Skipped {len(skipped_keys)} keys during load (missing or shape mismatch). "
                  f"This is safe for head parameters if the task is not semi-supervised.")

        print(f"[INFO] Successfully loaded pretrained weights: {msg}")

    except Exception as e:
        print(f"[ERROR] Failed to load pretrained state from {state_path}: {e}")


def get_model(args: argparse.Namespace) -> (MAENet | tuple[OSCNet, OSCNet] | RMRSyllableClassifier |
                                            tuple[RMRSyllableClassifier, RMRSyllableClassifier]):
    if args.task == 'ssl_mae':
        model = MAENet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                       encoder_dropout=args.encoder_dropout, head_dropout=args.head_dropout)
        model.to(args.device).train()
        return model

    elif args.task == 'ssl_osc':
        student = OSCNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                         encoder_dropout=args.encoder_dropout, num_clusters=args.num_clusters)

        teacher = OSCNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                         encoder_dropout=args.encoder_dropout, num_clusters=args.num_clusters)

        teacher.load_state_dict(student.state_dict())
        teacher.requires_grad_(False)

        student.to(args.device).train()
        teacher.to(args.device).eval()

        return student, teacher

    elif args.task == 'supervised':
        model = RMRSyllableClassifier(num_classes=args.num_classes, input_dim=args.input_dim,
                                      hidden_dim=args.hidden_dim,
                                      num_layers=args.num_layers, encoder_dropout=args.encoder_dropout,
                                      head_dropout=args.head_dropout, head_norm=args.head_norm)

        if args.pretrained_path:
            load_pretrained_state(model, args.pretrained_path)
        else:
            print("[INFO] Initialized the model with random weights.")

        model.to(args.device).train()

        if not args.finetune:
            model.requires_grad_(False)
            model.head.requires_grad_(True)
            print("[INFO] Froze the encoder weights for linear probing.")

        return model

    elif args.task == 'semi-supervised':
        student = RMRSyllableClassifier(num_classes=args.num_classes, input_dim=args.input_dim,
                                        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                        encoder_dropout=args.encoder_dropout, head_dropout=args.head_dropout,
                                        head_norm=args.head_norm)

        teacher = RMRSyllableClassifier(num_classes=args.num_classes, input_dim=args.input_dim,
                                        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                        encoder_dropout=args.encoder_dropout, head_dropout=args.head_dropout,
                                        head_norm=args.head_norm)

        # must have pretrained weights from supervised training (stage 2)
        load_pretrained_state(student, args.pretrained_path)

        teacher.load_state_dict(student.state_dict())
        teacher.requires_grad_(False)

        student.to(args.device).train()
        teacher.to(args.device).eval()

        return student, teacher

    else:
        raise ValueError(f"Task {args.task} is not recognized by the builder.")
