__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
import yaml


def get_task_config() -> argparse.Namespace:
    """Parses and validates configuration arguments for the Birdsong Pipeline."""

    parser = argparse.ArgumentParser(
        prog='Birdsong Syllable Detection Pipeline',
        description='Executes SSL, Supervised, or Semi-Supervised training.'
    )

    # system
    sys_group = parser.add_argument_group('System & Environment')
    sys_group.add_argument('-c', '--config', type=str, help='Path to YAML config file.')
    sys_group.add_argument('--device', default='cuda', type=str)
    sys_group.add_argument('--train-num-workers', default=8, type=int)
    sys_group.add_argument('--val-num-workers', default=4, type=int)
    sys_group.add_argument('--test-num-workers', default=4, type=int)
    sys_group.add_argument('--unlabeled-num-workers', default=8, type=int)
    sys_group.add_argument('--save-interval', type=int)
    # sys_group.add_argument('--compile', action="store_true", help='Use PyTorch 2.0 torch.compile.')

    # data
    data_group = parser.add_argument_group('Data & Features')
    data_group.add_argument('--canary-df-path', type=str, help='Path to curated canary CSV')
    data_group.add_argument('--bengalese-finch-df-path', type=str, help='Path to curated bengalese finch CSV')
    data_group.add_argument('--species', type=str, choices=['canary', 'bengalese_finch', 'mixed'])
    data_group.add_argument('--bird', type=str, help='Specific individual bird ID for downstream tasks.')
    data_group.add_argument('--sr', default=32000, type=int)
    data_group.add_argument('--n-fft', default=512, type=int)
    data_group.add_argument('--hop-length', default=64, type=int)
    data_group.add_argument('--sample-dur', default=10, type=float, help='Duration in seconds for training crops')
    data_group.add_argument('--unlabeled-sample-dur', default=10, type=float)

    # model
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--num-classes', type=int, help='Includes background class (0).')
    model_group.add_argument('--num-clusters', default=1024, type=int, help='For OSC task.')
    model_group.add_argument('--input-dim', default=256, type=int, help='Should match n_fft // 2')
    model_group.add_argument('--hidden-dim', default=512, type=int)
    model_group.add_argument('--num-layers', default=2, type=int)
    model_group.add_argument('--encoder-dropout', default=0.25, type=float)
    model_group.add_argument('--head-dropout', default=0.25, type=float)
    model_group.add_argument('--head-norm', action="store_true")
    model_group.add_argument('--pretrained-path', type=str, default="", help='Path to load pretrained weights.')
    model_group.add_argument('--finetune', action="store_true", help='If False, freezes encoder.')

    # training
    train_group = parser.add_argument_group('Training & Optimization')
    train_group.add_argument('--train-batch-size', default=32, type=int)
    train_group.add_argument('--val-batch-size', default=4, type=int)
    train_group.add_argument('--test-batch-size', default=4, type=int)
    train_group.add_argument('--unlabeled-batch-size', default=32, type=int)
    train_group.add_argument('--optimization-steps', default=10000, type=int)
    train_group.add_argument('--val-frequency', default=100, type=int)
    train_group.add_argument('--metric-to-track', default='f1', type=str)
    train_group.add_argument('--lr', default=1e-3, type=float)
    train_group.add_argument('--min-lr', default=1e-6, type=float)
    train_group.add_argument('--lr-warmup-steps', default=1000, type=int)
    train_group.add_argument('--weight-decay', default=0.001, type=float)
    train_group.add_argument('--adam-beta1', default=0.9, type=float)
    train_group.add_argument('--adam-beta2', default=0.999, type=float)
    train_group.add_argument('--encoder-lr-scale', default=1.0, type=float)
    train_group.add_argument('--grad-clip-norm', type=float, default=1.0)
    train_group.add_argument('--grad-accumulation-steps', default=1, type=int)

    # augmentations
    aug_group = parser.add_argument_group('Augmentations')
    aug_group.add_argument('--augment', action="store_true", help='Enable augmentations during training')
    aug_group.add_argument('--use-bernoulli-noise', action="store_true")
    aug_group.add_argument('--bernoulli-noise-chance', default=0.9, type=float)
    aug_group.add_argument('--use-random-gain', action="store_true")
    aug_group.add_argument('--use-color-noise', action="store_true")
    aug_group.add_argument('--color-noise-chance', default=0.9, type=float)
    aug_group.add_argument('--use-tf-mask', action="store_true")
    aug_group.add_argument('--freq-mask-param', default=8, type=int)
    aug_group.add_argument('--time-mask-param', default=8, type=int)
    aug_group.add_argument('--tf-mask-repeats', default=1, type=int)

    # task
    task_group = parser.add_argument_group('Task Specifics')
    task_group.add_argument('--task', type=str,
                            choices=['ssl_mae', 'ssl_osc', 'supervised', 'semi-supervised'])
    task_group.add_argument('--confidence-margin', type=float, default=0.95)
    task_group.add_argument('--consistency-loss-weight', type=float, default=1.0)
    task_group.add_argument('--ema-decay-start', type=float, default=0.995)
    task_group.add_argument('--ema-decay-end', type=float, default=0.99998)
    task_group.add_argument('--ema-warmup-steps', type=int, default=5000)

    # we parse config file first, then command line overrides it
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                safe_config = {k.replace('-', '_'): v for k, v in yaml_config.items()}
                parser.set_defaults(**safe_config)

    args = parser.parse_args()
    return _validate_and_format_args(args)


def _validate_and_format_args(args: argparse.Namespace) -> argparse.Namespace:
    assert args.task in ['ssl_mae', 'ssl_osc', 'supervised', 'semi-supervised'], args.task
    assert args.species in ['canary', 'bengalese_finch', 'mixed'], args.species

    if args.task in ['supervised', 'semi-supervised']:
        if not args.bird:
            raise ValueError(f"Task '{args.task}' requires the '--bird' argument to target a specific individual.")

    return args
