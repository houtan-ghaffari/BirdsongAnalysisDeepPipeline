"""
Execution entry point for the Birdsong Training Pipeline.
"""

__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from config import get_task_config
from engine import seed_everything

def main() -> None:
    seed_everything()
    args = get_task_config()

    # Route to the correct training pipeline based on the task
    if args.task == 'ssl_mae':
        from engine.ssl_trainer import run_ssl_mae
        run_ssl_mae(args)

    elif args.task == 'ssl_osc':
        from engine.ssl_trainer import run_ssl_osc
        run_ssl_osc(args)

    elif args.task == 'supervised':
        from engine.trainer import run_supervised_experiment
        run_supervised_experiment(args)

    elif args.task == 'semi-supervised':
        from engine.semisl_trainer import run_semisupervised_experiment
        run_semisupervised_experiment(args)

    else:
        raise ValueError(f"Unknown task configuration: {args.task}")

if __name__ == '__main__':
    main()