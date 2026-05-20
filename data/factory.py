__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import argparse
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from .birdsong_datasets import SSLDataset, BirdSongDataset, OSCDataTransforms, MAEDataTransforms, DataTransforms


def _prepare_split_df(df_split: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to reset indices, convert strings to bytes to prevent memory leaks,
    and recreate the multi-level index required by the BirdSongDataset.
    """
    if df_split.empty:
        return df_split

    # reset indices to remove gaps from splitting
    df_split = df_split.reset_index(drop=True)

    # this is to prevent memory leak in data loader workers
    df_split['audio_file'] = df_split['audio_file'].astype(np.bytes_)

    # recreate multi-level indexing based on unique audio files
    df_split['file_id'] = pd.factorize(df_split['audio_file'])[0]
    df_split['row_id'] = df_split.index

    return df_split.set_index(['file_id', 'row_id'])


def get_dataloaders(args: argparse.Namespace):
    """Routes dataset initialization based on args.task."""

    if args.task in ['ssl_osc', 'ssl_mae']:  # SSL Tasks
        dfs = []
        species_list = ['canary', 'bengalese_finch'] if args.species == 'mixed' else [args.species]

        for sp in species_list:
            if sp == 'canary':
                df = pd.read_csv(args.canary_df_path, index_col=[0, 1])
            elif sp == 'bengalese_finch':
                df = pd.read_csv(args.bengalese_finch_df_path, index_col=[0, 1])
            else:
                raise ValueError(f"Species {sp} is not recognized for SSL.")

            # for SSL tasks, we only need 'audio_file' and 'audio_duration' fields.
            # we remove multi-index/row duplicates to keep only one row per file.
            df_ssl = df[['audio_file', 'audio_duration']].drop_duplicates().reset_index(drop=True)
            df_ssl['audio_file'] = df_ssl['audio_file'].astype(np.bytes_)
            dfs.append(df_ssl)

        df = pd.concat(dfs, ignore_index=True)
        assert not df.empty, "dataframe is empty!"

        train_set = SSLDataset(df, sample_dur=args.sample_dur, sr=args.sr)
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.train_num_workers, collate_fn=train_set.collate_fn,
                                  persistent_workers=True)

        if args.task == 'ssl_osc':
            data_transforms = OSCDataTransforms(
                sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, use_random_gain=args.use_random_gain,
                use_color_noise=args.use_color_noise, color_noise_chance=args.color_noise_chance,
                use_bernoulli_noise=args.use_bernoulli_noise, bernoulli_noise_chance=args.bernoulli_noise_chance)

        elif args.task == 'ssl_mae':
            data_transforms = MAEDataTransforms(
                sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, use_random_gain=args.use_random_gain,
                use_color_noise=args.use_color_noise, color_noise_chance=args.color_noise_chance)

        return train_loader, data_transforms.to(args.device)

    # Supervised and Semi-Supervised Tasks
    if args.species == 'canary':
        df = pd.read_csv(args.canary_df_path, index_col=[0, 1])
    elif args.species == 'bengalese_finch':
        df = pd.read_csv(args.bengalese_finch_df_path, index_col=[0, 1])
    else:
        raise ValueError(f'Species {args.species} is not recognized')

    # filter the df based on 'bird' column to have only files for the specific individual
    if not hasattr(args, 'bird') or args.bird is None:
        raise ValueError("args.bird must be specified for supervised tasks.")

    df = df[df['bird'] == args.bird].copy()
    if df.empty:
        raise ValueError(f"No data found for bird: {args.bird}!")

    train_df = _prepare_split_df(df[df['split'] == 'train'])
    test_df = _prepare_split_df(df[df['split'] == 'test'])
    val_df = _prepare_split_df(df[df['split'] == 'val'])

    assert train_df.label.unique().shape[0] == test_df.label.unique().shape[0]
    num_classes = train_df.label.unique().shape[0] + 1  # +1 for background class

    train_set = BirdSongDataset(train_df, sample_dur=args.sample_dur, sr=args.sr, hop_length=args.hop_length, crop=True)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, persistent_workers=True,
                              collate_fn=train_set.collate_fn, num_workers=args.train_num_workers)

    test_set = BirdSongDataset(test_df, sr=args.sr, hop_length=args.hop_length, test=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, persistent_workers=False,
                             collate_fn=test_set.collate_fn, num_workers=args.test_num_workers)

    if not val_df.empty:
        val_set = BirdSongDataset(val_df, sr=args.sr, hop_length=args.hop_length, test=True)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, collate_fn=val_set.collate_fn,
                                num_workers=args.val_num_workers, persistent_workers=True)
        assert val_df.label.unique().shape[0] == test_df.label.unique().shape[0]
    else:
        val_loader = None

    data_transforms = DataTransforms(
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, use_bernoulli_noise=args.use_bernoulli_noise,
        use_random_gain=args.use_random_gain, use_color_noise=args.use_color_noise, use_tf_mask=args.use_tf_mask,
        color_noise_chance=args.color_noise_chance, bernoulli_noise_chance=args.bernoulli_noise_chance,
        freq_mask_param=args.freq_mask_param, time_mask_param=args.time_mask_param,
        tf_mask_repeats=args.tf_mask_repeats).to(args.device)

    if args.task == 'supervised':
        return train_loader, val_loader, test_loader, data_transforms, num_classes

    # for Semi-Supervised
    unlabeled_set = BirdSongDataset(test_df, sample_dur=args.unlabeled_sample_dur, sr=args.sr,
                                    hop_length=args.hop_length,
                                    crop=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.unlabeled_batch_size, shuffle=True,
                                  persistent_workers=True, collate_fn=unlabeled_set.collate_fn,
                                  num_workers=args.unlabeled_num_workers, drop_last=True)

    return train_loader, val_loader, test_loader, data_transforms, unlabeled_loader, num_classes
