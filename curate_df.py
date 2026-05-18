__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import concurrent.futures


def assign_bird_splits(df: pd.DataFrame, train_fraction: float, val_fraction: float) -> pd.DataFrame:
    """
    Assigns each unique audio file to 'train', 'val', or 'test' for a single bird.
    Ensures every syllable class is represented in all active splits before filling quotas.
    """
    unique_files = df['audio_file'].unique()
    total_files = len(unique_files)
    target_train = max(1, int(total_files * train_fraction))
    target_val = max(1, int(total_files * val_fraction)) if val_fraction > 0 else 0

    # dictionary mapping each file to the set of labels it contains
    file_to_labels = df.groupby('audio_file')['label'].unique().apply(set).to_dict()

    # global label counts for this bird to prioritize minority classes
    label_counts = df['label'].value_counts().sort_values()
    sorted_labels = label_counts.index.tolist()

    # dictionary mapping each label to a list of files that contain it
    label_to_files = {label: [] for label in sorted_labels}
    for f, labels in file_to_labels.items():
        for l in labels:
            label_to_files[l].append(f)

    splits_dict = {'train': set(), 'test': set()}
    if val_fraction > 0:
        splits_dict['val'] = set()

    available_files = set(unique_files)

    # a minimal set ensures every syllable is present in all active splits
    for split_name in splits_dict.keys():
        represented_labels = set()
        for label in sorted_labels:
            if label in represented_labels:
                continue

            candidates = [f for f in label_to_files[label] if f in available_files]

            if not candidates:
                raise ValueError(f"Data shortage to create representative splits for label {label}! Consider removing "
                                 f"this syllable label or adding more data.")

            chosen_file = np.random.choice(candidates)
            splits_dict[split_name].add(chosen_file)
            available_files.remove(chosen_file)

            represented_labels.update(file_to_labels[chosen_file])

    # we fill the remaining quotas, prioritizing minority classes
    needed_train = target_train - len(splits_dict['train'])
    needed_val = target_val - len(splits_dict.get('val', set()))

    remaining_list = list(available_files)

    def sample_files(needed, remaining):
        if needed <= 0 or not remaining:
            return [], remaining

        weights = []
        for f in remaining:
            f_labels = file_to_labels[f]
            w = sum(1.0 / label_counts[l] for l in f_labels)
            weights.append(w)

        weights = np.array(weights)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(len(remaining)) / len(remaining)

        chosen = np.random.choice(remaining, size=min(needed, len(remaining)), replace=False, p=weights)
        return chosen.tolist(), list(set(remaining) - set(chosen))

    # fill train
    add_train, remaining_list = sample_files(needed_train, remaining_list)
    splits_dict['train'].update(add_train)

    # fill val (if applicable)
    if val_fraction > 0:
        add_val, remaining_list = sample_files(needed_val, remaining_list)
        splits_dict['val'].update(add_val)

    # the rest defaults to test
    splits_dict['test'].update(remaining_list)

    # map the assignments back to the dataframe rows
    file_to_split = {}
    for split_name, files in splits_dict.items():
        for f in files:
            file_to_split[f] = split_name

    df['split'] = df['audio_file'].map(file_to_split)
    return df


# Canaries
def prepare_canary_df(root_dir="/home/Datasets/canary/",
                      train_fraction=0.05,
                      val_fraction=0.0):
    base_path = Path(root_dir)

    bird2dfs = []
    birds = ['llb11', 'llb16', 'llb3']

    for bird in birds:
        path_to_data = base_path / f"{bird}_data" / f"{bird}_annot.csv"
        df = pd.read_csv(path_to_data)

        prefix = str(base_path / f"{bird}_data" / f"{bird}_songs") + "/"
        df['audio_file'] = prefix + df['audio_file'].astype(str)
        df['bird'] = bird
        bird2dfs.append(df)

    df = pd.concat(bird2dfs, ignore_index=True)

    def get_duration_and_sr(path: str):
        try:
            x, sr = sf.read(path)
            return path, (x.shape[0] / sr), sr
        except Exception as e:
            print(f"Warning: {path} is corrupt/unreadable. Error: {e}")
            return path, None, None

    file_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        results = executor.map(get_duration_and_sr, df['audio_file'].unique())
        for path, dur, sr in results:
            if dur is not None:
                file_data[path] = {'audio_duration': dur, 'native_sr': sr}

    meta_df = pd.DataFrame.from_dict(file_data, orient='index').reset_index(names='audio_file')
    df = df.merge(meta_df, on='audio_file', how='inner')

    # df = df.groupby('bird', group_keys=False).apply(lambda x: assign_bird_splits(x, train_fraction, val_fraction))
    dfs = []
    for _, group in df.groupby('bird'):
        dfs.append(assign_bird_splits(group.copy(), train_fraction, val_fraction))
    df = pd.concat(dfs, ignore_index=True)

    df['file_id'] = pd.factorize(df['audio_file'])[0]
    df['row_id'] = df.index
    df = df.set_index(['file_id', 'row_id'])

    save_path = base_path / "curated_canary_df.csv"
    df.to_csv(save_path)
    return df


# Bengalese Finch
def prepare_bengalese_finch_df(root_dir="/home/Datasets/bengalese_finch/",
                               train_fraction=0.05,
                               val_fraction=0.0):
    birds_keys = {
        'bl26lb16': ['a', 'b', 'c', 'd', 'e', 'f', 'i'],
        'gr41rd51': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'm'],
        'gy6or6': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
        'or60yw70': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i'],
    }

    base_path = Path(root_dir)

    bird2dfs = []
    birds = ['bl26lb16', 'gr41rd51', 'gy6or6', 'or60yw70']

    for bird in birds:
        dfs = []
        path_to_data = base_path / f"{bird}"
        wav_files = list(path_to_data.rglob("*.wav"))

        valid_labels = set(birds_keys[bird])

        for f in wav_files:
            csv_path = f.as_posix() + '.csv'
            if Path(csv_path).is_file():
                df = pd.read_csv(csv_path)

                file_labels = set(df['label'].unique())
                if file_labels.issubset(valid_labels):
                    df['audio_file'] = f.as_posix()
                    df['bird'] = bird
                    dfs.append(df)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            label_map = {label: i for i, label in enumerate(birds_keys[bird], start=1)}
            df['label'] = df['label'].map(label_map)
            bird2dfs.append(df)

    df = pd.concat(bird2dfs, ignore_index=True)

    def get_actual_duration_and_sr(path: str):
        try:
            x, sr = sf.read(path)
            return path, (x.shape[0] / sr), sr
        except Exception as e:
            print(f"Warning: {path} is corrupt/unreadable. Error: {e}")
            return path, None, None

    file_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        results = executor.map(get_actual_duration_and_sr, df['audio_file'].unique())
        for path, dur, sr in results:
            if dur is not None:
                file_data[path] = {'audio_duration': dur, 'native_sr': sr}

    meta_df = pd.DataFrame.from_dict(file_data, orient='index').reset_index(names='audio_file')
    df = df.merge(meta_df, on='audio_file', how='inner')

    dfs = []
    for _, group in df.groupby('bird'):
        dfs.append(assign_bird_splits(group.copy(), train_fraction, val_fraction))
    df = pd.concat(dfs, ignore_index=True)

    # multi-level indexing helps for efficient data loading later
    df['file_id'] = pd.factorize(df['audio_file'])[0]
    df['row_id'] = df.index
    df = df.set_index(['file_id', 'row_id'])

    save_path = base_path / "curated_bengalese_finch_df.csv"
    df.to_csv(save_path)

    return df


if __name__ == "__main__":
    root_dir = "/home/Datasets/"

    print(f"\n[INFO] Curating Bengalese Finch Dataset Metadata...")

    _ = prepare_bengalese_finch_df(root_dir=root_dir + "bengalese_finch/", train_fraction=0.05, val_fraction=0.1)

    print(f"\n[INFO] Curated Bengalese Finch Dataset Metadata is Saved at:"
          f" {root_dir + 'bengalese_finch/curated_bengalese_finch_df.csv'}")

    print(f"\n[INFO] Curating Canary Dataset Metadata...")

    _ = prepare_canary_df(root_dir=root_dir + "canary/", train_fraction=0.05, val_fraction=0.1)

    print(f"\n[INFO] Curated Canary Dataset Metadata is Saved at:"
          f" {root_dir + 'canary/curated_canary_df.csv'}")
