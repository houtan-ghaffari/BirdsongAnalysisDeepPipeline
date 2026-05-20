__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import math
import random
import librosa
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking

from typing import Tuple, List, Union
from .augmentations import apply_color_noise, apply_random_gain, apply_bernoulli_noise


def minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.flatten(1)
    min_, max_ = x.aminmax(dim=1, keepdim=True)
    x = (x - min_) / (max_ - min_ + 1e-9)
    return x.reshape(shape)


class BirdSongDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 sample_dur: Union[int, float] = 10,
                 sr: int = 32000,
                 hop_length: int = 64,
                 test: bool = False,
                 crop: bool = False) -> None:

        super().__init__()
        self.df = df
        self.sr = sr
        self.hop_length = hop_length
        self.num_samples = int(sample_dur * sr)
        self.sample_dur = sample_dur
        self.test = test
        self.crop = crop
        self.size = self.df.audio_file.unique().shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.df.loc[idx]  # multi-index to get multiple rows for the recording
        f = entry.iloc[0].audio_file
        x, _ = librosa.load(path=f, sr=self.sr, mono=True, res_type='soxr_vhq')
        x = torch.from_numpy(x)

        # labels must align with the spectrogram time frames
        num_time_frames = math.ceil(x.shape[0] / self.hop_length)
        y = torch.zeros(num_time_frames, dtype=torch.float32)
        if not pd.isna(entry['label']).any():
            on_off_label = list(zip(entry['onset_s'], entry['offset_s'], entry['label']))
            for on, off, l in on_off_label:
                on = math.floor(on * self.sr / self.hop_length)
                off = math.ceil(off * self.sr / self.hop_length)
                y[on:off + 1] = l

        if self.test:
            return x, y

        if x.shape[0] > self.num_samples and self.crop:
            offset = int(random.random() * (x.shape[0] - self.num_samples))
            x = x[offset:offset + self.num_samples]
            offset = int(offset / self.hop_length)  # offset in frames
            num_time_frames = math.ceil(x.shape[0] / self.hop_length)
            y = y[offset:offset + num_time_frames]

        return x, y

    def __len__(self) -> int:
        return self.size

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0.).float()
        y = pad_sequence(y, batch_first=True, padding_value=-1.).long()  # we ignore -1 in loss functions
        return x, y


class DataTransforms(torch.nn.Module):

    def __init__(self,
                 sr: int = 32000,
                 n_fft: int = 512,
                 hop_length: int = 64,
                 top_db: Union[int, float] = 80,
                 use_bernoulli_noise: bool = False,
                 use_random_gain: bool = False,
                 use_color_noise: bool = False,
                 color_noise_chance: float = 0.2,
                 bernoulli_noise_chance: float = 0.2,
                 use_tf_mask: bool = False,
                 freq_mask_param: int = 8,
                 time_mask_param: int = 16,
                 tf_mask_repeats: int = 1) -> None:

        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
        self.db = AmplitudeToDB('power', top_db=top_db)
        self.use_bernoulli_noise = use_bernoulli_noise
        self.use_random_gain = use_random_gain
        self.use_color_noise = use_color_noise
        self.color_noise_chance = color_noise_chance
        self.bernoulli_noise_chance = bernoulli_noise_chance
        self.use_tf_mask = use_tf_mask
        self.tf_mask_repeats = tf_mask_repeats
        self.freq_masking = FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_masking = TimeMasking(time_mask_param=time_mask_param, iid_masks=True)

    def forward(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        x: batch of waveforms (batch_size, time)
        """

        if augment and self.use_color_noise:
            x = apply_color_noise(x, p_apply=self.color_noise_chance)

        x = self.spec(x)[:, 1:]  # -> B, F, T, drop DC
        x = self.db(x.unsqueeze(1)).squeeze(1)  # add a channel so DB conversion is point-wise, not batch-wise
        x = minmax_normalize(x)  # -> B, F, T  float for all the rest

        if augment and self.use_random_gain:
            x = apply_random_gain(x)

        if augment and self.use_bernoulli_noise:
            x = apply_bernoulli_noise(x, p_apply=self.bernoulli_noise_chance)

        if augment and self.use_tf_mask:
            for _ in range(self.tf_mask_repeats):
                x = self.time_masking(x)
                x = self.freq_masking(x)

        return x.transpose(1, 2)  # B, T, F


# Upstream SSL Data Modules
class SSLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sample_dur: Union[int, float] = 3, sr: int = 32000) -> None:
        super().__init__()
        self.df = df
        self.sr = sr
        self.sample_dur = sample_dur
        self.num_samples = int(sample_dur * sr)
        self.size = self.df.audio_file.unique().shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        entry = self.df.iloc[idx]
        f, dur = entry[['audio_file', 'audio_duration']]
        offset = max(0, random.uniform(0, dur - self.sample_dur))
        x, _ = librosa.load(path=f, sr=self.sr, offset=offset, duration=self.sample_dur, mono=True, res_type='soxr_vhq')
        x = torch.from_numpy(x)
        x = nn.functional.pad(x, (0, self.num_samples - x.shape[0]))
        return x

    def __len__(self) -> int:
        return self.size

    def collate_fn(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(x).float()


class MAEDataTransforms(torch.nn.Module):

    def __init__(self,
                 sr: int = 32000,
                 n_fft: int = 512,
                 hop_length: int = 64,
                 top_db: Union[int, float] = 80,
                 use_random_gain: bool = True,
                 use_color_noise: bool = True,
                 color_noise_chance: float = .9) -> None:

        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
        self.db = AmplitudeToDB('power', top_db=top_db)
        self.use_random_gain = use_random_gain
        self.use_color_noise = use_color_noise
        self.color_noise_chance = color_noise_chance

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: batch of waveforms (batch_size, time)
        """

        # target
        y = self.spec(x)[:, 1:]  # B, F, T; we drop dc
        y = self.db(y.unsqueeze(1)).squeeze(1)
        y = minmax_normalize(y).transpose(1, 2)  # B, T, F

        # input
        if self.use_color_noise:
            x = apply_color_noise(x, self.color_noise_chance)

        x = self.spec(x)[:, 1:]  # -> B, F, T
        x = self.db(x.unsqueeze(1)).squeeze(1)
        x = minmax_normalize(x)

        if self.use_random_gain:
            x = apply_random_gain(x)

        x = self.mask(x)

        x = x.transpose(1, 2)  # B, T, F

        return x, y

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        b, f, t = x.shape
        n = t // 200
        for i in range(n):
            s = i * 200 + random.randint(0, 100)
            e = s + random.randint(50, 200)
            x[:, :, s:e] = 0.
        return x


class OSCDataTransforms(torch.nn.Module):

    def __init__(self,
                 sr: int = 32000,
                 n_fft: int = 512,
                 hop_length: int = 64,
                 top_db: Union[int, float] = 80,
                 use_random_gain: bool = True,
                 use_color_noise: bool = True,
                 color_noise_chance: float = 1.0,
                 use_bernoulli_noise: bool = True,
                 bernoulli_noise_chance: float = 1.0) -> None:

        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
        self.db = AmplitudeToDB('power', top_db=top_db)
        self.use_random_gain = use_random_gain
        self.use_bernoulli_noise = use_bernoulli_noise
        self.use_color_noise = use_color_noise
        self.color_noise_chance = color_noise_chance
        self.bernoulli_noise_chance = bernoulli_noise_chance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch of waveforms (batch_size, time)
        """

        if self.use_color_noise:
            x = apply_color_noise(x, p_apply=self.color_noise_chance)

        x = self.spec(x)[:, 1:]  # -> B, F, T, drop DC
        x = self.db(x.unsqueeze(1)).squeeze(1)  # add a channel so DB conversion is point-wise, not batch-wise
        x = minmax_normalize(x)  # -> B, F, T  float for all the rest

        if self.use_random_gain:
            x = apply_random_gain(x)

        if self.use_bernoulli_noise:
            x = apply_bernoulli_noise(x, p_apply=self.bernoulli_noise_chance, min_keep_prob=0.7)

        return x.transpose(1, 2)  # B, T, F
