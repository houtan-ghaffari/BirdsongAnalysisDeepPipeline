
# A Deep Learning Pipeline for Fine-Grained Birdsong Analysis

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2511.12158-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2511.12158&#41;)

This repository contains the official PyTorch code for the paper: **[Data-Efficient Self-Supervised Algorithms for Fine-Grained Birdsong Analysis](https://arxiv.org/abs/2511.12158)**, to be published in Ecological Informatics. 

It provides a robust, three-stage training pipeline for developing reliable deep birdsong syllable detectors with minimal annotation labor.

---

## Installation

This codebase was developed using **Python 3.11**, **PyTorch 2.8**, and **TorchAudio 2.8**. It is expected to work seamlessly with more recent versions of Python and PyTorch.

1. I highly recommend creating an isolated Conda environment, for example:
```bash
   conda update -n base conda
   conda create -n birdsong_ssl python=3.11
   conda activate birdsong_ssl
```

2. Install PyTorch and TorchAudio according to your system specifications from the [official PyTorch website](https://pytorch.org/). For example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. Install the `librosa` package within your environment (use pip):
```bash
   pip install librosa
```

---

## Data Preparation

### 1. Download Public Datasets

This project evaluates methods using two publicly available datasets. Please credit the original authors of these datasets separately:

* **Canary Dataset:** Available on Dryad ([doi:10.5061/dryad.xgxd254f4](https://doi.org/10.5061/dryad.xgxd254f4)).
* **Bengalese Finch Dataset:** Available on Figshare ([doi:10.6084/m9.figshare.4805749](https://www.google.com/search?q=https://https://doi.org/10.6084/m9.figshare.4805749)). *Note: Only retain the `.zip` files containing `.wav` formats.*

### 2. Directory Structure & Cleanup

Assume your preferred dataset path is `root_dir=/home/Datasets`. Unzip the Canary dataset into `/home/Datasets/canaries` and the Bengalese finch `.wav` files into `/home/Datasets/bengalese_finch`.

Organize the audio files and their corresponding `.csv` metadata annotations into directories named after the specific bird. For example, the Bengalese finch dataset includes four birds (`bl26lb16`, etc.). The `curate_df.py` script expects a structure similar to this:

```text
/home/Datasets/
├── canaries/
│   ├── llb3_data/
│   │   ├── llb3_annot.csv
│   │   └── llb3_songs/
│   │       ├── *.wav
│   │       └── ...
│   └── (other canaries)
└── bengalese_finch/
    ├── bl26lb16/
    │   ├── 041912
    │   │   ├── bl26lb16_190412_0721.20144.wav
    │   │   ├── bl26lb16_190412_0721.20144.wav.csv
    │   │   └── ...
    │   ├── 042012
    │   │   └── ...
    │   └── 042112
    │       └── ...
    └── (other finches)
```

### 3. Curation and Data Splitting

Open `curate_df.py` and update the `root_dir` variable to match your local path. You can also adjust the dataset split ratio for each species (e.g., `train_fraction=0.05` and `val_fraction=0.1`).

**Note:** The script first separates a minimal subset of recordings for the train, test, and validation (if val_fraction > 0) splits to ensure at least one example of *every* syllable type is present in all sets.

Run the curation script:

```bash
python curate_df.py
```

This generates a curated metadata CSV file for each species inside their respective dataset directories (`curated_canary_df.csv` and `curated_bengalese_finch_df.csv`).

---

## Training Pipeline

### Stage 1: Self-Supervised Pretraining

You can pretrain a model on a single species or across all species together using Masked Autoencoders (MAE) or Online Syllable Clustering (OSC).

* **Algorithms:** `ssl_mae`, `ssl_osc` 
* **Species:** `bengalese_finch`, `canary`, `mixed` *(uses both datasets)*

**Basic Pretraining (MAE):**

```bash
python -c configs/ssl_config.yaml --task ssl_mae --species bengalese_finch
```

**Pretraining with OSC (Requires Augmentation):**

```bash
python -c configs/ssl_config.yaml --task ssl_osc --species bengalese_finch --augment
```

*(Alternatively, configure these options directly in `configs/ssl_config.yaml`).* The pretrained model weights will be saved in an automatically created logs directory inside your project directory.

### Stage 2: Supervised Finetuning

To finetune the syllable detection model for a specific bird, provide the path to your pretrained SSL state. **Ensure the pretrained model matches the species you are finetuning on** (unless you used `--species mixed`, which generalizes to both).

```bash
python -c configs/supervised_config.yaml \
  --species bengalese_finch \
  --bird bl26lb16 \
  --pretrained_path /path/to/logs/SSL/step(10000)_MAE_state_bengalese_finch_{time_stamp}.pt
```

*Omit the `--pretrained_path` argument to train a model from scratch with random initialization.*

### Stage 3: Semi-Supervised Tuning (Optional)

You should already have a very srong model after Stage 2. However, if you want to squeeze maximum performance out of your dataset, you can apply semi-supervised tuning.

**A word of caution:** While semi-supervised learning is an established topic in machine learning literature, it can be notoriously finicky in practice. I strongly recommend using a validation set to ensure the model is actually improving. A robust default configuration is provided, but you may need to tune hyperparameters and augmentations to unlock meaningful gains for your specific dataset.

Provide the supervised finetuned state from Stage 2:

```bash
python -c configs/semisl_config.yaml \
  --species bengalese_finch \
  --bird bl26lb16 \
  --pretrained_path /path/to/logs/Supervised/mae_supervised..._{time_stamp}.pt
```

---

## How to Use This Code for Your Own Dataset

If you are planning to annotate a personal, unlabelled dataset, follow these steps to dramatically reduce your manual workload:

1. **Prepare Data Paths:** Create a CSV file containing the file paths to all your raw recordings. Modify `data/factory.py` so it can read your custom CSV format.
2. **Pretrain:** I recommend using the **MAE algorithm** (`ssl_mae`), as it is simpler and faster. The number of species or birds you have is not a bottleneck, in fact, the more diverse your unlabeled data, the stronger the pretrained model will be. You can even merge your private dataset with the public datasets provided above for pretraining.
3. **Smart Annotation Sampling:** Use your newly pretrained model to extract features from syllables in your unlabeled songs. By clustering these features, you can intelligently sample recordings from distinct clusters. Labeling a few songs from each distinct cluster is far superior to random sampling, ensuring you capture rare syllable patterns quickly.
4. **Finetune & Automate:** With just a few minutes of manually labeled songs, run the Stage 2 Finetuning script. You can then use this finetuned model to automatically label the remainder of your massive unlabelled dataset!

*Note: A Jupyter Notebook template for temporal syllable segmentation and feature extraction will be provided soon to demonstrate this workflow.*

---

## Citation

If this pipeline or our results were helpful to your research, kindly consider citing our paper:

```bibtex
@misc{ghaffari2025data,
      title={Data-Efficient Self-Supervised Algorithms for Fine-Grained Birdsong Analysis}, 
      author={Houtan Ghaffari and Lukas Rauch and Paul Devos},
      year={2025},
      eprint={2511.12158},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2511.12158](https://arxiv.org/abs/2511.12158)}
}
```
