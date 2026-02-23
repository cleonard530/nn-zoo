# Neural Network Exploration Repo

A PyTorch-based repository for studying popular neural network architectures. Models are small enough to run on CPU/single GPU and readable for learning.

## Layout

| Folder      | Description |
|------------|-------------|
| `models/`  | Architecture definitions (MLP, CNN, ResNet, VAE, Transformer, GAN, LSTM, U-Net, etc.) |
| `data/`    | Dataset loading (MNIST, CIFAR-10, Fashion-MNIST, synthetic sequences, MNIST segmentation) |
| `training/`| One training script per model |
| `weights/` | Pretrained checkpoints (train to generate or add your own) |
| `inference/` | Scripts to run inference with saved weights |
| `utils/`   | Shared helpers (device, checkpoint, logging) |

## Architectures and datasets

| Model        | File            | Typical dataset        | Train script           | Inference script        |
|-------------|-----------------|------------------------|------------------------|-------------------------|
| MLP         | `models/mlp.py` | MNIST                  | `training/train_mlp.py`| `inference/run_mlp.py`  |
| CNN         | `models/cnn.py` | MNIST / CIFAR-10       | `training/train_cnn.py`| `inference/run_cnn.py`  |
| Autoencoder | `models/autoencoder.py` | MNIST          | `training/train_autoencoder.py` | `inference/run_autoencoder.py` |
| VAE         | `models/vae.py` | MNIST                  | `training/train_vae.py`| `inference/run_vae.py`  |
| ResNet      | `models/resnet.py` | CIFAR-10            | `training/train_resnet.py` | `inference/run_resnet.py` |
| BERT tiny   | `models/bert_tiny.py` | Synthetic sequence | `training/train_bert_tiny.py` | `inference/run_bert_tiny.py` |
| GPT tiny    | `models/gpt_tiny.py` | Synthetic sequence  | `training/train_gpt_tiny.py` | `inference/run_gpt_tiny.py` |
| GAN         | `models/gan.py` | MNIST                  | `training/train_gan.py`| `inference/run_gan.py`  |
| LSTM        | `models/lstm.py` | Synthetic sequence   | `training/train_lstm.py`| `inference/run_lstm.py` |
| U-Net       | `models/unet.py` | MNIST segmentation    | `training/train_unet.py`| `inference/run_unet.py` |

## Install

```bash
pip install -r requirements.txt
```

## Data

Datasets are downloaded on first use (e.g. when you run a training script). Set a common root with `--data_dir` (default: `./data`). Raw files go under `data/raw/` (MNIST, CIFAR, etc.). Synthetic sequence data for BERT/GPT/LSTM is generated in memory.

## Train

From the repo root, one script per model:

```bash
# Image classification
python training/train_mlp.py --data_dir ./data
python training/train_cnn.py --data_dir ./data
python training/train_cnn.py --data_dir ./data --cifar
python training/train_resnet.py --data_dir ./data

# Reconstruction / generation
python training/train_autoencoder.py --data_dir ./data
python training/train_vae.py --data_dir ./data
python training/train_gan.py --data_dir ./data

# Sequence
python training/train_bert_tiny.py
python training/train_gpt_tiny.py
python training/train_lstm.py --mode classify

# Segmentation
python training/train_unet.py --data_dir ./data
```

Checkpoints are written to `weights/<model_name>/` (e.g. `weights/mlp/best.pt`, `weights/mlp/last.pt`).

## Inference

After training (or with your own checkpoints in `weights/`):

```bash
python inference/run_mlp.py --checkpoint weights/mlp/best.pt
python inference/run_cnn.py --checkpoint weights/cnn/best.pt
python inference/run_autoencoder.py --checkpoint weights/autoencoder/best.pt
python inference/run_vae.py --checkpoint weights/vae/best.pt
python inference/run_resnet.py --checkpoint weights/resnet/best.pt
python inference/run_bert_tiny.py --checkpoint weights/bert_tiny/best.pt
python inference/run_gpt_tiny.py --checkpoint weights/gpt_tiny/best.pt
python inference/run_gan.py --checkpoint weights/gan/best.pt
python inference/run_lstm.py --checkpoint weights/lstm/best.pt
python inference/run_unet.py --checkpoint weights/unet/best.pt
```

## Weights

The `weights/` directory holds pretrained checkpoints. `*.pt` files are gitignored; either train locally or use Git LFS if you want to share a few example checkpoints. See each training script for options (epochs, batch size, lr, etc.).
