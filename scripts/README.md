# Script Guide

This directory contains the command-line entry points for training models,
compressing them with MPO layers, and running DeepLOB-specific compression
experiments.

For a first hands-on pass, use the generic pipeline:

```bash
python scripts/train.py --model mlp --smoke
python scripts/compress.py --ckpt checkpoints/mlp.pt --smoke
```

That uses synthetic FI-2010-shaped data, so it checks that the code runs but
does not measure useful accuracy.

Weights & Biases logging is optional. Install and authenticate it once:

```bash
pip install wandb
wandb login
```

Then add `--wandb` to either generic script:

```bash
python scripts/train.py --model mlp --device mps --epochs 50 --wandb
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --wandb
```

Use `--wandb-mode offline` to log locally without syncing immediately.

## Important Modules

The scripts are thin wrappers around the `deeplob_mpo` package:

| module | role |
| --- | --- |
| `deeplob_mpo.config` | `Config` dataclass for data, model, training, and MPO options. |
| `deeplob_mpo.data` | FI-2010 loading, sliding-window construction, `DataLoader` creation, and smoke loaders. |
| `deeplob_mpo.models` | Model definitions: `DeepLOBNet`, `MLPNet`, `TransformerNet`, and `LinearLSTM`. |
| `deeplob_mpo.mpo` | MPO layers: `MPOLinear`, `MPOConv2d`, factorization helpers, and TT-SVD warm start. |
| `deeplob_mpo.compress` | Generic post-hoc replacement of eligible `nn.Linear` layers with `MPOLinear`. |
| `deeplob_mpo.engine` | Shared `train_one_epoch`, `evaluate`, and `fit` loops. |

The generic training and compression flow uses only plain PyTorch modules plus
`MPOLinear`. The DeepLOB experiment scripts also use `MPOConv2d` and
`LinearLSTM` to expose convolution and LSTM matrices for compression.

## Full Generic Pipeline

The intended reusable workflow is:

1. Extract FI-2010 data.
2. Train a dense baseline model.
3. Save the dense checkpoint and its config JSON.
4. Load the checkpoint.
5. Replace large `nn.Linear` layers with `MPOLinear`.
6. Evaluate the compressed model before fine-tuning.
7. Fine-tune the compressed model.
8. Compare dense vs compressed parameter count and accuracy.

### 1. Extract the Dataset

The data archive is committed at `data/data.zip`. The loader expects the
extracted FI-2010 `.txt` files inside `jupyter_pytorch` by default:

```bash
unzip -n data/data.zip -d jupyter_pytorch
```

If you extract the files somewhere else, pass that directory with `--data-dir`.

### 2. Train a Dense Baseline

Use `scripts/train.py`:

```bash
python scripts/train.py --model mlp --device mps --epochs 50
```

Common variants:

```bash
python scripts/train.py --model transformer --device cuda --epochs 50 --d-model 128
python scripts/train.py --model deeplob --device mps --epochs 50 --head-hidden 512
python scripts/train.py --model mlp --data-dir /path/to/FI2010 --device cpu
```

By default this writes:

```text
checkpoints/<model>.pt
checkpoints/<model>_config.json
```

The config JSON matters: `scripts/compress.py` uses it to rebuild the same
model architecture before loading the checkpoint.

Important options:

| option | meaning |
| --- | --- |
| `--model {mlp,transformer,deeplob}` | Selects the dense baseline architecture. |
| `--epochs` | Number of dense training epochs. |
| `--device` | PyTorch device, commonly `cpu`, `cuda`, or `mps`. |
| `--batch-size` | Batch size override. |
| `--lr` | Adam learning rate. |
| `--weight-decay` | Adam L2 regularization term. |
| `--seed` | Random seed for reproducibility. |
| `--horizon` | FI-2010 prediction horizon index, `0..4`. |
| `--window` | Sliding-window length, default `100`. |
| `--data-dir` | Directory holding the extracted FI-2010 `.txt` files, default `jupyter_pytorch`. |
| `--out` | Custom checkpoint path, default `checkpoints/<model>.pt`. |
| `--log-every` | Print training progress every N epochs, default `1`. |
| `--smoke` | Uses tiny random data and two epochs for a fast pipeline check. |
| `--wandb` | Logs run config, train/validation/test metrics, and checkpoint files to W&B. |
| `--wandb-project` | W&B project name. Default `deeplob-mpo`. |
| `--wandb-entity` | Optional W&B team/user entity. |
| `--wandb-run-name` | Optional human-readable run name. |
| `--wandb-mode {online,offline,disabled}` | W&B mode when `--wandb` is enabled. |
| `--wandb-tags` | Optional comma-separated W&B tags. |

Model-specific options (only affect the matching `--model`, all default to the
`Config` values in `deeplob_mpo/config.py` when omitted):

| option | applies to | meaning |
| --- | --- | --- |
| `--head-hidden` | `deeplob` | Hidden-layer width of the FCN classification head. Default `512`. |
| `--head-depth` | `deeplob` | Number of hidden `Linear` layers in the FCN head. Default `2`. |
| `--dropout` | `deeplob` | Dropout rate in the FCN head. Default `0.1`. |
| `--mlp-hidden` | `mlp` | Width of the stem and each residual block. Default `1024`. |
| `--mlp-blocks` | `mlp` | Number of residual blocks. Default `4`. |
| `--d-model` | `transformer` | Transformer embedding dimension. Default `128`. |
| `--n-heads` | `transformer` | Number of attention heads. Default `4`. |
| `--tf-depth` | `transformer` | Number of encoder layers. Default `2`. |
| `--ff-mult` | `transformer` | Feed-forward hidden size as a multiple of `--d-model`. Default `4`. |

### 3. Inspect Compressible Layers

Before compressing, list the model's linear layers:

```bash
python scripts/compress.py --ckpt checkpoints/mlp.pt --list-layers
```

The generic compressor only compresses `nn.Linear` layers where
`min(in_features, out_features) >= --min-dim`.

### 4. Compress and Fine-Tune

Use `scripts/compress.py`:

```bash
python scripts/compress.py \
  --ckpt checkpoints/mlp.pt \
  --bond-dim 16 \
  --device mps \
  --finetune-epochs 10 \
  --finetune-lr 5e-5 \
  --out checkpoints/mlp_mpo_bond16.pt
```

What this does:

1. Loads `checkpoints/mlp_config.json`.
2. Rebuilds the dense `mlp` model.
3. Loads `checkpoints/mlp.pt`.
4. Evaluates dense baseline test accuracy.
5. Replaces eligible `nn.Linear` layers with `MPOLinear`.
6. Initializes each `MPOLinear` from the dense weight using TT-SVD.
7. Reports dense params, MPO params, ratio, and reconstruction error per layer.
8. Evaluates the compressed model before fine-tuning.
9. Fine-tunes the compressed model with Adam and cross-entropy.
10. Saves the best fine-tuned compressed state dict to `--out`.

Important options:

| option | meaning |
| --- | --- |
| `--ckpt` | Dense checkpoint to load. The matching config is expected next to it as `<name>_config.json`. |
| `--bond-dim` | Main compression knob. Lower is smaller but riskier; higher is larger but usually more accurate. |
| `--n-cores` | Number of MPO cores used when factorizing a matrix. |
| `--min-dim` | Skip layers whose smaller matrix dimension is below this value. |
| `--only` | Comma-separated layer names to compress, for example `stem` or `blocks.0.fc,blocks.1.fc`. |
| `--no-warm-start` | Randomly initializes MPO cores instead of TT-SVD from dense weights. Mostly useful for experiments. |
| `--finetune-epochs` | Number of fine-tuning epochs after compression. Use `0` to skip fine-tuning. |
| `--finetune-lr` | Learning rate for fine-tuning. Default is smaller than normal training. |
| `--batch-size` | Batch size override for compression/fine-tuning. |
| `--out` | Output path for the fine-tuned compressed checkpoint. |
| `--smoke` | Uses synthetic data for a fast end-to-end check. |
| `--wandb` | Logs run config, baseline/compressed/fine-tuned metrics, parameter counts, per-layer compression stats, and checkpoint files to W&B. |
| `--wandb-project` | W&B project name. Default `deeplob-mpo`. |
| `--wandb-entity` | Optional W&B team/user entity. |
| `--wandb-run-name` | Optional human-readable run name. |
| `--wandb-mode {online,offline,disabled}` | W&B mode when `--wandb` is enabled. |
| `--wandb-tags` | Optional comma-separated W&B tags. |

Logged classification metrics include loss, accuracy, macro/weighted
precision, macro/weighted recall, macro/weighted F1, balanced accuracy, and R2
computed from predicted class labels.

### 5. Sweep Bond Dimensions Manually

Run the same checkpoint at several compression strengths:

```bash
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 4  --device mps --out checkpoints/mlp_mpo_bond4.pt
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 8  --device mps --out checkpoints/mlp_mpo_bond8.pt
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --device mps --out checkpoints/mlp_mpo_bond16.pt
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 32 --device mps --out checkpoints/mlp_mpo_bond32.pt
```

Read the printed report as:

| column | meaning |
| --- | --- |
| `dense` | Original parameter count for that layer. |
| `mpo` | Parameter count after MPO replacement. |
| `ratio` | `mpo / dense`; lower means more compression. |
| `TT-SVD err` | Relative reconstruction error immediately after approximating the dense weight. |

## What Fine-Tuning Means Here

Fine-tuning is normal supervised training after compression. The model already
contains `MPOLinear` layers, so Adam updates the MPO cores directly.

By default, all trainable parameters are updated, not only the MPO cores. That
lets the rest of the model adapt to the compression error.

The shared implementation is `deeplob_mpo.engine.fit`:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

## Generic Scripts

### `train.py`

Trains a dense baseline model and writes a checkpoint plus config JSON.

Examples:

```bash
python scripts/train.py --model mlp --smoke
python scripts/train.py --model mlp --device mps --epochs 50
python scripts/train.py --model transformer --device cuda --epochs 50 --d-model 128 --tf-depth 2
python scripts/train.py --model deeplob --device mps --epochs 50 --head-depth 2 --head-hidden 512
```

Use this for the clean, reusable baseline-training path.

### `compress.py`

Compresses eligible `nn.Linear` layers in a trained baseline, evaluates the
compressed model, then fine-tunes it.

Examples:

```bash
python scripts/compress.py --ckpt checkpoints/mlp.pt --list-layers
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --device mps
python scripts/compress.py --ckpt checkpoints/transformer.pt --bond-dim 8 --finetune-epochs 20
python scripts/compress.py --ckpt checkpoints/mlp.pt --only stem --bond-dim 8
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --finetune-epochs 0
```

Use this for the clean, reusable post-hoc MPO compression path.

## DeepLOB-Specific Experiment Scripts

These scripts operate on the original saved PyTorch DeepLOB model at
`jupyter_pytorch/best_val_model_pytorch` by default. They are useful for
experiments on the inherited notebook checkpoint, but they are less general
than `train.py` and `compress.py`.

### `compress_deeplob_lstm.py`

Compresses one LSTM matrix from the original DeepLOB checkpoint.

The original model uses PyTorch `nn.LSTM`, whose input and recurrent matrices
are packed into internal parameters. This script first swaps it for
`LinearLSTM`, which exposes:

| matrix | meaning |
| --- | --- |
| `ih` | input-to-hidden matrix, shape `256 x 192`. |
| `hh` | hidden-to-hidden matrix, shape `256 x 64`. |

Examples:

```bash
python scripts/compress_deeplob_lstm.py --baseline-only --device mps
python scripts/compress_deeplob_lstm.py --matrix ih --bond-dim 8 --finetune-epochs 5 --device mps
python scripts/compress_deeplob_lstm.py --matrix hh --bond-dim 8 --freeze-backbone --device mps
```

Notes:

- The `LinearLSTM` runs as a Python loop, so it is slower than the original
  cuDNN-backed `nn.LSTM`.
- `--train-subset` and `--eval-subset` exist to keep those runs manageable.
- `--freeze-backbone` trains only the compressed MPO layer.

### `compress_deeplob_conv.py`

Compresses eligible convolution layers from the original DeepLOB checkpoint
using `MPOConv2d`, while keeping the fast built-in `nn.LSTM`.

Example:

```bash
python scripts/compress_deeplob_conv.py --bond 8 --epochs 30 --device mps
```

This is the fastest DeepLOB-specific experiment because the LSTM remains the
optimized PyTorch implementation.

### `compress_deeplob_full.py`

Compresses a selected set of heavy DeepLOB layers together:

- LSTM input matrix `lstm.ih`
- `conv3.0`
- `inp1.3`
- `inp2.3`

Example:

```bash
python scripts/compress_deeplob_full.py --bond 8 --finetune-epochs 3 --device mps
```

Use this when you want a controlled "important layers together" experiment
without compressing every small layer.

### `compress_deeplob_all.py`

Attempts to compress every eligible `Conv2d` and exposed LSTM `Linear` matrix
in the original DeepLOB checkpoint. It skips layers where the MPO replacement
would have at least as many parameters as the dense layer.

Example:

```bash
python scripts/compress_deeplob_all.py --bond 8 --finetune-epochs 3 --device mps
```

Use this for an aggressive whole-model compression experiment.

### `sweep_lstm_bond.py`

Sweeps bond dimensions for the original DeepLOB LSTM input-to-hidden matrix.

Example:

```bash
python scripts/sweep_lstm_bond.py --bonds 2 4 6 8 10 --device mps
```

It prints a table comparing LSTM MPO parameter count, total model parameter
count, no-retrain accuracy, and retrained accuracy for each bond.

### `train_deeplob_mpo_scratch.py`

Trains DeepLOB from scratch with a randomly initialized MPO LSTM input matrix.
This is different from post-hoc compression: there is no TT-SVD initialization
from a trained dense matrix.

Example:

```bash
python scripts/train_deeplob_mpo_scratch.py --bond 8 --epochs 25 --device mps
```

Use this to compare paper-style "train compressed from zero" against the
post-hoc "train dense, compress, then fine-tune" workflow.

## Recommended First Hands-On Session

Run these in order:

```bash
# 1. Fast code-path check.
python scripts/train.py --model mlp --smoke
python scripts/compress.py --ckpt checkpoints/mlp.pt --smoke

# 2. Prepare real data.
unzip -n data/data.zip -d jupyter_pytorch

# 3. Train one dense model.
python scripts/train.py --model mlp --device mps --epochs 50

# 4. Inspect layers.
python scripts/compress.py --ckpt checkpoints/mlp.pt --list-layers

# 5. Compress and fine-tune.
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --device mps

# 6. Try a smaller and larger bond dimension.
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 8 --device mps --out checkpoints/mlp_mpo_bond8.pt
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 32 --device mps --out checkpoints/mlp_mpo_bond32.pt
```

If you are not on Apple Silicon, replace `--device mps` with `--device cuda`
for an NVIDIA GPU or `--device cpu` for CPU-only runs.

## Troubleshooting

- `FileNotFoundError` for FI-2010 `.txt` files: extract `data/data.zip` into
  `jupyter_pytorch` or pass the correct directory with `--data-dir`.
- Compression says "no eligible Linear layers": lower `--min-dim` or use a
  model with larger linear layers, such as `mlp` or `transformer`.
- Accuracy drops immediately after compression: that is expected for small
  bond dimensions. Increase `--bond-dim` or use more fine-tuning epochs.
- DeepLOB LSTM scripts are slow: reduce `--train-subset` / `--eval-subset`, or
  use `compress_deeplob_conv.py` if you only need a fast original-checkpoint
  experiment.
- `torch.load` errors on the original checkpoint: run from the repository root.
  The DeepLOB-specific scripts set up imports so the notebook model class can
  be unpickled.
