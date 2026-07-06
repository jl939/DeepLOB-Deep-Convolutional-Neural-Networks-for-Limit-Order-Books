# deeplob_mpo

MPO (matrix product operator) compression of a trained DeepLOB model on FI-2010,
following Gao et al., *Compressing deep neural networks by matrix product
operators*, Phys. Rev. Research **2**, 023300 (2020).

## Why architecture matters for MPO

MPO compresses **weight matrices with exploitable low-rank structure**, and it
only saves parameters when the matrix is large enough that the factored cores
cost less than the dense matrix. `compress_model` targets `nn.Linear` and
(optionally) `nn.Conv2d` kernels; the fused `nn.LSTM` gates are reached by first
rewriting the LSTM as a `LinearLSTM` (its `ih`/`hh` become plain `nn.Linear`).

Pick with `--model {deeplob,mlp,transformer}`.

| model | MPO targets | character |
|-------|-------------|-----------|
| `deeplob` | CNN+Inception+LSTM backbone (+ optional wide FCN head) | strong predictor; compress the LSTM gates + inception convs |
| `mlp`     | deep residual MLP: stem `Linear(T*40 → hidden)` + every block `Linear(hidden²)` | many large matrices → most dramatic compression (~60×) |
| `transformer` | every attention q/k/v/o + FFN in each block | uniformly compressible; the published MPO-on-transformers regime |

```
deeplob:      (B,1,100,40) ─ Conv→Inception→LSTM ─(B,H)─ [Linear head] ─ Linear(H→3)
mlp:          (B,1,100,40) ─ flatten(4000) ─ stem(4000→1024) ─ [ResBlock(1024²)]×N ─ Linear(1024→3)
transformer:  (B,1,100,40) ─ embed(40→d) +pos ─ [q,k,v,o, FFN]×L ─ meanpool ─ Linear(d→3)
```

(`EncoderLayer` uses explicit `Linear` q/k/v/o rather than `nn.MultiheadAttention`
so the attention projections are compressible too, not just the FFN.)

### The safe MPO rule: "replace only if it actually shrinks"

Not every matrix is a good MPO target — small dims, or dims that are prime /
awkwardly factored (e.g. `in=2`, `out=3`), make the MPO *bigger* than the dense
layer. `compress_model` therefore applies two gates before replacing a layer:

- `min(in, out) >= --min-dim` (cheap size pre-filter), **and**
- `mpo_params < --max-ratio * dense_params` (the ratio guard).

So you can safely say "compress everything" (`--compress-conv --compress-lstm`)
and the poor candidates are dropped automatically. Use `--list-layers` to see
exactly what would be replaced (and its reconstruction error) before committing.

## DeepLOB: which layers can be compressed, and how to tune them

The backbone's **kernel shapes are structural** — `(1,2)/(1,2)/(1,10)` walk the
40-column LOB layout (pair price+size → merge ask/bid → merge all 10 levels) and
`(4,1)/(3,1)/(5,1)` are the temporal windows. They are *fixed*. The **channel
widths are tunable capacity knobs**, and scaling them up is how you enlarge the
MPO targets while staying faithful to the DeepLOB architecture. Defaults
reproduce the paper.

| layer | matrix (out×in), paper widths | good MPO target? | scale its size with |
|-------|------------------------------|------------------|---------------------|
| `conv1.0` | 32×2 | ✗ (in=2 → MPO grows it) | — (structural) |
| `conv1.3/1.6`, `conv2.3/2.6`, `conv3.3/3.6` | 32×128 | ~ only at low bond | `--conv-channels` |
| `conv2.0` | 32×64 | ✗ marginal | `--conv-channels` |
| `conv3.0` | 32×320 | ✓ | `--conv-channels` |
| `inp{1,2,3}.0`, `inp3.1` | 64×32 | ✗ (1×1 projections) | `--conv/inception-channels` |
| `inp1.3` | 64×192 | ✓ | `--inception-channels` |
| `inp2.3` | 64×320 | ✓ | `--inception-channels` |
| `lstm.ih` | 256×192 | ✓✓ (biggest matrix) | `--lstm-hidden`, `--inception-channels` |
| `lstm.hh` | 256×64 | ✓ | `--lstm-hidden` |
| head hidden `Linear(H→512)`, `(512→512)` | up to 512×512 | ✓✓ | `--head-hidden`, `--head-depth` |
| head output `Linear(→3)` | 3×… | ✗ (out=3, task-forced) | — |

- Convs are reached with `--compress-conv`; the LSTM with `--compress-lstm`.
- `lstm.ih` out-dim is `4 × lstm_hidden`, in-dim is `3 × inception_channels`;
  `lstm.hh` is `4·lstm_hidden × lstm_hidden`. So `--lstm-hidden 128` roughly
  doubles the two biggest matrices in the network.
- The head layers only exist with `--head-depth >= 1`. `--head-depth 0` gives the
  paper's bare `Linear(lstm_hidden → 3)` classifier (no extra MPO targets).

**Caveats.** `MPOConv2d` rebuilds its kernel each forward (saves storage, not
compute), and `LinearLSTM` is a Python-loop LSTM (slower than cuDNN). Compressing
the backbone trades inference speed for parameter count.

## Layout

| file | role |
|------|------|
| `config.py`   | `Config` dataclass: data / model / train / mpo hyperparameters |
| `data.py`     | FI-2010 loading, windowing, DataLoaders (+ synthetic smoke loaders) |
| `models.py`   | `MLPNet`, `TransformerNet`, `DeepLOBNet`, `build_model` |
| `mpo.py`      | `MPOLinear` — an `nn.Linear` whose weight is an MPO (+ TT-SVD warm start) |
| `compress.py` | replace trained `Linear` layers with `MPOLinear`, report compression |
| `engine.py`   | `train_one_epoch`, `evaluate`, `fit` |

Scripts: `scripts/train.py` (train any model) and `scripts/compress.py`
(post-hoc MPO compression + fine-tune). Run either with `--help` for all args.

## Workflow: compress an already-trained network

```bash
# 0. one-time: extract the dataset next to the .txt loader
unzip -n data/data.zip -d jupyter_pytorch

# 1. train a baseline (use a GPU; --device cuda/mps). Checkpoint -> checkpoints/<model>.pt
python scripts/train.py --model mlp --device mps --epochs 50

# 2. MPO-compress the trained Linear layers, then fine-tune to recover accuracy
python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 16 --device mps
```

`--bond-dim` is the compression knob: smaller D → fewer parameters, larger
reconstruction error. Sweep it (4, 8, 16, 32) to trace the accuracy/size curve.

## Reproducing / tuning DeepLOB (start here)

```bash
# --- reproduce the paper's DeepLOB (bare Linear(64->3) head, no weight decay) ---
python scripts/train.py --model deeplob --device mps \
  --head-depth 0 --weight-decay 0 --eps 1.0 --lr 0.01 --batch-size 32 \
  --epochs 200 --early-stopping-patience 20 --monitor val_acc

# --- same architecture, more capacity => bigger/better MPO targets ---
#     lstm-hidden is the highest-leverage knob (grows the two biggest matrices).
python scripts/train.py --model deeplob --device mps --head-depth 0 \
  --lstm-hidden 128 --inception-channels 96 --conv-channels 48

# --- see which backbone layers would compress, before committing ---
python scripts/compress.py --ckpt checkpoints/deeplob.pt --device mps \
  --bond-dim 16 --min-dim 32 --compress-conv --compress-lstm --list-layers

# --- compress the LSTM gates + inception convs, then fine-tune ---
python scripts/compress.py --ckpt checkpoints/deeplob.pt --device mps \
  --bond-dim 16 --min-dim 32 --max-ratio 0.9 --compress-conv --compress-lstm
```

**Where to start tuning:**
- Accuracy / faithfulness → `--head-depth` (0 = paper), `--lr`, `--eps`,
  `--weight-decay`, `--epochs`, `--early-stopping-patience`.
- Size of the MPO targets → `--lstm-hidden` (biggest lever), then
  `--inception-channels`, `--conv-channels` (and `--head-hidden/--head-depth`
  if you want a wide FCN head too).
- Compression aggressiveness → `--bond-dim` (sweep 4/8/16/32), gated by
  `--min-dim` and `--max-ratio`; add `--compress-conv` / `--compress-lstm` to
  reach the backbone.

`deeplob` capacity flags (`--conv-channels`, `--inception-channels`,
`--lstm-hidden`) default to the paper's `32 / 64 / 64`. Whatever you pick is
saved in `<ckpt>_config.json`, so `compress.py` rebuilds the matching model
automatically.

### Quick pipeline check (no dataset, no GPU)

```bash
python scripts/train.py --model mlp --smoke
python scripts/compress.py --ckpt checkpoints/mlp.pt --smoke
```

`--smoke` uses tiny random tensors shaped like FI-2010 — it verifies the code
runs, not model quality (accuracy is chance on random data).

## Notes

- Post-hoc TT-SVD alone usually loses accuracy (the trained weights only
  partially live in the low-bond manifold); the **fine-tune** step recovers it.
- The model outputs **logits**; loss is `nn.CrossEntropyLoss`. (The original
  notebook applied `softmax` before cross-entropy, which double-counts the
  softmax — fixed here.)
- Requires: `torch`, `numpy`. (No scikit-learn.)
```
