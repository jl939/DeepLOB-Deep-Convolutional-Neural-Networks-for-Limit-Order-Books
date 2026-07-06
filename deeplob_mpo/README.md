# deeplob_mpo

MPO (matrix product operator) compression of a trained DeepLOB model on FI-2010,
following Gao et al., *Compressing deep neural networks by matrix product
operators*, Phys. Rev. Research **2**, 023300 (2020).

## Why architecture matters for MPO

MPO compresses **large `nn.Linear` weight matrices** that have exploitable
low-rank structure. The original DeepLOB has almost no FC parameters (`fc1` is
64×3 = 192) and a near-full-rank LSTM, so it is a poor MPO target. The three
models below are all good targets, and — crucially — their compressible layers
are all plain `nn.Linear`, so `compress.compress_model` works on every one
without architecture-specific code. (The LSTM was awkward precisely because its
gate matrices are fused inside `nn.LSTM`.)

Pick with `--model {deeplob,mlp,transformer}`.

| model | MPO targets | character |
|-------|-------------|-----------|
| `deeplob` | wide FCN head on the CNN+LSTM backbone | strong predictor, modest compressible fraction |
| `mlp`     | deep residual MLP: stem `Linear(T*40 → hidden)` + every block `Linear(hidden²)` | many large matrices → most dramatic compression (~150×) |
| `transformer` | every attention q/k/v/o + FFN in each block | uniformly compressible; the published MPO-on-transformers regime |

```
deeplob:      (B,1,100,40) ─ Conv→Inception→LSTM ─(B,64)─ Linear(64→512)→(512→512)→(512→3)
mlp:          (B,1,100,40) ─ flatten(4000) ─ stem(4000→1024) ─ [ResBlock(1024²)]×N ─ Linear(1024→3)
transformer:  (B,1,100,40) ─ embed(40→d) +pos ─ [q,k,v,o, FFN]×L ─ meanpool ─ Linear(d→3)
```

(`EncoderLayer` uses explicit `Linear` q/k/v/o rather than `nn.MultiheadAttention`
so the attention projections are compressible too, not just the FFN.)

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
