# GPU run log - train_surrogate.py

**Date:** 2026-08-07
**Platform:** Google Colab, T4 GPU runtime
**torch.cuda.is_available():** True
**Device:** Tesla T4

## What this demonstrates

`train_surrogate.py` is device-agnostic: it auto-selects CUDA when a GPU is
present and runs unchanged on CPU otherwise (every tensor and the model are
moved to the selected `device`). It was run on two backends from the same
source:

- Local CPU (Apple Silicon): trained in ~120 s.
- Colab T4 GPU (this run): trained in ~967 s.

## Honest note on timing

For this surrogate - a 3-layer, 128-wide MLP (~35k weights) - the CPU is
faster. The model is too small for its matrix multiplies to amortise GPU
kernel-launch and host-to-device overhead, so a T4 spends most of its time
idle between tiny batches. The GPU path is the right default for scaling up:
a larger surrogate, the Neural-ODE variant, or batched multi-seed training,
and for running on a cluster GPU node. The point demonstrated here is a
working, portable device-selection workflow, not a speedup on this model.

## nvidia-smi

```
Fri Aug  7 00:54:43 2026
+----------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------+-------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|====================================+=========================+========================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   39C    P8              9W /   70W |       3MiB /  15360MiB |      0%      Default |
+-----------------------------------+-------------------------+-----------------------+
```
