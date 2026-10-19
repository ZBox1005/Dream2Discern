<div align="center">

<img src="assets/logo_full.png" width="620" alt="Dream2Discern logo">

### Learning from Hard Latent Variants for Runtime Failure Prediction in Multi-Agent Systems

[![arXiv](https://img.shields.io/badge/arXiv-2610.16071-b31b1b)](https://arxiv.org/abs/2610.16071)
[![Project Page](https://img.shields.io/badge/Project-Page-269E95)](https://zbox1005.github.io/dream2discern-project/)
[![AFTraj-2K](https://img.shields.io/badge/Dataset-AFTraj--2K-8864CF)](https://huggingface.co/datasets/ZBox008003/AFTraj)
[![StepShield](https://img.shields.io/badge/Dataset-StepShield-8864CF)](https://huggingface.co/datasets/glo26/stepshield)
[![License](https://img.shields.io/badge/License-MIT-269E95)](LICENSE)

</div>

---

## Overview

Dream2Discern trains an auditor to predict whether a multi-agent run will fail from its observed trajectory. During training, **Dream** searches for difficult latent variants and **Discern** learns from them. At runtime, the auditor scores the observed prefix without this search.

This repository contains the training, calibration, and inference implementation.

## Setup

Use Python 3.13, a CUDA GPU, and the pinned dependencies:

```bash
pip install -r requirements.txt
```

Backbone weights are downloaded from Hugging Face unless `--model-path` points to a local copy. Llama models may require access approval from their publisher.

## Data format

Download [AFTraj-2K](https://huggingface.co/datasets/ZBox008003/AFTraj) or [StepShield](https://huggingface.co/datasets/glo26/stepshield) from their original dataset pages ([StepShield format notes](https://github.com/glo26/stepshield)). The scripts expect a normalized trajectory JSONL file and a training-prefix index JSONL file. For example:

```json
{"trajectory_id":"run-001","parent_group_id":"task-001","public_context":"Task instructions","outcome":1,"events":[{"actor_role":"assistant","actor_id":"agent-1","event_type":"message","action_text":"First action","observation_text":"","done":false},{"actor_role":"tool","actor_id":"tool-1","event_type":"tool_result","action_text":"","observation_text":"Result","done":true}]}
```

```json
{"trajectory_id":"run-001","prefix_t":1}
```

`outcome` is `0` for success and `1` for failure. `parent_group_id` and `outcome` are required for training; runtime prediction only needs the observed events. `prefix_t` is the number of observed events, starting at 1. Supported `event_type` values are `message`, `tool_call`, `tool_result`, `observation`, and `control`.

**Reproduction scope:** This repository does not yet include dataset conversion scripts, the exact benchmark splits/subset lists, or trained checkpoints. The commands below run once the JSONL inputs have been prepared; they do not by themselves reproduce the paper's reported numbers.

## Train

Run the supervised warm start, then Dream/Discern training:

```bash
python train.py --stage warmup --benchmark aftraj --backbone qwen3-4b \
  --trajectories data/train.jsonl --train-index data/train_index.jsonl --output runs/warm.pt
python train.py --stage train --benchmark aftraj --backbone qwen3-4b --seed 0 \
  --warm-start runs/warm.pt --trajectories data/train.jsonl \
  --train-index data/train_index.jsonl --output runs/auditor.pt
```

Backbones: `qwen3-4b`, `qwen3-8b`, `llama3.2-3b`, `llama3.1-8b`. Set `--benchmark stepshield` for StepShield. The default training budgets are in `runtime.py`; `--steps` overrides them.

## Calibrate and predict

Calibrate thresholds from successful training trajectories, then evaluate or predict:

```bash
python inference.py --mode calibrate --checkpoint runs/auditor.pt \
  --trajectories data/train.jsonl --train-index data/train_index.jsonl \
  --output runs/thresholds.json
python inference.py --mode evaluate --checkpoint runs/auditor.pt \
  --thresholds runs/thresholds.json --trajectories data/test.jsonl \
  --output runs/metrics.json
python inference.py --mode predict --checkpoint runs/auditor.pt \
  --thresholds runs/thresholds.json --trajectories data/observed.jsonl \
  --output runs/predictions.jsonl
```

`evaluate` reports AUROC, FPR95, wACC, and TWA. `predict` outputs the failure score, calibrated threshold, and binary prediction for each trajectory's latest observed event.

## Files

| File | Purpose |
| --- | --- |
| `train.py`, `dream.py` | Warm start and Dream/Discern optimization |
| `model.py`, `runtime.py` | Auditor architecture and backbone configuration |
| `data.py` | JSONL validation, prefix serialization, and sampling |
| `inference.py`, `calibration.py`, `metrics.py` | Runtime scoring, calibration, and evaluation |

## Citation

```bibtex
@article{zhang2026dream2discern,
  title={Dream2Discern: Learning from Hard Latent Variants for Runtime Failure Prediction in Multi-Agent Systems},
  author={Zhang, Boxuan and Zhu, Jianing and Han, Xiaotian and Tang, Ruixiang},
  journal={arXiv preprint arXiv:2610.16071},
  year={2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The linked datasets and backbone models retain their own licenses and access terms.
