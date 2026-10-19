import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import BalancedSampler, encode_batch, load_prefix_index, load_trajectories
from dream import solve_dream, temporary_perturbation
from runtime import BACKBONES, STEPS, configure_numerics, file_digest, load_auditor, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a Dream2Discern runtime auditor")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--benchmark", choices=STEPS, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--stage", choices=("warmup", "train"), required=True)
    parser.add_argument("--warm-start")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.stage == "warmup" and (args.seed != 0 or args.warm_start):
        parser.error("The shared warm start uses seed 0 and no earlier checkpoint")
    if args.stage == "train" and not args.warm_start:
        parser.error("The Dream/Discern stage requires --warm-start")
    steps = args.steps if args.steps is not None else (100 if args.stage == "warmup" else STEPS[args.benchmark])
    if steps < 1:
        parser.error("--steps must be positive")
    if Path(args.output).exists():
        parser.error("The output checkpoint already exists; select a new path")
    configure_numerics(args.seed)
    records = load_trajectories(args.trajectories)
    index = load_prefix_index(args.train_index, records)
    sampler = BalancedSampler(index)
    config = dict(BACKBONES[args.backbone], backbone=args.backbone, benchmark=args.benchmark,
                  precision=args.precision, max_length=2048, head_keep=256, serializer="chat",
                  lora_rank=8, lora_alpha=16, perturbation_rank=8, batch_size=32,
                  microbatch_size=8, learning_rate=1e-4, weight_decay=0.0, gradient_clip=1.0,
                  seed=args.seed, stage=args.stage, steps=steps, train_index_sha256=file_digest(args.train_index))
    auditor, tokenizer = load_auditor(config, args.device, args.model_path)
    rng = np.random.default_rng(args.seed)
    if args.warm_start:
        warm = read_checkpoint(args.warm_start)
        if warm["config"]["stage"] != "warmup" or warm["config"]["seed"] != 0:
            raise ValueError("The initialization must be a seed-0 warm-start checkpoint")
        for key in ("backbone", "benchmark", "precision", "max_length", "train_index_sha256"):
            if warm["config"][key] != config[key]:
                raise ValueError(f"Warm-start configuration mismatch: {key}")
        auditor.load_trainable_state_dict(warm["state"])
        config["warm_start_steps"] = warm["config"]["steps"]
        config["warm_start_sha256"] = file_digest(args.warm_start)
    head = list(auditor.head.parameters())
    adapters = [parameter for name, parameter in auditor.named_parameters()
                if parameter.requires_grad and not name.startswith("head.")]
    trainable = head + adapters
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": 1e-4}, {"params": adapters, "lr": 1e-4}
    ], weight_decay=0.0)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rows = sampler.sample(rng, 32)
        microbatches = [encode_batch(tokenizer, records, rows[offset:offset + 8], args.device)
                        for offset in range(0, 32, 8)]
        perturbation, statistics = (None, {})
        if args.stage == "train":
            perturbation, statistics = solve_dream(auditor, microbatches, config["gamma"], rank=8)
        measured_loss = 0.0
        with temporary_perturbation(auditor, perturbation):
            for batch in microbatches:
                logits = auditor(batch["input_ids"], batch["attention_mask"])
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="sum") / 32
                if not torch.isfinite(loss):
                    raise FloatingPointError("The Discern loss is nonfinite")
                loss.backward()
                measured_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            print(json.dumps(dict(step=step + 1, loss=measured_loss, **statistics), allow_nan=False), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "dream2discern-v1", "config": config,
                "state": auditor.trainable_state_dict()}, destination)


if __name__ == "__main__":
    main()
