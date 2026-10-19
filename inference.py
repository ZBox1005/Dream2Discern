import argparse
import json
import math
from pathlib import Path

import torch

from calibration import fit_thresholds, select_calibration_ids, threshold_at
from data import encode_batch, load_prefix_index, load_trajectories
from runtime import configure_numerics, file_digest, load_auditor, read_checkpoint, write_json, write_jsonl


@torch.no_grad()
def score_prefixes(auditor, tokenizer, records, rows, device, batch_size=8):
    result = []
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset:offset + batch_size]
        batch = encode_batch(tokenizer, records, selected, device)
        scores = auditor(batch["input_ids"], batch["attention_mask"]).sigmoid().float().cpu().tolist()
        for row, score in zip(selected, scores):
            if not math.isfinite(score):
                raise FloatingPointError("A runtime failure score is nonfinite")
            result.append({"trajectory_id": row["trajectory_id"], "prefix_t": row["prefix_t"], "score": score})
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate or run a Dream2Discern auditor")
    parser.add_argument("--mode", choices=("calibrate", "predict", "evaluate"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--train-index")
    parser.add_argument("--prefix-index")
    parser.add_argument("--thresholds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.mode == "calibrate" and not args.train_index:
        parser.error("Calibration requires the training-prefix index")
    if args.mode != "calibrate" and not args.thresholds:
        parser.error("Prediction and evaluation require calibrated thresholds")
    if args.prefix_index and args.mode != "predict":
        parser.error("--prefix-index is used only for prediction")
    configure_numerics(0)
    checkpoint = read_checkpoint(args.checkpoint)
    records = load_trajectories(args.trajectories, require_outcomes=args.mode != "predict")
    auditor, tokenizer = load_auditor(checkpoint["config"], args.device, args.model_path)
    auditor.load_trainable_state_dict(checkpoint["state"])
    if args.mode == "calibrate":
        if file_digest(args.train_index) != checkpoint["config"]["train_index_sha256"]:
            raise ValueError("Calibration must use the checkpoint's training-prefix index")
        index = load_prefix_index(args.train_index, records)
        ids = select_calibration_ids(records, index, seed=0, fraction=0.4)
        rows = [{"trajectory_id": tid, "prefix_t": t} for tid in ids
                for t in range(1, len(records[tid]["events"]))]
        scores = score_prefixes(auditor, tokenizer, records, rows, args.device, args.batch_size)
        band = fit_thresholds(scores, alpha=args.alpha, min_support=20)
        band["checkpoint_sha256"] = file_digest(args.checkpoint)
        write_json(args.output, band)
        return
    band = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    if band["checkpoint_sha256"] != file_digest(args.checkpoint):
        raise ValueError("The thresholds were calibrated for a different checkpoint")
    if args.mode == "evaluate":
        rows = [{"trajectory_id": tid, "prefix_t": t} for tid in sorted(records)
                for t in range(1, len(records[tid]["events"]))]
    elif args.prefix_index:
        rows = load_prefix_index(args.prefix_index, records, require_outcomes=False)
    else:
        rows = [{"trajectory_id": tid, "prefix_t": len(record["events"])} for tid, record in records.items()]
    predictions = score_prefixes(auditor, tokenizer, records, rows, args.device, args.batch_size)
    for row in predictions:
        threshold = threshold_at(band, row["prefix_t"])
        row["threshold"] = threshold if math.isfinite(threshold) else None
        row["prediction"] = int(row["score"] >= threshold)
    if args.mode == "evaluate":
        from metrics import evaluate

        write_json(args.output, evaluate(predictions, records))
        if args.predictions_output:
            write_jsonl(args.predictions_output, predictions)
    else:
        write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()
