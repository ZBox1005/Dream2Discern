import numpy as np


def _auroc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    base = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = base[i:j + 1].mean()
        i = j + 1
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    return float((ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def _fpr95(labels, scores):
    positive = np.sort(scores[labels == 1])
    negative = scores[labels == 0]
    index = min(len(positive) - 1, int(np.floor((1.0 - 0.95) * len(positive))))
    threshold = positive[index]
    return float((negative >= threshold).mean())


def evaluate(prediction_rows, records):
    grouped = {}
    for tid, record in records.items():
        if record.get("outcome") not in (0, 1):
            raise ValueError("Offline evaluation requires outcome=0 or outcome=1")
        if len(record["events"]) >= 2:
            grouped[tid] = {}
    seen = set()
    for row in prediction_rows:
        tid, t = str(row["trajectory_id"]), row["prefix_t"]
        if tid not in records:
            raise ValueError(f"Unknown trajectory_id: {tid}")
        length = len(records[tid]["events"])
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 1 <= t <= length:
            raise ValueError("prefix_t must be an integer between 1 and trajectory length")
        if (tid, t) in seen:
            raise ValueError(f"Duplicate prediction: {(tid, t)}")
        seen.add((tid, t))
        score = float(row["score"])
        prediction = row["prediction"]
        if not np.isfinite(score):
            raise ValueError("Evaluation scores must be finite")
        if prediction not in (0, 1):
            raise ValueError("prediction must be 0 or 1")
        if t < length:
            grouped[tid][int(t)] = (score, int(prediction))
    if not grouped:
        raise ValueError("No trajectories have nonterminal prefixes")
    labels, scores, fired, first_alarm, lengths = [], [], [], [], []
    for tid, values in grouped.items():
        record = records[tid]
        length = len(record["events"])
        if set(values) != set(range(1, length)):
            raise ValueError(f"Evaluation requires every nonterminal prefix for {tid}")
        alarms = [t for t, (_, prediction) in values.items() if prediction == 1]
        labels.append(int(record["outcome"]))
        scores.append(max(score for score, _ in values.values()))
        fired.append(bool(alarms))
        first_alarm.append(min(alarms) if alarms else np.inf)
        lengths.append(length)
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fired = np.asarray(fired, dtype=bool)
    first_alarm = np.asarray(first_alarm, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    positive, negative = labels == 1, labels == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        raise ValueError("Evaluation requires both successful and failed trajectories")
    true_positive = positive & fired
    true_negative = negative & ~fired
    tpr = float(true_positive.sum() / n_positive)
    tnr = float(true_negative.sum() / n_negative)
    success_fraction = n_negative / (n_positive + n_negative)
    timeliness = float(np.sum(1.0 - first_alarm[true_positive] / lengths[true_positive]) / n_positive)
    return {
        "n_trajectories": n_positive + n_negative,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_true_positive": int(true_positive.sum()),
        "n_true_negative": int(true_negative.sum()),
        "trajectory_auroc": _auroc(labels, scores),
        "fpr_at_tpr95": _fpr95(labels, scores),
        "wACC": float(success_fraction * tnr + (1.0 - success_fraction) * tpr),
        "TWA": float(0.5 * (timeliness + tnr)),
        "tpr_recall": tpr,
        "tnr": tnr,
    }
