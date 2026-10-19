import math

import numpy as np


def select_calibration_ids(trajectories, index, seed=0, fraction=0.4):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    represented_ids = {row["trajectory_id"] for row in index}
    groups = np.asarray(sorted({trajectories[tid]["parent_group_id"] for tid in represented_ids}))
    if not len(groups):
        raise ValueError("The training index contains no parent groups")
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(groups) * fraction)))
    selected = set(groups[rng.permutation(len(groups))[:count]])
    return sorted(tid for tid in represented_ids
                  if trajectories[tid]["parent_group_id"] in selected
                  and int(trajectories[tid]["outcome"]) == 0
                  and len(trajectories[tid]["events"]) >= 2)


def fit_thresholds(prediction_rows, alpha=0.05, min_support=20):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    runs = {}
    for row in prediction_rows:
        tid, t, score = str(row["trajectory_id"]), row["prefix_t"], float(row["score"])
        if isinstance(t, bool) or not isinstance(t, int) or t < 1:
            raise ValueError("prefix_t must be a positive integer")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Calibration scores must be finite and lie in [0, 1]")
        run = runs.setdefault(tid, {})
        if t in run:
            raise ValueError(f"Duplicate calibration prefix: {(tid, t)}")
        run[t] = score
    if len(runs) < min_support:
        raise ValueError(f"Calibration requires at least {min_support} successful trajectories")
    for run in runs.values():
        if sorted(run) != list(range(1, max(run) + 1)):
            raise ValueError("Every calibration trajectory must supply all nonterminal steps from 1")
    t_max = max(max(run) for run in runs.values())
    columns = [np.sort(np.asarray([run[t] for run in runs.values() if t in run], dtype=float))
               for t in range(1, t_max + 1)]
    support = [len(column) for column in columns]
    last_supported = max(t for t, count in enumerate(support, 1) if count >= min_support)
    reference = lambda t: columns[min(t, last_supported) - 1]
    maxima = []
    for run in runs.values():
        maxima.append(max(float(np.searchsorted(reference(t), score, side="right") / len(reference(t)))
                          for t, score in run.items()))
    maxima = np.sort(np.asarray(maxima, dtype=float))
    k = int(np.ceil((len(runs) + 1) * (1 - alpha)))
    q = 0.0 if k <= 0 else (None if k > len(runs) else float(maxima[k - 1]))
    thresholds = [None if q is None else float(np.quantile(reference(t), q, method="higher"))
                  for t in range(1, t_max + 1)]
    return {"alpha": float(alpha), "min_support": min_support,
            "n_calibration_trajectories": len(runs), "rank_cutoff": q,
            "order_statistic_k": k, "thresholds": thresholds, "support": support,
            "last_supported_step": last_supported, "t_max": t_max,
            "infinite_threshold": "null"}


def threshold_at(band, t):
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("The interaction step must be a positive integer")
    thresholds = band["thresholds"]
    if not thresholds:
        raise ValueError("The threshold band is empty")
    value = thresholds[min(int(t), len(thresholds)) - 1]
    return math.inf if value is None else float(value)
