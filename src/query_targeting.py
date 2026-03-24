import numpy as np


def compute_mutated_neighbor_fraction(gt_I, mutated_ids_set, k):
    """
    For each query, compute fraction of top-k exact neighbors that are mutated.
    """
    fracs = np.zeros((gt_I.shape[0],), dtype=np.float32)
    for i in range(gt_I.shape[0]):
        count = 0
        for nid in gt_I[i, :k]:
            if int(nid) in mutated_ids_set:
                count += 1
        fracs[i] = count / k
    return fracs


def bucket_queries_by_target_fraction(fracs):
    """
    Buckets queries into the exact bins you asked for:
      0.0, (0,0.25], (0.25,0.5], (0.5,0.75], (0.75,1.0]
    Returns dict of bucket_name -> query indices.
    """
    buckets = {
        "0.00": np.where(fracs == 0.0)[0],
        "0.25": np.where((fracs > 0.0) & (fracs <= 0.25))[0],
        "0.50": np.where((fracs > 0.25) & (fracs <= 0.50))[0],
        "0.75": np.where((fracs > 0.50) & (fracs <= 0.75))[0],
        "1.00": np.where((fracs > 0.75) & (fracs <= 1.00))[0],
    }
    return buckets


def bucket_queries_by_nearest_target(fracs, targets=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Alternative: assign each query to nearest target fraction.
    Useful if some bins are sparse.
    """
    targets = np.asarray(targets, dtype=np.float32)
    buckets = {f"{t:.2f}": [] for t in targets}

    for i, f in enumerate(fracs):
        t = targets[np.argmin(np.abs(targets - f))]
        buckets[f"{t:.2f}"].append(i)

    for key in buckets:
        buckets[key] = np.asarray(buckets[key], dtype=np.int64)

    return buckets