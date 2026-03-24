import numpy as np


def mutate_by_ids(xb, ids, sigma=2.0, seed=42):
    rng = np.random.default_rng(seed)
    xb_mut = xb.copy()
    noise = rng.normal(0.0, sigma, size=(len(ids), xb.shape[1])).astype("float32")
    xb_mut[ids] += noise
    return xb_mut


def mutate_range(xb, start_idx, end_idx, sigma=2.0, seed=42):
    ids = np.arange(start_idx, end_idx, dtype=np.int64)
    xb_mut = mutate_by_ids(xb, ids, sigma=sigma, seed=seed)
    return xb_mut, ids


def mutate_cluster_subset(xb, assignments, cluster_ids, sigma=2.0, seed=42):
    mask = np.isin(assignments, cluster_ids)
    ids = np.where(mask)[0].astype(np.int64)
    xb_mut = mutate_by_ids(xb, ids, sigma=sigma, seed=seed)
    return xb_mut, ids


def mutate_toward_other_centroids(
    xb, assignments, centroids, ids, alpha=0.5, seed=42
):
    """
    Larger directional drift toward other centroids.
    Useful for 'major centroid drift' experiments.
    """
    rng = np.random.default_rng(seed)
    xb_mut = xb.copy()
    nlist = centroids.shape[0]

    for idx in ids:
        old_lid = int(assignments[idx])
        new_lid = rng.integers(0, nlist)
        while new_lid == old_lid:
            new_lid = rng.integers(0, nlist)

        shift = centroids[new_lid] - centroids[old_lid]
        xb_mut[idx] = xb_mut[idx] + alpha * shift.astype("float32")

    return xb_mut


def choose_random_clusters(nlist, fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    num = max(1, int(nlist * fraction))
    cluster_ids = rng.choice(nlist, size=num, replace=False)
    return np.sort(cluster_ids.astype(np.int64))


def choose_range_from_fraction(n, fraction=0.1):
    num = max(1, int(n * fraction))
    start = 0
    end = start + num
    return start, end