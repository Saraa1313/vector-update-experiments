import numpy as np
from coarse_quantizer import compute_centroid_ranking


def _search_single_query(q, xb_current, centroids, lists, nprobe, k):
    """
    Search one query using custom IVF:
      1. rank centroids
      2. probe top nprobe lists
      3. exact L2 search within candidates
    Returns:
      topk_ids, topk_distances, probed_list_ids, candidate_count
    """
    q = np.ascontiguousarray(q.reshape(1, -1).astype("float32"))
    _, ranked = compute_centroid_ranking(q, centroids, topn=nprobe)
    probed_lists = ranked[0]

    candidate_ids = []
    for lid in probed_lists:
        candidate_ids.extend(lists[int(lid)])

    if len(candidate_ids) == 0:
        return (
            np.full((k,), -1, dtype=np.int64),
            np.full((k,), np.inf, dtype=np.float32),
            probed_lists.copy(),
            0,
        )

    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    cand_vecs = xb_current[candidate_ids]

    diff = cand_vecs - q[0]
    dists = np.sum(diff * diff, axis=1)

    if len(dists) <= k:
        order = np.argsort(dists)
    else:
        partial = np.argpartition(dists, k)[:k]
        order = partial[np.argsort(dists[partial])]

    top_ids = candidate_ids[order]
    top_dists = dists[order]

    if len(top_ids) < k:
        pad = k - len(top_ids)
        top_ids = np.concatenate([top_ids, np.full((pad,), -1, dtype=np.int64)])
        top_dists = np.concatenate([top_dists, np.full((pad,), np.inf, dtype=np.float32)])

    return top_ids, top_dists, probed_lists.copy(), len(candidate_ids)


def search_custom_ivf(xq, xb_current, centroids, lists, nprobe, k):
    """
    Batch custom IVF search.
    Returns:
      I: neighbor ids, shape (nq, k)
      D: distances, shape (nq, k)
      probe_ids: probed list ids, shape (nq, nprobe)
      candidate_counts: number of scanned candidates per query
    """
    nq = xq.shape[0]
    I = np.empty((nq, k), dtype=np.int64)
    D = np.empty((nq, k), dtype=np.float32)
    probe_ids = np.empty((nq, nprobe), dtype=np.int64)
    candidate_counts = np.empty((nq,), dtype=np.int64)

    for i in range(nq):
        top_ids, top_dists, probed, cand_count = _search_single_query(
            xq[i], xb_current, centroids, lists, nprobe, k
        )
        I[i] = top_ids
        D[i] = top_dists
        probe_ids[i] = probed
        candidate_counts[i] = cand_count

    return I, D, probe_ids, candidate_counts