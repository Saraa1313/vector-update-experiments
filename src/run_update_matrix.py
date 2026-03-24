import faiss
import numpy as np
from load_sift import read_fvecs
from metrics import compute_recall, save_result_row, timed_call
from coarse_quantizer import (
    train_coarse_quantizer,
    assign_to_centroids,
    build_inverted_lists,
)
from custom_ivf import search_custom_ivf
from updates import (
    mutate_range,
    mutate_cluster_subset,
    mutate_toward_other_centroids,
    choose_random_clusters,
    choose_range_from_fraction,
)
from query_targeting import (
    compute_mutated_neighbor_fraction,
    bucket_queries_by_nearest_target,
)


def exact_topk(xb_current, xq, k):
    xb_current = np.ascontiguousarray(xb_current.astype("float32"))
    xq = np.ascontiguousarray(xq.astype("float32"))
    d = xb_current.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(xb_current)
    D, I = index.search(xq, k)
    return D, I


def run_one_case(
    xb,
    xq,
    centroids,
    base_assignments,
    base_lists,
    nlist,
    k,
    update_type,
    drift_mode,
    mutation_fraction,
    nprobe_values,
    seed=42,
):
    n = xb.shape[0]

    # -------------------------
    # choose mutated ids + build xb_mut
    # -------------------------
    if update_type == "range":
        start, end = choose_range_from_fraction(n, mutation_fraction)
        xb_mut, mutated_ids = mutate_range(
            xb, start, end, sigma=2.0 if drift_mode == "low" else 8.0, seed=seed
        )
        if drift_mode == "high":
            xb_mut = mutate_toward_other_centroids(
                xb_mut,
                base_assignments,
                centroids,
                mutated_ids,
                alpha=0.5,
                seed=seed,
            )

    elif update_type == "cluster_subset":
        chosen_clusters = choose_random_clusters(nlist, fraction=mutation_fraction, seed=seed)
        xb_mut, mutated_ids = mutate_cluster_subset(
            xb,
            base_assignments,
            chosen_clusters,
            sigma=2.0 if drift_mode == "low" else 8.0,
            seed=seed,
        )
        if drift_mode == "high":
            xb_mut = mutate_toward_other_centroids(
                xb_mut,
                base_assignments,
                centroids,
                mutated_ids,
                alpha=0.5,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown update_type: {update_type}")

    xb_mut = np.ascontiguousarray(xb_mut.astype("float32"))
    mutated_ids_set = set(int(x) for x in mutated_ids.tolist())

    # -------------------------
    # exact mutated ground truth
    # -------------------------
    _, gt_I = exact_topk(xb_mut, xq, k)

    # -------------------------
    # stale lists: keep old assignments
    # refreshed lists: recompute assignments
    # -------------------------
    stale_lists = base_lists
    refreshed_assignments = assign_to_centroids(xb_mut, centroids)
    refreshed_lists = build_inverted_lists(refreshed_assignments, nlist)

    # -------------------------
    # query targeting fractions from mutated GT overlap
    # -------------------------
    fracs = compute_mutated_neighbor_fraction(gt_I, mutated_ids_set, k)
    buckets = bucket_queries_by_nearest_target(fracs)

    for target_bucket, qids in buckets.items():
        if len(qids) == 0:
            continue

        xq_bucket = xq[qids]
        gt_bucket = gt_I[qids]

        for index_state, lists in [("stale", stale_lists), ("refreshed", refreshed_lists)]:
            for nprobe in nprobe_values:
                (I, D, probe_ids, cand_counts), search_time = timed_call(
                    search_custom_ivf,
                    xq_bucket,
                    xb_mut,
                    centroids,
                    lists,
                    nprobe,
                    k,
                )

                recall = compute_recall(I, gt_bucket, k)

                row = {
                    "exp": "update_matrix",
                    "update_type": update_type,
                    "drift_mode": drift_mode,
                    "index_state": index_state,
                    "mutation_fraction": float(mutation_fraction),
                    "target_bucket": target_bucket,
                    "nlist": int(nlist),
                    "nprobe": int(nprobe),
                    "k": int(k),
                    "num_queries": int(len(qids)),
                    "num_mutated_vectors": int(len(mutated_ids)),
                    "avg_candidates": float(np.mean(cand_counts)),
                    "search_time_sec": float(search_time),
                    "recall": float(recall),
                }
                print(row)
                save_result_row("results/update_matrix.csv", row)


def main():
    print("Loading SIFT...")
    xb = np.ascontiguousarray(read_fvecs("data/sift/sift_base.fvecs").astype("float32"))
    xq = np.ascontiguousarray(read_fvecs("data/sift/sift_query.fvecs").astype("float32"))

    d = xb.shape[1]
    nlist = 4096
    k = 10
    nprobe_values = [1, 2, 5, 10, 20, 50]
    mutation_fractions = [0.01, 0.05, 0.10, 0.20]

    print("Training coarse quantizer...")
    centroids = train_coarse_quantizer(xb, d, nlist, seed=123)

    print("Building base assignments/lists...")
    base_assignments = assign_to_centroids(xb, centroids)
    base_lists = build_inverted_lists(base_assignments, nlist)

    for update_type in ["range", "cluster_subset"]:
        for drift_mode in ["low", "high"]:
            for mutation_fraction in mutation_fractions:
                print(
                    f"Running update_type={update_type}, "
                    f"drift_mode={drift_mode}, mutation_fraction={mutation_fraction}"
                )
                run_one_case(
                    xb=xb,
                    xq=xq,
                    centroids=centroids,
                    base_assignments=base_assignments,
                    base_lists=base_lists,
                    nlist=nlist,
                    k=k,
                    update_type=update_type,
                    drift_mode=drift_mode,
                    mutation_fraction=mutation_fraction,
                    nprobe_values=nprobe_values,
                    seed=42,
                )


if __name__ == "__main__":
    main()