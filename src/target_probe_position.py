import faiss
import numpy as np
from load_sift import read_fvecs
from metrics import compute_recall, save_result_row, timed_call
from coarse_quantizer import (
    train_coarse_quantizer,
    assign_to_centroids,
    build_inverted_lists,
    compute_centroid_ranking,
)
from custom_ivf import search_custom_ivf


def exact_topk(xb_current, xq, k):
    xb_current = np.ascontiguousarray(xb_current.astype("float32"))
    xq = np.ascontiguousarray(xq.astype("float32"))
    d = xb_current.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(xb_current)
    D, I = index.search(xq, k)
    return D, I


def mutate_selected_ids(xb, ids, sigma=8.0, seed=42):
    rng = np.random.default_rng(seed)
    xb_mut = xb.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=xb_mut[ids].shape).astype("float32")
    xb_mut[ids] += noise
    return np.ascontiguousarray(xb_mut.astype("float32"))


def sample_ids_from_cluster(assignments, cluster_id, sample_size, rng):
    ids = np.where(assignments == cluster_id)[0]
    if len(ids) == 0:
        return None
    take = min(sample_size, len(ids))
    return rng.choice(ids, size=take, replace=False)


def main():
    print("Loading SIFT...")
    xb = np.ascontiguousarray(read_fvecs("data/sift/sift_base.fvecs").astype("float32"))
    xq = np.ascontiguousarray(read_fvecs("data/sift/sift_query.fvecs").astype("float32"))

    d = xb.shape[1]
    nlist = 4096
    k = 10
    nprobe = 10
    sample_size = 20
    sigma = 8.0
    max_queries = 1000   # use smaller first for speed/debug

    print("Training coarse quantizer...")
    centroids = train_coarse_quantizer(xb, d, nlist, seed=123)

    print("Building base assignments/lists...")
    base_assignments = assign_to_centroids(xb, centroids)
    base_lists = build_inverted_lists(base_assignments, nlist)

    print("Ranking clusters for queries...")
    # need nprobe+1 so we have an 'outside' cluster
    _, ranked = compute_centroid_ranking(xq, centroids, topn=nprobe + 1)

    num_queries = min(max_queries, xq.shape[0])

    for qi in range(num_queries):
        if qi % 100 == 0:
            print(f"Processing query {qi}/{num_queries}")

        q = xq[qi:qi+1]
        rng = np.random.default_rng(1000 + qi)

        closest_cluster = int(ranked[qi, 0])
        farthest_probed_cluster = int(ranked[qi, nprobe - 1])
        outside_cluster = int(ranked[qi, nprobe])

        cases = [
            ("closest", closest_cluster),
            ("farthest_probed", farthest_probed_cluster),
            ("outside", outside_cluster),
        ]

        for case_name, cluster_id in cases:
            chosen_ids = sample_ids_from_cluster(
                base_assignments, cluster_id, sample_size, rng
            )
            if chosen_ids is None or len(chosen_ids) == 0:
                continue

            xb_mut = mutate_selected_ids(
                xb, chosen_ids, sigma=sigma, seed=2000 + qi
            )

            # exact GT for this mutated dataset and this query
            _, gt_I = exact_topk(xb_mut, q, k)

            stale_lists = base_lists
            refreshed_assignments = assign_to_centroids(xb_mut, centroids)
            refreshed_lists = build_inverted_lists(refreshed_assignments, nlist)

            results = {}
            for index_state, lists in [("stale", stale_lists), ("refreshed", refreshed_lists)]:
                (I, D, probe_ids, cand_counts), search_time = timed_call(
                    search_custom_ivf,
                    q,
                    xb_mut,
                    centroids,
                    lists,
                    nprobe,
                    k,
                )

                recall = compute_recall(I, gt_I, k)
                results[index_state] = recall

                row = {
                    "exp": "per_query_cluster_position_controlled",
                    "query_id": int(qi),
                    "case": case_name,
                    "index_state": index_state,
                    "cluster_id": int(cluster_id),
                    "num_mutated_vectors": int(len(chosen_ids)),
                    "sample_size_target": int(sample_size),
                    "sigma": float(sigma),
                    "nlist": int(nlist),
                    "nprobe": int(nprobe),
                    "k": int(k),
                    "avg_candidates": float(np.mean(cand_counts)),
                    "search_time_sec": float(search_time),
                    "recall": float(recall),
                }
                save_result_row("results/per_query_cluster_position_controlled.csv", row)

            drop_row = {
                "exp": "per_query_cluster_position_controlled_drop",
                "query_id": int(qi),
                "case": case_name,
                "cluster_id": int(cluster_id),
                "num_mutated_vectors": int(len(chosen_ids)),
                "sample_size_target": int(sample_size),
                "sigma": float(sigma),
                "recall_stale": float(results["stale"]),
                "recall_refreshed": float(results["refreshed"]),
                "recall_drop": float(results["refreshed"] - results["stale"]),
            }
            save_result_row("results/per_query_cluster_position_controlled_drop.csv", drop_row)


if __name__ == "__main__":
    main()
