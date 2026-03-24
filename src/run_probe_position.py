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
from updates import mutate_cluster_subset


def exact_topk(xb_current, xq, k):
    xb_current = np.ascontiguousarray(xb_current.astype("float32"))
    xq = np.ascontiguousarray(xq.astype("float32"))
    d = xb_current.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(xb_current)
    D, I = index.search(xq, k)
    return D, I


def main():
    print("Loading SIFT...")
    xb = np.ascontiguousarray(read_fvecs("data/sift/sift_base.fvecs").astype("float32"))
    xq = np.ascontiguousarray(read_fvecs("data/sift/sift_query.fvecs").astype("float32"))

    d = xb.shape[1]
    nlist = 4096
    k = 10
    nprobe = 10

    print("Training coarse quantizer...")
    centroids = train_coarse_quantizer(xb, d, nlist, seed=123)

    print("Building base assignments/lists...")
    base_assignments = assign_to_centroids(xb, centroids)
    base_lists = build_inverted_lists(base_assignments, nlist)

    print("Ranking top-nprobe clusters for queries...")
    _, ranked = compute_centroid_ranking(xq, centroids, topn=nprobe)

    # query-conditioned sets
    closest_clusters = ranked[:, 0]
    farthest_clusters = ranked[:, nprobe - 1]

    # choose clusters that appear often in those positions
    closest_unique, closest_counts = np.unique(closest_clusters, return_counts=True)
    far_unique, far_counts = np.unique(farthest_clusters, return_counts=True)

    m = 100
    closest_top = closest_unique[np.argsort(-closest_counts)[:m]]
    farthest_top = far_unique[np.argsort(-far_counts)[:m]]

    for case_name, chosen_clusters in [
        ("closest_probed_mutated", closest_top),
        ("farthest_probed_mutated", farthest_top),
    ]:
        print(f"Running case: {case_name}")

        xb_mut, mutated_ids = mutate_cluster_subset(
            xb, base_assignments, chosen_clusters, sigma=8.0, seed=42
        )
        xb_mut = np.ascontiguousarray(xb_mut.astype("float32"))

        _, gt_I = exact_topk(xb_mut, xq, k)

        # stale vs refreshed
        stale_lists = base_lists
        refreshed_assignments = assign_to_centroids(xb_mut, centroids)
        refreshed_lists = build_inverted_lists(refreshed_assignments, nlist)

        for index_state, lists in [("stale", stale_lists), ("refreshed", refreshed_lists)]:
            (I, D, probe_ids, cand_counts), search_time = timed_call(
                search_custom_ivf,
                xq,
                xb_mut,
                centroids,
                lists,
                nprobe,
                k,
            )

            recall = compute_recall(I, gt_I, k)

            row = {
                "exp": "probe_position",
                "case": case_name,
                "index_state": index_state,
                "num_mutated_clusters": int(len(chosen_clusters)),
                "nlist": int(nlist),
                "nprobe": int(nprobe),
                "k": int(k),
                "avg_candidates": float(np.mean(cand_counts)),
                "search_time_sec": float(search_time),
                "recall": float(recall),
            }
            print(row)
            save_result_row("results/probe_position.csv", row)


if __name__ == "__main__":
    main()