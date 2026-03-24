import numpy as np
from load_sift import read_fvecs, read_ivecs
from metrics import compute_recall, save_result_row, timed_call
from coarse_quantizer import train_coarse_quantizer, assign_to_centroids, build_inverted_lists
from custom_ivf import search_custom_ivf
from remote_sim import (
    estimate_list_sizes,
    simulate_remote_fetch,
    ListCache,
    estimate_request_cost,
)


def main():
    print("Loading SIFT...")
    xb = np.ascontiguousarray(read_fvecs("data/sift/sift_base.fvecs").astype("float32"))
    xq = np.ascontiguousarray(read_fvecs("data/sift/sift_query.fvecs").astype("float32"))
    gt = read_ivecs("data/sift/sift_groundtruth.ivecs")

    d = xb.shape[1]
    nlist = 4096
    k = 10
    nprobe_values = [1, 2, 5, 10, 20, 50]
    cache_sizes = [0, 64, 256, 1024]

    print("Training coarse quantizer...")
    centroids = train_coarse_quantizer(xb, d, nlist, seed=123)

    print("Building assignments/lists...")
    assignments = assign_to_centroids(xb, centroids)
    lists = build_inverted_lists(assignments, nlist)
    list_sizes = estimate_list_sizes(lists, d)

    for cache_size in cache_sizes:
        cache = None if cache_size == 0 else ListCache(cache_size)

        for nprobe in nprobe_values:
            print(f"Running cache_size={cache_size}, nprobe={nprobe}")

            (I, D, probe_ids, cand_counts), search_time = timed_call(
                search_custom_ivf,
                xq,
                xb,
                centroids,
                lists,
                nprobe,
                k,
            )

            recall = compute_recall(I, gt, k)

            total_bytes = 0
            total_remote_latency = 0.0
            total_misses = 0
            total_cost = 0.0

            for q in range(xq.shape[0]):
                sim = simulate_remote_fetch(
                    probe_ids[q],
                    list_sizes,
                    cache=cache,
                    bandwidth_bytes_per_sec=200e6,
                    fixed_overhead_ms=2.0,
                )
                total_bytes += sim["bytes_fetched"]
                total_remote_latency += sim["remote_latency_sec"]
                total_misses += sim["misses"]
                total_cost += estimate_request_cost(sim["bytes_fetched"])

            row = {
                "exp": "remote_tradeoff",
                "cache_lists": int(cache_size),
                "nlist": int(nlist),
                "nprobe": int(nprobe),
                "k": int(k),
                "avg_candidates": float(np.mean(cand_counts)),
                "search_time_sec": float(search_time),
                "recall": float(recall),
                "avg_bytes_fetched": float(total_bytes / xq.shape[0]),
                "avg_remote_latency_sec": float(total_remote_latency / xq.shape[0]),
                "avg_estimated_cost": float(total_cost / xq.shape[0]),
                "total_misses": int(total_misses),
            }
            print(row)
            save_result_row("results/remote_tradeoff.csv", row)


if __name__ == "__main__":
    main()