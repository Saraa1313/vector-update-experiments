import faiss
import numpy as np
from load_sift import read_fvecs, read_ivecs
from metrics import compute_recall, save_result_row, timed_call
from coarse_quantizer import train_coarse_quantizer, assign_to_centroids, build_inverted_lists
from custom_ivf import search_custom_ivf


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
    gt = read_ivecs("data/sift/sift_groundtruth.ivecs")

    d = xb.shape[1]
    nlist = 4096
    k = 10
    nprobe_values = [1, 2, 5, 10, 20, 50, 100]

    print("Training coarse quantizer...")
    centroids = train_coarse_quantizer(xb, d, nlist, seed=123)

    print("Assigning base vectors...")
    assignments = assign_to_centroids(xb, centroids)
    lists = build_inverted_lists(assignments, nlist)

    for nprobe in nprobe_values:
        print(f"Running nprobe={nprobe}")
        (I, D, probe_ids, cand_counts), search_time = timed_call(
            search_custom_ivf, xq, xb, centroids, lists, nprobe, k
        )

        recall = compute_recall(I, gt, k)

        row = {
            "exp": "baseline",
            "nlist": nlist,
            "nprobe": nprobe,
            "k": k,
            "avg_candidates": float(np.mean(cand_counts)),
            "search_time_sec": float(search_time),
            "recall": float(recall),
        }
        print(row)
        save_result_row("results/baseline.csv", row)


if __name__ == "__main__":
    main()