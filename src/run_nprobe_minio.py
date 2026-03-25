import csv
import time
import numpy as np
from load_sift import read_fvecs, read_ivecs
from minio_ivf_utils import (
    get_minio_client,
    build_coarse_quantizer,
    search_remote_ivf,
)

def compute_recall(I, gt, k):
    correct = 0
    for i in range(len(I)):
        correct += len(set(I[i][:k]) & set(gt[i][:k]))
    return correct / (len(I) * k)

print("Loading data...")
xb = read_fvecs("data/sift/sift_base.fvecs").astype("float32")
xq = read_fvecs("data/sift/sift_query.fvecs").astype("float32")
gt = read_ivecs("data/sift/sift_groundtruth.ivecs")

xb = np.ascontiguousarray(xb)
xq = np.ascontiguousarray(xq)

# smaller query subset for first run
xq = xq[:200]
gt = gt[:200]

d = 128
nlist = 4096
k = 10
bucket_name = "ivf-index"
nprobe_values = [1, 2, 5, 10, 20, 50, 100]

print("Building centroids locally...")
centroids, _ = build_coarse_quantizer(xb, d, nlist)

print("Connecting to MinIO...")
client = get_minio_client()

results = []
output_file = "results/nprobe_minio.csv"

print("Running nprobe sweep...")
for nprobe in nprobe_values:
    start = time.time()

    D, I, total_bytes, total_fetch_time, total_objects = search_remote_ivf(
        client=client,
        bucket_name=bucket_name,
        centroids=centroids,
        queries=xq,
        k=k,
        nprobe=nprobe
    )

    end = time.time()
    total_time = end - start
    recall = compute_recall(I, gt, k)

    row = {
        "nprobe": nprobe,
        "num_queries": len(xq),
        "recall_at_10": recall,
        "total_time_sec": total_time,
        "avg_latency_per_query_ms": (total_time / len(xq)) * 1000.0,
        "avg_fetch_time_per_query_ms": (total_fetch_time / len(xq)) * 1000.0,
        "avg_bytes_per_query": total_bytes / len(xq),
        "avg_objects_per_query": total_objects / len(xq),
    }
    results.append(row)
    print(row)

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved results to {output_file}")
