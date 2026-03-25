import csv
import time
import numpy as np
from load_sift import read_fvecs, read_ivecs
from minio_ivf_utils import (
    get_minio_client,
    build_coarse_quantizer,
    search_remote_ivf_with_cache,
)

def compute_recall(I, gt, k):
    correct = 0
    for i in range(len(I)):
        correct += len(set(I[i][:k]) & set(gt[i][:k]))
    return correct / (len(I) * k)

def estimate_cost(bytes_fetched_per_query, objects_fetched_per_query):
    # simple rough S3-style cost model
    # GET requests: $0.0004 per 1000 requests
    # data transfer / retrieval: $0.023 per GB
    request_cost_per_query = objects_fetched_per_query * (0.0004 / 1000.0)
    transfer_cost_per_query = (bytes_fetched_per_query / (1024**3)) * 0.023
    total_cost_per_query = request_cost_per_query + transfer_cost_per_query

    return {
        "request_cost_per_1m_queries": request_cost_per_query * 1_000_000,
        "transfer_cost_per_1m_queries": transfer_cost_per_query * 1_000_000,
        "total_cost_per_1m_queries": total_cost_per_query * 1_000_000,
    }

print("Loading data...")
xb = read_fvecs("data/sift/sift_base.fvecs").astype("float32")
xq = read_fvecs("data/sift/sift_query.fvecs").astype("float32")
gt = read_ivecs("data/sift/sift_groundtruth.ivecs")

xb = np.ascontiguousarray(xb)
xq = np.ascontiguousarray(xq)

# static workload: repeat same query set multiple times
base_xq = xq[:200]
base_gt = gt[:200]

repeat_factor = 10
xq_stream = np.tile(base_xq, (repeat_factor, 1))
gt_stream = np.tile(base_gt, (repeat_factor, 1))

d = 128
nlist = 4096
k = 10
nprobe = 10
bucket_name = "ivf-index"
cache_sizes = [0, 64, 256, 1024]

print("Building centroids locally...")
centroids, _ = build_coarse_quantizer(xb, d, nlist)

print("Connecting to MinIO...")
client = get_minio_client()

results = []
output_file = "results/cache_static.csv"

print("Running cache experiment...")
for cache_size in cache_sizes:
    start = time.time()

    D, I, total_bytes, total_fetch_time, total_objects, cache_hits, cache_misses, hit_rate = search_remote_ivf_with_cache(
        client=client,
        bucket_name=bucket_name,
        centroids=centroids,
        queries=xq_stream,
        k=k,
        nprobe=nprobe,
        cache_size=cache_size
    )

    end = time.time()
    total_time = end - start
    recall = compute_recall(I, gt_stream, k)
    throughput_qps = len(xq_stream) / total_time

    avg_bytes_per_query = total_bytes / len(xq_stream)
    avg_objects_per_query = total_objects / len(xq_stream)

    costs = estimate_cost(avg_bytes_per_query, avg_objects_per_query)

    row = {
        "cache_size_lists": cache_size,
        "nprobe": nprobe,
        "num_queries": len(xq_stream),
        "recall_at_10": recall,
        "avg_latency_per_query_ms": (total_time / len(xq_stream)) * 1000.0,
        "avg_fetch_time_per_query_ms": (total_fetch_time / len(xq_stream)) * 1000.0,
        "avg_bytes_per_query": avg_bytes_per_query,
        "avg_objects_fetched_per_query": avg_objects_per_query,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": hit_rate,
        "throughput_qps": throughput_qps,
        "estimated_request_cost_per_1m_queries": costs["request_cost_per_1m_queries"],
        "estimated_transfer_cost_per_1m_queries": costs["transfer_cost_per_1m_queries"],
        "estimated_total_cost_per_1m_queries": costs["total_cost_per_1m_queries"],
    }

    results.append(row)
    print(row)

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved results to {output_file}")
