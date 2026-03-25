import io
import numpy as np
import faiss
from minio import Minio
import time
from collections import OrderedDict


def get_minio_client():
    return Minio(
        "127.0.0.1:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )


def ensure_bucket(client, bucket_name):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def build_coarse_quantizer(xb, d, nlist):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(xb)

    # assign every base vector to its nearest centroid
    _, assignments = index.quantizer.search(xb, 1)
    assignments = assignments[:, 0]

    # reconstruct centroids
    centroids = np.zeros((nlist, d), dtype=np.float32)
    for i in range(nlist):
        centroids[i] = index.quantizer.reconstruct(i)

    return centroids, assignments


def build_inverted_lists(xb, assignments, nlist):
    lists = [[] for _ in range(nlist)]
    for vid, lid in enumerate(assignments):
        lists[int(lid)].append(int(vid))

    payloads = {}
    for lid in range(nlist):
        ids = np.array(lists[lid], dtype=np.int64)
        if len(ids) > 0:
            vecs = xb[ids]
        else:
            vecs = np.empty((0, xb.shape[1]), dtype=np.float32)

        payloads[lid] = {
            "ids": ids,
            "vecs": vecs,
        }
    return payloads


def upload_ivf_lists_to_minio(client, bucket_name, payloads):
    for lid, payload in payloads.items():
        buf = io.BytesIO()
        np.savez(buf, ids=payload["ids"], vecs=payload["vecs"])
        buf.seek(0)
        raw = buf.getvalue()

        object_name = f"lists/list_{lid:06d}.npz"
        client.put_object(
            bucket_name,
            object_name,
            io.BytesIO(raw),
            length=len(raw),
            content_type="application/octet-stream",
        )


def upload_centroids_to_minio(client, bucket_name, centroids):
    buf = io.BytesIO()
    np.save(buf, centroids)
    buf.seek(0)
    raw = buf.getvalue()

    client.put_object(
        bucket_name,
        "centroids.npy",
        io.BytesIO(raw),
        length=len(raw),
        content_type="application/octet-stream",
    )
import time

def fetch_list_from_minio(client, bucket_name, lid):
    object_name = f"lists/list_{lid:06d}.npz"

    start = time.time()
    response = client.get_object(bucket_name, object_name)
    raw = response.read()
    response.close()
    response.release_conn()
    end = time.time()

    buf = io.BytesIO(raw)
    data = np.load(buf)
    ids = data["ids"]
    vecs = data["vecs"]

    return ids, vecs, len(raw), (end - start)


def get_probed_lists(centroids, queries, nprobe):
    q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
    c_norm = np.sum(centroids ** 2, axis=1)
    dists = q_norm + c_norm - 2 * queries @ centroids.T

    probe_ids = np.argpartition(dists, kth=nprobe - 1, axis=1)[:, :nprobe]

    sorted_probe_ids = np.zeros_like(probe_ids)
    for i in range(queries.shape[0]):
        row = probe_ids[i]
        row_d = dists[i, row]
        order = np.argsort(row_d)
        sorted_probe_ids[i] = row[order]

    return sorted_probe_ids


def search_remote_ivf(client, bucket_name, centroids, queries, k, nprobe):
    probed_lists = get_probed_lists(centroids, queries, nprobe)

    all_I = []
    all_D = []

    total_bytes = 0
    total_fetch_time = 0.0
    total_objects = 0

    for qi in range(queries.shape[0]):
        q = queries[qi]
        candidate_ids = []
        candidate_vecs = []

        for lid in probed_lists[qi]:
            ids, vecs, nbytes, fetch_time = fetch_list_from_minio(client, bucket_name, int(lid))
            total_bytes += nbytes
            total_fetch_time += fetch_time
            total_objects += 1

            if len(ids) > 0:
                candidate_ids.append(ids)
                candidate_vecs.append(vecs)

        if len(candidate_ids) == 0:
            all_I.append(np.full(k, -1, dtype=np.int64))
            all_D.append(np.full(k, np.inf, dtype=np.float32))
            continue

        candidate_ids = np.concatenate(candidate_ids, axis=0)
        candidate_vecs = np.concatenate(candidate_vecs, axis=0)

        diff = candidate_vecs - q
        dist = np.sum(diff * diff, axis=1)

        topk_idx = np.argsort(dist)[:k]
        all_I.append(candidate_ids[topk_idx])
        all_D.append(dist[topk_idx])

    return (
        np.array(all_D, dtype=np.float32),
        np.array(all_I, dtype=np.int64),
        total_bytes,
        total_fetch_time,
        total_objects
    )
def search_remote_ivf_with_cache(client, bucket_name, centroids, queries, k, nprobe, cache_size):
    probed_lists = get_probed_lists(centroids, queries, nprobe)

    cache = ListCache(cache_size) if cache_size > 0 else None

    all_I = []
    all_D = []

    total_bytes = 0
    total_fetch_time = 0.0
    total_objects = 0
    cache_hits = 0
    cache_misses = 0

    for qi in range(queries.shape[0]):
        q = queries[qi]
        candidate_ids = []
        candidate_vecs = []

        for lid in probed_lists[qi]:
            lid = int(lid)

            cached_value = None
            if cache is not None:
                cached_value = cache.get(lid)

            if cached_value is not None:
                ids, vecs = cached_value
                cache_hits += 1
            else:
                ids, vecs, nbytes, fetch_time = fetch_list_from_minio(client, bucket_name, lid)
                total_bytes += nbytes
                total_fetch_time += fetch_time
                total_objects += 1
                cache_misses += 1

                if cache is not None:
                    cache.put(lid, (ids, vecs))

            if len(ids) > 0:
                candidate_ids.append(ids)
                candidate_vecs.append(vecs)

        if len(candidate_ids) == 0:
            all_I.append(np.full(k, -1, dtype=np.int64))
            all_D.append(np.full(k, np.inf, dtype=np.float32))
            continue

        candidate_ids = np.concatenate(candidate_ids, axis=0)
        candidate_vecs = np.concatenate(candidate_vecs, axis=0)

        diff = candidate_vecs - q
        dist = np.sum(diff * diff, axis=1)

        topk_idx = np.argsort(dist)[:k]
        all_I.append(candidate_ids[topk_idx])
        all_D.append(dist[topk_idx])

    total_accesses = cache_hits + cache_misses
    hit_rate = cache_hits / total_accesses if total_accesses > 0 else 0.0

    return (
        np.array(all_D, dtype=np.float32),
        np.array(all_I, dtype=np.int64),
        total_bytes,
        total_fetch_time,
        total_objects,
        cache_hits,
        cache_misses,
        hit_rate,
    )

class ListCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
