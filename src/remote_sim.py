import numpy as np
from collections import OrderedDict


class ListCache:
    def __init__(self, capacity_lists):
        self.capacity = int(capacity_lists)
        self.cache = OrderedDict()

    def access(self, list_id):
        list_id = int(list_id)
        if list_id in self.cache:
            self.cache.move_to_end(list_id)
            return True

        self.cache[list_id] = True
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return False


def estimate_list_sizes(lists, d, bytes_per_float=4, bytes_per_id=8):
    """
    Estimate per-list storage footprint.
    """
    sizes = {}
    for lid, ids in enumerate(lists):
        count = len(ids)
        vec_bytes = count * d * bytes_per_float
        id_bytes = count * bytes_per_id
        sizes[lid] = vec_bytes + id_bytes
    return sizes


def simulate_remote_fetch(
    probe_ids,
    list_sizes,
    cache=None,
    bandwidth_bytes_per_sec=200e6,
    fixed_overhead_ms=2.0,
):
    """
    Simulate remote fetch for one query.
    Returns bytes fetched, misses, remote latency.
    """
    total_bytes = 0
    misses = 0

    for lid in probe_ids:
        lid = int(lid)
        if cache is not None:
            if cache.access(lid):
                continue
        misses += 1
        total_bytes += list_sizes[lid]

    latency_sec = (misses * fixed_overhead_ms / 1000.0) + (
        total_bytes / bandwidth_bytes_per_sec
    )

    return {
        "bytes_fetched": total_bytes,
        "misses": misses,
        "remote_latency_sec": latency_sec,
    }


def estimate_request_cost(bytes_fetched, price_per_gb=0.023):
    """
    Very simple storage-cost-style estimate based on bytes moved.
    Treat this as an estimate, not a real bill.
    """
    gb = bytes_fetched / (1024 ** 3)
    return gb * price_per_gb