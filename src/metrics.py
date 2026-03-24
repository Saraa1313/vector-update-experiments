import csv
import os
import time


def compute_recall(I, gt, k):
    correct = 0
    nq = len(I)
    for i in range(nq):
        correct += len(set(I[i][:k]) & set(gt[i][:k]))
    return correct / (nq * k)


def timed_call(fn, *args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    end = time.time()
    return out, end - start


def save_result_row(filepath, row_dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    fieldnames = list(row_dict.keys())

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)