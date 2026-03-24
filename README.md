# Vector Update Experiments

This project looks at how **vector updates affect ANN search** when using an IVF-style index.

In real systems, vectors (embeddings) change over time, but indexes are not always rebuilt immediately. This creates a mismatch between the data and the index structure. The experiments here try to measure how bad that mismatch gets, and what tradeoffs are involved in fixing it.

---

## Setup

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd vector-update-exp

python3 -m venv venv
source venv/bin/activate
pip install faiss-cpu numpy
```

---

## Dataset

We use the SIFT1M dataset.

Put the files here:

```
data/sift/
```

Required files:

* sift_base.fvecs
* sift_query.fvecs
* sift_groundtruth.ivecs

The dataset itself is static — all updates are simulated in code.

---

## Running the experiments

### 1. Baseline

```bash
python src/run_baseline.py
```

This runs a standard IVF search and shows how recall changes with `nprobe`.
Use this to sanity check everything.

---

### 2. Update matrix (main experiment)

```bash
python src/run_update_matrix.py
```

This is the main experiment.

It simulates updates and compares two cases:

* **stale index** → vectors change but list assignments don’t
* **refreshed index** → vectors are reassigned after updates

It varies:

* how vectors are updated (range vs cluster-based)
* how much they move (low vs high drift)
* how many vectors are changed
* how much each query overlaps the updated region
* `nprobe`

Results go to:

```
results/update_matrix.csv
```

---

### 3. Probe position experiment

```bash
python src/run_probe_position.py
```

This checks whether it matters *which* probed clusters are affected —
i.e., is it worse if the closest cluster is stale vs one of the farther ones.

---

### 4. Remote fetch + cache tradeoff

```bash
python src/run_remote_tradeoff.py
```

This simulates a cloud setting where IVF lists are stored remotely (like S3).

It estimates:

* how much data is fetched per query
* latency from remote reads
* effect of caching
* rough cost

This is not a real S3 benchmark — just a simple model to show trends.

---

## Results

All outputs are written to:

```
results/
```

That folder is ignored by git.

If you want to share results, copy selected files to:

```
results_final/
```

and commit those.

---

## How the experiments work (short version)

The setup separates three things:

* **data** → the actual vectors (which we mutate)
* **index structure** → centroids + list assignments
* **ground truth** → recomputed after updates

After mutation:

* ground truth is recomputed on updated vectors
* the index is either:

  * kept stale, or
  * refreshed (reassignment)

Then we compare recall.

---

## What to expect

If things are working correctly:

* recall increases with `nprobe`
* stale index performs worse than refreshed
* high drift hurts more than low drift
* queries that hit updated regions degrade more
* increasing `nprobe` helps, but increases cost
* cache reduces remote fetch overhead

---

## Notes

* Experiments can take time (especially the update matrix)
* For debugging, reduce:

  ```python
  xq = xq[:500]
  nprobe_values = [10]
  ```

---

## Summary

The main takeaway is:

Updating vectors without updating the index can hurt recall, and fixing it by probing more lists comes with a cost — especially in a cloud setting.

---
