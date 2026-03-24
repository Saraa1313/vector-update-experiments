import faiss
import time
import numpy as np
from load_sift import read_fvecs

def compute_recall(I, gt, k):
    correct = 0
    for i in range(len(I)):
        correct += len(set(I[i][:k]) & set(gt[i][:k]))
    return correct / (len(I) * k)

def mutate_vectors(xb, fraction=0.1, sigma=10.0, seed=42):
    """
    Mutate a fraction of vectors by adding Gaussian noise.
    Returns:
      xb_mut: mutated copy
      mutated_ids: indices of mutated vectors
    """
    rng = np.random.default_rng(seed)
    xb_mut = xb.copy()

    n = xb.shape[0]
    num_mut = int(fraction * n)
    mutated_ids = rng.choice(n, size=num_mut, replace=False)

    noise = rng.normal(loc=0.0, scale=sigma, size=(num_mut, xb.shape[1])).astype("float32")
    xb_mut[mutated_ids] += noise

    return xb_mut, mutated_ids

print("Loading SIFT data...")
xb = read_fvecs("data/sift/sift_base.fvecs").astype("float32")
xq = read_fvecs("data/sift/sift_query.fvecs").astype("float32")

xb = np.ascontiguousarray(xb)
xq = np.ascontiguousarray(xq)

print("Data loaded.")
print("Base shape:", xb.shape)
print("Query shape:", xq.shape)

# -----------------------------
# Parameters
# -----------------------------
d = 128
nlist = 4096
nprobe = 10
k = 10
mutation_fraction = 0.10

# Try two settings later:
# sigma = 2.0   # low drift
# sigma = 20.0  # high drift
sigma = 10.0

# -----------------------------
# Step 1: Build stale IVF on original data
# -----------------------------
print("\nBuilding stale IVF index on original vectors...")
quantizer = faiss.IndexFlatL2(d)
ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist)

print("Training IVF...")
ivf_index.train(xb)

print("Adding original vectors...")
ivf_index.add(xb)
ivf_index.nprobe = nprobe

# -----------------------------
# Step 2: Mutate vectors
# -----------------------------
print("\nMutating vectors...")
xb_mut, mutated_ids = mutate_vectors(
    xb,
    fraction=mutation_fraction,
    sigma=sigma,
    seed=42
)

xb_mut = np.ascontiguousarray(xb_mut)

print(f"Mutated {len(mutated_ids)} vectors out of {xb.shape[0]}")

# -----------------------------
# Step 3: Build fresh exact index on mutated data
# This acts as ground truth after updates
# -----------------------------
print("\nBuilding exact index on mutated vectors for ground truth...")
gt_index = faiss.IndexFlatL2(d)
gt_index.add(xb_mut)

print("Computing mutated ground truth...")
gt_D, gt_I = gt_index.search(xq, k)

# -----------------------------
# Step 4: Search with stale IVF index
# -----------------------------
print("\nSearching stale IVF index...")
start = time.time()
D_stale, I_stale = ivf_index.search(xq, k)
end = time.time()

stale_time = end - start
stale_recall = compute_recall(I_stale, gt_I, k)

print("\n--- Results ---")
print(f"Mutation fraction: {mutation_fraction}")
print(f"Drift sigma: {sigma}")
print(f"nprobe: {nprobe}")
print(f"Stale IVF search time: {stale_time:.4f} sec")
print(f"Recall@{k} vs mutated ground truth: {stale_recall:.4f}")

print("\nFirst query stale IVF neighbors:")
print(I_stale[0])

print("\nFirst query mutated ground-truth neighbors:")
print(gt_I[0])

