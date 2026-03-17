import faiss
import numpy as np
import time

# dimension
d = 64  

# database size
nb = 10000  

# number of queries
nq = 100  

# number of clusters
nlist = 100  

# number of clusters to probe
nprobe = 20 

np.random.seed(42)

# generate base vectors
xb = np.random.random((nb, d)).astype('float32')

# generate query vectors
xq = np.random.random((nq, d)).astype('float32')

# -----------------------------
# Build IVF index
# -----------------------------

# coarse quantizer
quantizer = faiss.IndexFlatL2(d)

# IVF index
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# train the index
index.train(xb)

# add vectors
index.add(xb)

# set nprobe
index.nprobe = nprobe

# -----------------------------
# Search
# -----------------------------

k = 10  # top-k neighbors

start = time.time()
D, I = index.search(xq, k)
end = time.time()

print("Search time:", end - start)
print("First query neighbors:", I[0])
