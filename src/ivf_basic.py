import faiss
import time
from load_sift import read_fvecs, read_ivecs

def compute_recall(I, gt, k):
    correct = 0
    for i in range(len(I)):
        correct += len(set(I[i][:k]) & set(gt[i][:k]))
    return correct / (len(I) * k)

print("Loading data...")
xb = read_fvecs("data/sift/sift_base.fvecs")
xq = read_fvecs("data/sift/sift_query.fvecs")
gt = read_ivecs("data/sift/sift_groundtruth.ivecs")
print("Data loaded.")

d = 128
nlist = 4096
nprobe = 50
k = 10

print("Building IVF index...")
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

print("Training index...")
index.train(xb)

print("Adding base vectors...")
index.add(xb)

index.nprobe = nprobe

print("Running search...")
start = time.time()
D, I = index.search(xq, k)
end = time.time()

print("Search time: {:.4f} seconds".format(end - start))

recall = compute_recall(I, gt, k)
print("Recall@{}: {:.4f}".format(k, recall))

print("First query neighbors:", I[0])
print("First query distances:", D[0])
