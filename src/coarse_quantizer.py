import faiss
import numpy as np


def train_coarse_quantizer(xb, d, nlist, seed=123):
    """
    Train coarse centroids using Faiss k-means.
    Returns centroids with shape (nlist, d).
    """
    xb = np.ascontiguousarray(xb.astype("float32"))
    kmeans = faiss.Kmeans(d, nlist, niter=20, verbose=True, seed=seed, gpu=False)
    kmeans.train(xb)
    centroids = np.ascontiguousarray(kmeans.centroids.astype("float32"))
    return centroids


def assign_to_centroids(x, centroids):
    """
    Assign each vector in x to its nearest centroid.
    Returns assignments of shape (len(x),).
    """
    x = np.ascontiguousarray(x.astype("float32"))
    centroids = np.ascontiguousarray(centroids.astype("float32"))

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, I = index.search(x, 1)
    return I[:, 0]


def build_inverted_lists(assignments, nlist):
    """
    Build Python lists of vector ids for each centroid/list.
    """
    lists = [[] for _ in range(nlist)]
    for vid, lid in enumerate(assignments):
        lists[int(lid)].append(int(vid))
    return lists


def compute_centroid_ranking(xq, centroids, topn=None):
    """
    For each query, rank centroids by distance.
    Returns:
      D: distances to ranked centroids
      I: centroid ids ranked nearest-first
    """
    xq = np.ascontiguousarray(xq.astype("float32"))
    centroids = np.ascontiguousarray(centroids.astype("float32"))
    d = centroids.shape[1]

    if topn is None:
        topn = centroids.shape[0]

    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    D, I = index.search(xq, topn)
    return D, I