import numpy as np


def read_fvecs(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    return fv.reshape(-1, dim + 1)[:, 1:]


def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    dim = iv[0]
    return iv.reshape(-1, dim + 1)[:, 1:]


if __name__ == "__main__":
    base = read_fvecs("data/sift/sift_base.fvecs")
    query = read_fvecs("data/sift/sift_query.fvecs")
    gt = read_ivecs("data/sift/sift_groundtruth.ivecs")

    print("Base shape:", base.shape, base.dtype)
    print("Query shape:", query.shape, query.dtype)
    print("GT shape:", gt.shape, gt.dtype)
    print("First base vector first 5 values:", base[0][:5])
