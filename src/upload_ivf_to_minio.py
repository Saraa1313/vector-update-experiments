import numpy as np
from load_sift import read_fvecs
from minio_ivf_utils import (
    get_minio_client,
    ensure_bucket,
    build_coarse_quantizer,
    build_inverted_lists,
    upload_ivf_lists_to_minio,
    upload_centroids_to_minio,
)

print("Loading base vectors...")
xb = read_fvecs("data/sift/sift_base.fvecs").astype("float32")
xb = np.ascontiguousarray(xb)

d = 128
nlist = 4096
bucket_name = "ivf-index"

print("Connecting to MinIO...")
client = get_minio_client()
ensure_bucket(client, bucket_name)

print("Training coarse quantizer...")
centroids, assignments = build_coarse_quantizer(xb, d, nlist)

print("Building inverted lists...")
payloads = build_inverted_lists(xb, assignments, nlist)

print("Uploading centroids...")
upload_centroids_to_minio(client, bucket_name, centroids)

print("Uploading IVF lists...")
upload_ivf_lists_to_minio(client, bucket_name, payloads)

print("Upload complete.")
print(f"Bucket: {bucket_name}")
print(f"Uploaded {len(payloads)} list objects.")
