import faiss
import numpy as np

# dimension of each vector
d = 4  

# create 100 random vectors (database)
xb = np.random.random((100, d)).astype('float32')

# create index (exact search using L2 distance)
index = faiss.IndexFlatL2(d)

# add vectors to index
index.add(xb)

# create 5 random query vectors
xq = np.random.random((5, d)).astype('float32')

# search top 3 nearest neighbors
D, I = index.search(xq, 3)

print("Distances:\n", D)
print("Indices:\n", I)
