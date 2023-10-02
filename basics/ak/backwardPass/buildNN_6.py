from torch import randint

from buildDataset_5 import G, X, y, embeddings

BATCH_SIZE = n = 2

X_train = embeddings[X]
y_train = embeddings[y]

parameters = [embeddings]
for p in parameters: p.requires_grad = True
X_train.backward()

# create a random batch
idxs = randint(0, X_train.shape[0], (n,), generator=G)
X_train_batch, y_train_batch = X_train[idxs], y_train[idxs]
