from torch import (
    allclose, arange, randint, randn
    )
from torch.nn import Embedding, LayerNorm, Linear
from torch.nn.functional import cross_entropy

from buildDataset import (
    G, vocab_size, X, y
    )

l = Linear(2, 26)
batch_size, number_of_alphabets, embedding_dim = 2, 13, 2
last_dim = embedding_dim
embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
X_train_embeds = embedding(X).view(
    batch_size,          # 2 batches, with each batch
    number_of_alphabets, # containing 13 alphabets, where
    embedding_dim        # each alphabet has two features to describe itself
    )


layer_norm = LayerNorm(last_dim, elementwise_affine = False)
layer_norm_out = layer_norm(X_train_embeds)
print("y: ", layer_norm_out)



import torch

batch_size, seq_size, dim = 2, 3, 4
last_dims = 4

embedding = torch.randn(batch_size, seq_size, dim)
print("x: ", embedding)

layer_norm = torch.nn.LayerNorm(last_dims, elementwise_affine = False)
layer_norm_out = layer_norm(embedding)
print("y: ", layer_norm_out)



