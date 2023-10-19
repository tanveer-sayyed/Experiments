"""
    BN basically forcefully creates :
    - new normialised shift (beta) which depends on the mean(Expectation[x])
    - new normialised scale (gamma) which depends on the standard deviation
    Why? Else the issue would be that the gradient descent optimization
    will not take into account the fact that the normalization takes place.

    BN is an alternative that performs input normalization in a way
    that is differentiable and does not require the analysis of
    the entire training set after every parameter update.

    Each mini-batch produces estimates of the mean and variance of
    each activation. Gamma and Beta are inserted to be learned by the network.
    They are those transformations in the network that can represent the
    identity transform - so as to recover the original activations.
    
    The final normalised values have expected value = 0 and variance = 1.
"""

from torch import (
    allclose, arange, randint, randn
    )
from torch.nn.functional import cross_entropy
from buildDataset import (
    G, vocab_size, X, y
    )

ALONG_COLUMNS = 0 # helper variable
ALONG_ROWS = 1    # helper variable
BATCH_SIZE = n = 13

epsilon = 0.0001 # constant added for numerical stability
embedding_dim = 3
neurons_in_hidden_layer = 2
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )
weights = randn(
    size=(embedding_dim, neurons_in_hidden_layer), generator=G
    ) * (5/3) / (embedding_dim**0.5) # kiaming_normal with golden ration
bias = randn(size=(neurons_in_hidden_layer,), generator=G) * 0.1
shift_factor_beta =  randn((1, neurons_in_hidden_layer), generator=G)*0.1 + 0.1
scale_factor_gamma = randn((1, neurons_in_hidden_layer), generator=G)

# get a random batch
idxs = randint(0, X.shape[0], (BATCH_SIZE,), generator=G)
X_train_batch, y_train_batch = X[idxs], y[idxs]

#
parameters = [embeddings, scale_factor_gamma, shift_factor_beta]
for p in parameters: p.requires_grad = True # else None

# Manual forward pass
X_train_batch_embeds = embeddings[X_train_batch]
X_i = X_train_batch_embeds @ weights + bias    # layer input to the BN
# for d-dimensional input x, each dimesion (i.e. each column) would be normalised
mu_b = 1/n * X_i.sum(ALONG_COLUMNS, keepdim=True)
X_minus_mu_b = X_i - mu_b
X_minus_mu_b_squared = X_minus_mu_b**2
sigma_squared_b = 1 / (n-1) * X_minus_mu_b_squared.sum( # bassel's correction
    ALONG_COLUMNS, keepdim=True
    )
sigma_squared_b_plus_epsilon = sigma_squared_b + epsilon
sq_root_of_sigma_squared_b_plus_epsilon = sigma_squared_b_plus_epsilon ** 0.5
normalised_X_train_batch_embeds = X_minus_mu_b / sq_root_of_sigma_squared_b_plus_epsilon
y_bn = scale_factor_gamma*normalised_X_train_batch_embeds + shift_factor_beta
# return 42286
# onward 39566
