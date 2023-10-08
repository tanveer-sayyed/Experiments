"""
Keep in mind:
    1. shape of tensor whose gradient needs to be calculated
    2. shape of the tensor == shape of its gradient
    3. broadcasting affects gradient
    4. operations along axes affects gradients:
        - grads might accumulate
        - grads might not accumulate
        ensure that gradients flow appropriately backward, (becuase shapes change)
    5. variables, only meant for numerical stability, should have grad = 0.0
    6. braoadcast in forward pass means a sum in backward pass
    7. sum in forward pass means a broadcast in backward pass
"""

from torch import (
    all, allclose, arange, float32, ones_like,
    randint, randn, tensor, zeros_like
    )
from torch.nn.functional import cross_entropy
from typing import Tuple

from buildDataset_5 import (
    G, vocab_size, X, y
    )

ALONG_COLUMNS = 0 # helper variable
ALONG_ROWS = 1    # helper variable
BATCH_SIZE = n = 2
# our characters would be initially scattered randomly 
# within this space; this space has 2 dimensions
embedding_dim = 2
neurons_in_hidden_layer = 26
# creating the mapping to scatter the characters randomly onto this space
embeddings = randn(
    size=(vocab_size, embedding_dim),
    generator=G
    )
# get the randomly scattered embeddings of the training data
X_train = embeddings[X]
# get a random batch
idxs = randint(0, X_train.shape[0], (BATCH_SIZE,), generator=G)
X_train_batch, y_train_batch = X_train[idxs], y[idxs]


# a manual linear layer with its own weights and biases
weights = randn(
    size=(embedding_dim, neurons_in_hidden_layer), generator=G
    ) * (5/3) / (embedding_dim**0.5) # kiaming_normal with golden ration
bias = randn(size=(neurons_in_hidden_layer,), generator=G) * 0.1


# initialise parameters
parameters = [embeddings, weights, bias]
for p in parameters: p.requires_grad = True # else None


# Manual backward pass
logits = X_train_batch @ weights + bias                                     # 10.
# - postive numbers can overflow exponential, hence offseting by max
# - final probabilities are invariant to offsets
# - which means the derivatives of these offsets must be (close to) zero
logits_max_along_rows = logits.max(ALONG_ROWS, keepdims=True).values        # 9.
logits_normalised = logits - logits_max_along_rows                          # 8.
# expanding log signals to increase the distance between them
exponentiated = logits_normalised.exp()                                     # 7.
exponentiated_sum_along_rows = exponentiated.sum(ALONG_ROWS, keepdims=True) # 6.
# normalise, to get probabilities
normalised_exponents = exponentiated / exponentiated_sum_along_rows         # 5.
# select only the required ones (current probabilities in our batch)
req_normalised_exponents = normalised_exponents[arange(n), y_train_batch]   # 4.
# after getting probabilities lets return to the log world
log_probabilities = req_normalised_exponents.log()                          # 3.
# likelihood is just a scalar denoting probability, here it's mean
mean_log_likelihood = log_probabilities.mean()                              # 2.
negative_mean_log_likelihood = -mean_log_likelihood                         # 1.
# backpropogation would be initiated from: negative_mean_log_likelihood (scalar)
# based on this loss, the rearrangement of the scattered points will take place
# the gradients would direct the scattered points to their final places, eventually
assert allclose(
    cross_entropy(logits, y_train_batch), 
    negative_mean_log_likelihood
    ), "something went wrong"

### PyTorch backward pass
variables = [
        'logits',
        'logits_max_along_rows',
        'logits_normalised',
        'exponentiated',
        'exponentiated_sum_along_rows',
        'normalised_exponents',
        'req_normalised_exponents',
        'log_probabilities',
        'mean_log_likelihood',
        'negative_mean_log_likelihood',
        ]
for p in parameters: p.grad = None
for v in variables: eval(v).retain_grad()
negative_mean_log_likelihood.backward()


### Manual backward pass
def compareTheDerivatives(x:str) -> Tuple[bool, str]:
    torch_grad = eval(x).grad; our_grad = eval(x+"_global_grad")
    if all(torch_grad == our_grad).item(): return True, "exact"
    else: return (
        allclose(torch_grad, our_grad),
        "%.16f"%float((torch_grad - our_grad).abs().max().item())
        )

negative_mean_log_likelihood_global_grad = tensor([1.0]) # see chainRule_2.ipynb
compareTheDerivatives("negative_mean_log_likelihood")

# 1. negative_mean_log_likelihood = -mean_log_likelihood
mean_log_likelihood_global_grad = tensor([-1.0]) * \
                                    negative_mean_log_likelihood_global_grad
compareTheDerivatives("mean_log_likelihood")

# 2. mean_log_likelihood = log_probabilities.mean()
log_probabilities_local_grad = ones_like(log_probabilities, dtype=float32) / n
log_probabilities_global_grad = log_probabilities_local_grad * \
                                    mean_log_likelihood_global_grad
compareTheDerivatives("log_probabilities")

# 3. log_probabilities = req_normalised_exponents.log()
req_normalised_exponents_local_grad = 1/req_normalised_exponents
req_normalised_exponents_global_grad = req_normalised_exponents_local_grad * \
                                    log_probabilities_global_grad
compareTheDerivatives("req_normalised_exponents")

# 4. req_normalised_exponents = normalised_exponents[arange(n), y_train_batch]
normalised_exponents_global_grad = zeros_like(
    normalised_exponents, dtype=float32
    )
normalised_exponents_global_grad[
    arange(n), y_train_batch
    ] = req_normalised_exponents_global_grad
compareTheDerivatives("normalised_exponents")

# 5. normalised_exponents = exponentiated / exponentiated_sum_along_rows
exponentiated_sum_along_rows_global_grad = (
     -exponentiated/exponentiated_sum_along_rows**2 * normalised_exponents_global_grad
     ).sum(ALONG_ROWS, keepdim=True)
compareTheDerivatives("exponentiated_sum_along_rows")

# 5. normalised_exponents = exponentiated / exponentiated_sum_along_rows
exponentiated_local_grad_1 = 1 / exponentiated_sum_along_rows
exponentiated_global_grad_1 = exponentiated_local_grad_1 * \
                                  normalised_exponents_global_grad
# 6. exponentiated_sum_along_rows = exponentiated.sum(ALONG_ROWS, keepdims=True)
exponentiated_global_grad_2 = ones_like(exponentiated, dtype=float32) * \
                                  exponentiated_sum_along_rows_global_grad
exponentiated_global_grad = exponentiated_global_grad_1 + \
                                exponentiated_global_grad_2
compareTheDerivatives("exponentiated")

# 7. exponentiated = logits_normalised.exp()
logits_normalised_local = logits_normalised.exp()
logits_normalised_global_grad = logits_normalised_local * \
                                    exponentiated_global_grad
compareTheDerivatives("logits_normalised")

# 8. logits_normalised = logits - logits_max_along_rows
logits_max_along_rows_global_grad = -logits_normalised_global_grad.clone().sum(ALONG_ROWS, keepdim=True)
compareTheDerivatives("logits_max_along_rows")

# 8. logits_normalised = logits - logits_max_along_rows
logits_global_grad_1 = logits_normalised_global_grad.clone()
compareTheDerivatives("logits")

# 9. logits_max_along_rows = logits.max(ALONG_ROWS, keepdims=True).values
logits_global_grad_2 = zeros_like(logits, dtype=float32)
logits_global_grad_2[range(n), logits.max(ALONG_ROWS, keepdims=True).indices]


# 10. logits = X_train_batch @ weights + bias
# in a sum operation the derivatives flow equally to all its elements of the tensor
# broadcasting in forward --> sum in backward
# note that it this is only a offset added for numerical stability, hence (all close) to zero
logits_max_along_rows_global_grad = -logits_normalised_global_grad.sum(
                                        ALONG_ROWS, keepdims=True
                                        )
# 8. logits_normalised = logits - logits_max_along_rows
logits_global_grad_2 = zeros_like(
    normalised_exponents, dtype=float32
    )


