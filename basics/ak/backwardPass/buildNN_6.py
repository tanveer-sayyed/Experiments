from torch import (
    allclose, arange, float32, ones_like, randint, randn, tensor, zeros_like
    )
from torch.nn.functional import cross_entropy

from buildDataset_5 import (
    G, vocab_size, X, y
    )

ALONG_COLUMNS = 0 # helper variable
ALONG_ROWS = 1    # helper variable
BATCH_SIZE = n = 2
# our characters would be initially scattered randomly 
# within this space; this space has 2 dimensions; (2 is actually too small)
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
logits = X_train_batch @ weights + bias
# 1. postive numbers can overflow exponential
# 2. final probabilities are invariant to offsets
# 3. which means the derivatives of these offsets must be (close to) zero
logits_max_along_rows = logits.max(ALONG_ROWS, keepdims=True).values
logits_normalised = logits - logits_max_along_rows ## <--
# expanding log signals to increase the distance between them
exponentiated = logits_normalised.exp()
exponentiated_sum_along_rows = exponentiated.sum(ALONG_ROWS, keepdims=True)
# normalise, to get probabilities
normalised_exponents = exponentiated / exponentiated_sum_along_rows
# select only the required ones (current probabilities in our batch)
req_normalised_exponents = normalised_exponents[arange(n), y_train_batch]
# after getting probabilities lets return to the log world
log_probabilities = req_normalised_exponents.log()
# likelihood is just a single number denoting probability, here it's mean
mean_log_likelihood = log_probabilities.mean()
negative_mean_log_likelihood = -mean_log_likelihood
print(cross_entropy(logits, y_train_batch), negative_mean_log_likelihood)
# based on this loss the rearrangement of the scattered points will take place

# these are the variables in our DAG
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
        'negative_mean_log_likelihood' # <-- loss
        ]

### PyTorch backward pass
for p in parameters: p.grad = None
for v in variables: eval(v).retain_grad()
negative_mean_log_likelihood.backward() # loss.backward()


### Check if manual grads == pytorch autograds
negative_mean_log_likelihood_global_grad = tensor(1.0) # see chainRule_2.ipynb
mean_log_probabilities_global_grad = tensor([-1.0]) * \
                                    negative_mean_log_likelihood_global_grad
log_probabilities_local_grad = ones_like(log_probabilities, dtype=float32) * 0.5
log_probabilities_global_grad = log_probabilities_local_grad * \
                                    mean_log_probabilities_global_grad
req_normalised_exponents_local_grad = 1/req_normalised_exponents
req_normalised_exponents_global_grad = req_normalised_exponents_local_grad * \
                                    log_probabilities_global_grad
normalised_exponents_global_grad = zeros_like(
    normalised_exponents, dtype=float32
    )
normalised_exponents_global_grad[
    arange(n), y_train_batch
    ] = req_normalised_exponents_global_grad
# global derivative must be directly multiplied before sum/broadcast
exponentiated_sum_along_rows_global_grad = (
     -exponentiated/exponentiated_sum_along_rows**2 * \
         normalised_exponents_global_grad
     ).sum(ALONG_ROWS, keepdim=True)
exponentiated_local_grad_1 = exponentiated_sum_along_rows**(-1)
# and from chainRule_2.ipynb we get
exponentiated_global_grad_1 = exponentiated_local_grad_1 * \
                                  normalised_exponents_global_grad
# in a sum operation the derivatives flow equally to all its elements of the tensor
exponentiated_global_grad_2 = ones_like(exponentiated, dtype=float32) * \
                                  exponentiated_sum_along_rows_global_grad
exponentiated_global_grad = exponentiated_global_grad_1 + \
                                exponentiated_global_grad_2
logits_normalised_local = logits_normalised.exp()
logits_normalised_global_grad = logits_normalised_local * \
                                    exponentiated_global_grad
# in a sum operation the derivatives flow equally to all its elements of the tensor
logits_global_grad_1 = logits_normalised_global_grad.clone()
# broadcasting in forward --> sum in backward
# note that it this is only a offset added for numerical stability, hence (all close) to zero
logits_max_along_rows_global_grad = -logits_normalised_global_grad.sum(
                                        ALONG_ROWS, keepdims=True
                                        )
logits_global_grad_2 = zeros_like(
    normalised_exponents, dtype=float32
    )

for v in variables[2:]: print(allclose(eval(v).grad, eval(v+"_global_grad")))

"""
logits = X_train_batch @ weights + bias

a00 a01     w00 w01 w02     a00*w00+a01*w10  a00*w01+a01*w11  a00*w02+a01*w12
a10 a11  @  w10 w11 w12  =  a10*w00+a11*w10  a10*w01+a11*w11  a10*w01+a11*w11



"""