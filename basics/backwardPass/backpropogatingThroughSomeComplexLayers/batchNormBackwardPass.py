from torch import (
    all, allclose, arange, float32, ones_like,
    tensor, zeros_like
    )
from typing import Tuple

from batchNormForwardPass import *

def compareTheDerivatives(x:str) -> Tuple[bool, str]:
    torch_grad = eval(x).grad; our_grad = eval(x+"_global_grad")
    if all(torch_grad == our_grad).item(): return True, "exact"
    else: return (
        allclose(torch_grad, our_grad),
        "%.16f"%float((torch_grad - our_grad).abs().max().item())
        )

# copy-paste from basics/manualBackwardPass_7.ipynb BEGINS >>>
negative_mean_log_likelihood_global_grad = tensor([1.0])
mean_log_likelihood_local_grad = -1
mean_log_likelihood_global_grad = mean_log_likelihood_local_grad * \
                                    negative_mean_log_likelihood_global_grad
log_probabilities_local_grad = ones_like(log_probabilities, dtype=float32) / n
log_probabilities_global_grad = log_probabilities_local_grad * \
                                    mean_log_likelihood_global_grad
req_normalised_exponents_local_grad = 1/req_normalised_exponents
req_normalised_exponents_global_grad = req_normalised_exponents_local_grad * \
                                    log_probabilities_global_grad
normalised_exponents_global_grad = zeros_like(
    normalised_exponents, dtype=float32
    )
normalised_exponents_global_grad[
    arange(n), y_train_batch
    ] = req_normalised_exponents_global_grad
exponentiated_sum_along_rows_global_grad = (
     -exponentiated/exponentiated_sum_along_rows**2 * normalised_exponents_global_grad
     ).sum(ALONG_ROWS, keepdim=True)
exponentiated_local_grad_1 = 1 / exponentiated_sum_along_rows
exponentiated_global_grad_1 = exponentiated_local_grad_1 * \
                                  normalised_exponents_global_grad
exponentiated_global_grad_2 = ones_like(exponentiated, dtype=float32) * \
                                  exponentiated_sum_along_rows_global_grad
exponentiated_global_grad = exponentiated_global_grad_1 + \
                                exponentiated_global_grad_2
logits_normalised_local = logits_normalised.exp()
logits_normalised_global_grad = logits_normalised_local * \
                                    exponentiated_global_grad
logits_max_along_rows_global_grad = -logits_normalised_global_grad.clone().sum(
    ALONG_ROWS, keepdim=True
    )
logits_global_grad_1 = logits_normalised_global_grad.clone()
logits_global_grad_2 = zeros_like(logits, dtype=float32)
logits_global_grad_2[
    arange(n),
    logits.max(ALONG_ROWS, keepdims=True).indices
    ] = logits_max_along_rows_global_grad
logits_global_grad = logits_global_grad_1 + logits_global_grad_2
# copy-paste from basics/manualBackwardPass_7.ipynb ENDS <<<

