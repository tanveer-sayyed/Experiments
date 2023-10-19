"""
    It has been long known (LeCun et al., 1998b; Wiesler & Ney,
    2011) that the network training converges faster if its in-
    puts are whitened – i.e., linearly transformed to have zero
    means and unit variances, and decorrelated.
    
    Whitening activations could be considered at every training step or 
    at some interval, either by modifying the network directly or 
    by changing the parameters of the optimization algorithm 
    to depend on the network activation values.
    
    Whitening is an expensive operation.
    
    the full whitening of each layer’s inputs is costly and
    not everywhere differentiable
"""
