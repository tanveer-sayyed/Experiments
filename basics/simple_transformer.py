"""
reference:
    https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    https://thegradient.pub/transformers-are-graph-neural-networks/
    http://jalammar.github.io/illustrated-transformer/
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py

    
write:
    https://docs.taichi-lang.org/blog/accelerate-python-code-100x
    
"""

from math import sin, cos, sqrt
from torch import nn, tensor, zeros, manual_seed
from random import randint
from matplotlib import pyplot as plt
from torch.autograd import Variable
from matplotlib.gridspec import GridSpec
N = 48
gs = GridSpec(33, N)
with plt.style.context(('dark_background')):
    ax1 = plt.subplot(gs[:, :int(N/3)])
    ax2 = plt.subplot(gs[:, int(2 * N/3):])
    ax3 = plt.subplot(gs[:, int(N/3):int(2 * N/3)])

# a transformer is transduction model
input_sequence = "all that is not given is lost" # input
input_sequence = {i:x for i,x in enumerate(input_sequence.split())}
embedding_dim = 2 # is model specific
seed = randint(0, 10000000000)
manual_seed(seed)
# manual_seed(1072050473)
print(seed)

# 1. For more semantic information - use more dimensions
#    to represent the data, rather than one hot encoding.
#    More the dimensions, more the computation time and 
#    more the complexity. Here let's represent numbers
#    in 2-dimesions, for simplicity
e = nn.Embedding(
    num_embeddings=len(input_sequence),
    embedding_dim=embedding_dim
    )
output = e(tensor(list(range(len(input_sequence)))))
temp = output.tolist()
print("e.weight.shape: ", e.weight.shape)
with plt.style.context(('dark_background')):
    ax1.scatter(
        x=[temp[n][0] for n in range(len(input_sequence))],
        y=[temp[n][1] for n in range(len(input_sequence))],
        marker="*",
        color="orange"
        )
    ax1.set_xticks(ticks=[], color='black')
    ax1.set_yticks(ticks=[], color='black')
    for i in range(len(input_sequence)):
        ax1.text(x=temp[i][0], y=temp[i][1], s=input_sequence[i], fontsize=8)
    ax1.grid(color='gray', linestyle='--')
    ax1.set_title("random embeddings of \nINDEX of our input", fontsize=6)

# 2.1 We see above that words are in a random order. But, sequence is
#    important. So how do we arrange the same numbers in an order which even
#    the computer understands as sequence...? We use the following method.
pe = zeros(len(input_sequence), embedding_dim)
for pos in range(len(input_sequence)):
    for i in range(0, embedding_dim, 2): 
        pe[pos, i] = \
        float(sin(pos / (10000 ** ((0.5 * i)/embedding_dim))))
        pe[pos, i + 1] = \
        float(cos(pos / (10000 ** ((0.5 * (i + 1))/embedding_dim))))
with plt.style.context(('dark_background')):
    ax2.scatter(
        x=[pe[n][0] for n in range(len(input_sequence))],
        y=[pe[n][1] for n in range(len(input_sequence))],
        marker="*",
        color="gray"
        )
    ax2.set_xticks(ticks=[], color='black')
    ax2.set_yticks(ticks=[], color='black')
    for i in range(len(input_sequence)):
        ax2.text(x=pe[i][0], y=pe[i][1], s=input_sequence[i], fontsize=8)
    ax2.grid(color='gray', linestyle='--')
    ax2.set_title("our FINAL output should look similar\nto this" + \
              "(after precise embeddings\nhave been calculated by training)", fontsize=6)
#    This will be our OUTPUT, or something similar to this.

# 2.2 Addionally, make embeddings relatively larger to increase the distance
#     between them. As weights change different numbers will gain
#     importance in each run. If you run steps - 1, 2  - multiple times,
#     you will notice that this graph changes with different initial weights.
increased_pe = output * sqrt(embedding_dim)
increased_pe = increased_pe + Variable(
    pe[:,:increased_pe.size(1)], requires_grad=False
    )
temp = increased_pe.tolist()
# if all([temp[i][1] > temp[i+1][1] for i in range(len(temp)-1)]): break
with plt.style.context(('dark_background')):
    ax3.scatter(
        x=[temp[n][0] for n in range(len(input_sequence))],
        y=[temp[n][1] for n in range(len(input_sequence))],
        marker="*",
        color="yellow"
        )
    ax3.set_xticks(ticks=[], color='black')
    ax3.set_yticks(ticks=[], color='black')
    for i in range(len(input_sequence)):
        ax3.text(x=temp[i][0], y=temp[i][1], s=input_sequence[i], fontsize=8)
    ax3.grid(color='gray', linestyle='--')
    ax3.set_title("our CURRENT output [based " + \
              "\non random embeddings(in orange)]", fontsize=6)
# plt.savefig("simple_transformer.png", dpi=200)
plt.show()
plt.close()


"""
Attention model has basically two secrets:
1. First is that they have the ability to amplify signal only from the relevant
   part of the sequence.
2. Secondly self-attention relies on comparison, that is input vector X_i:
    - is compared with every other vector to establish the weights for its own output(Y_i)
    - is compared with every other vector to establish the weights for 
keys are encoder hidden states ℎ_i
query is the single decoder hidden state s_(j−1)
attention weight a_ij is a function of two states encoder hidden state[ℎ_i] and decoder hidden
state[s_(j−1)]

alignment:
    cosine similarity == dot product
    for varying lengths of representation -->> scaled dot product
    Biased general alignment -->> focus more on some global content
    Activated general alignment -->> just add an activation layer after our nn

[https://arxiv.org/pdf/1409.0473.pdf]
An encoder neural network reads and encodes a source sen-
tence into a fixed-length vector. A decoder then outputs a translation from the encoded vector.
The whole encoder–decoder system, which consists of the encoder and the decoder for a language pair,
is jointly trained to maximize the probability of a correct translation given a source sentence.
A potential issue with this encoder–decoder approach is that a neural network needs to be able to
compress all the necessary information of a source sentence into a fixed-length vector. This may
make it difficult for the neural network to cope with long sentences, especially those that are longer
than the sentences in the training corpus. But the performance of
a basic encoder–decoder deteriorates rapidly as the length of an input sentence increases.
Hence what a transformer does is that each time the proposed model generates a word in a translation, it
(soft-)searches for a set of positions in a source sentence where the most relevant information is
concentrated. The model then predicts a target word based on the context vectors associated with
these source positions and all the previous generated target words.
Thus each new word is placed in context to the previous words already present in the same sentence.
The most important distinguishing feature of this approach from the basic encoder–decoder is that
it does not attempt to encode a whole input sentence into a single fixed-length vector.
Instead, it codes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively
while decoding the translation. This frees a neural translation model from having to squash all the
information of a source sentence, regardless of its length, into a fixed-length vector.

the alignment is not considered to be a latent variable. Instead, the alignment model directly com-
putes a soft alignment, which allows the gradient of the cost function to be backpropagated through.
This gradient can be used to train the alignment model as well as the whole translation model jointly.
Let α ij be a probability that
the target word y i is aligned to, or translated from, a source word x j . Then, the i-th context vector
c i is the expected annotation over all the annotations with probabilities α ij .
this implements a mechanism of attention in the decoder. The decoder decides parts of the source
sentence to pay attention to. By letting the decoder have an attention mechanism, we relieve the
encoder from the burden of having to encode all information in the source sentence into a fixed-
length vector.

We see strong weights along the diagonal of each matrix. However, we also
observe a number of non-trivial, non-monotonic alignments.

  The strength of the soft-alignment, opposed to a hard-alignment, is evident, for instance, from
  Fig. 3 (d). Consider the source phrase [the man] which was translated into [l’ homme]. Any hard
  alignment will map [the] to [l’] and [man] to [homme]. This is not helpful for translation, as one
  must consider the word following [the] to determine whether it should be translated into [le], [la],
  [les] or [l’]. Our soft-alignment solves this issue naturally by letting the model look at both [the] and
  [man], and in this example, we see that the model was able to correctly translate [the] into [l’]. We
  observe similar behaviors in all the presented cases in Fig. 3. An additional benefit of the soft align-
  ment is that it naturally deals with source and target phrases of different lengths, without requiring a
  counter-intuitive way of mapping some words to or from nowhere ([NULL]) (see, e.g., Chapters 4
  and 5 of Koehn, 2010).

Reverse engineering:
    - a new word requires context
    - how do we get the context?
    - so are we going bayenesian -> given a context where does the new word fit in?
    - Yes!
    - Both alignment and translation is simultaneous. Thus a transformer is trained to search
    based on probability, to predict the next word for a set of input words. The search is focused
    as the probability scores help the model focus only on the relevant information and not all information 
    at once. This is what yields good results.
    
By letting the decoder have an attention mechanism, we relieve the
encoder from the burden of having to encode all information in the source sentence into a fixed-
length vector. With this new approach the information can be spread throughout the sequence of
annotations, which can be selectively retrieved by the decoder accordingly.
    

[https://arxiv.org/pdf/1606.01933.pdf]
Attention heads decompose the task into subproblems that are solved separately.
Finally the task into subproblems that are solved separately.


The ability to amplify the signal from the relevant part of the input sequence makes attention models produce better results than models without attention. 
Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder:

V, K and Q stand for ‘key’, ‘value’ and ‘query’. These are terms used in attention functions,
but honestly, I don’t think explaining this terminology is particularly important for understanding the model.

In the case of the ENCODER, V, K and G will simply be identical copies of the embedding vector
(plus positional encoding).

Q, K, and V are batches of matrices, each with shape (batch_size, seq_length, num_features).
Multiplying the query (Q) and key (K) arrays results in a (batch_size, seq_length, seq_length) array,
which tells us roughly how important each element in the sequence is. This is the attention of
this layer — it determines which elements we “pay attention” to. 
"""

heads = 2
embeddings_per_head = embedding_dim // heads
q_linear = nn.Linear(embedding_dim, embedding_dim)
v_linear = nn.Linear(embedding_dim, embedding_dim)
k_linear = nn.Linear(embedding_dim, embedding_dim)
dropout = nn.Dropout(0.1)
out = nn.Linear(embedding_dim, embedding_dim)

# q(batch_size, len(input_sequence), embeddings_per_head)
# k
# v


# k_linear(k).view(q.size(0), -1, heads, embeddings_per_head)


