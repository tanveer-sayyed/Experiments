from math import log, sqrt
from torch import nn, tensor
import torch

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x) -> tensor:
        return self.embed(x) * sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # becuase of batching
        self.register_buffer("pe", pe) # part of state_dict yet not a parameter
    def forward(self, embeds):
        x = embeds + self.pe[:,embeds.shape[1],:].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    # single multiplicative and additive terms to amplify when needed
    def __init__(self, eps:float=1e-5) -> None:
        self.eps = eps # if denominator is almost zero then GPU cannot handle larget numbers
        self.mul = nn.Parameter(torch.ones(1))
        self.add = nn.Parameter(torch.zeros(1))
    def forward(self, x) -> tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.mul * (x-mean)/(std+self.eps) + self.add

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, dropout:float, d_ff:int) -> None:
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x) -> tensor:
        return self.fc2(self.dropout(nn.ReLU(self.fc1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, heads:int, d_model:int, dropout:float) -> None:
        self.d_model = d_model
        assert d_model % heads == 0, "d_model not divisible by heads"
        self.head = heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    @staticmethod
    def attention(q, k, v, mask, d_k, dropout):
        d_k = q.shape[-1]
        # batch, seq_len, d_k --> batch, seq_len, seq_len
        attention_scores = ((q @ k.transpose(-2, -1)) / sqrt(d_k))
        if mask is not None: attention_scores.maskedfill(mask==0, -1e-9)
        attention_scores = attention_scores.softmax(-1)
        if dropout is not None: attention_scores = dropout(attention_scores)
        return (attention_scores@v), attention_scores
    def forward(self, x, mask:tensor) -> tensor: # batch, seq_len, d_model
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        # batch, seq_len, d_model --> # batch, d_k, seq_len, d_k
        q = q.view(q.shape[0], q.shape[1], self.head, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.head, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.head, self.d_k).transpose(1, 2)
        x, attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)
        # batch, h, seq_len, d_k --> batch, seq_len, h, d_k --> batch, seq_len, d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.head*self.d_k)
        return self.w_o(x)
class ResidualBlock(nn.Module):
    def __init__(self, dropout:float) -> None:
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
    def forward(self, x, sublayer:tensor) -> tensor:
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block:MultiHeadAttention,
            feed_forward_block:FeedForwardBlock
            ):
        self.attention = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList([ResidualBlock for _ in range(2)])
    def forward(self, x):
        x = self.residual_block(x, )
        
        
