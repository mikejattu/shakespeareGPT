import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self,n_embds,head_size,block_size,dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embds, head_size, bias = False)
        self.query = nn.Linear(n_embds, head_size, bias = False)
        self.value = nn.Linear(n_embds, head_size, bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
        self.Dropoutt = nn.Dropout(p = dropout)
        self.dropoutt = dropout
    def forward(self,x):
        B,T,C = x.size()
        k = self.key(x)     # (Batch,Time,head_size)
        q = self.query(x)   # (Batch,Time,head_size)
        # calculating the scaled dot product attention
        weight = q @ k.transpose(-2,-1) * (C ** - 0.5)  # (Batch,Time,Time) # scaling the weight so that the variance is not too high
        tril = torch.tril(torch.ones(x.size(1),x.size(1))) # (Time,Time)
        weight = weight.masked_fill(self.tril[:T,:T] == 0,float('-inf')) # decoder block which prevents the model from looking into the future
        weight = F.softmax(weight, dim=-1)
        weight = self.Dropoutt(weight)
        v = self.value(x) # (Batch,Time,head_size)
        out = weight @ v
        return out

class MultiHead(nn.Module):
    def __init__(self,n_embds,head_size,block_size,n_heads,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embds,head_size,block_size,dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embds,n_embds)
        self.Dropoutt = nn.Dropout(p = dropout)
    
    def forward(self,x):
        out =  torch.cat([head(x) for head in self.heads],dim=-1)
        return self.Dropoutt(self.projection(out))
    
class FeedForward(nn.Module):
    def __init__(self,n_embds,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embds,4 * n_embds),
            nn.ReLU(),
            nn.Linear(4 * n_embds,n_embds),         # the projection layer
            nn.Dropout(p = dropout) # adding dropout
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """
    Description:This class defines the block of the transformer that 
                does the communication followed by computation.
    """
    def __init__(self,n_embds,block_size,n_heads,dropout):
        super().__init__()
        head_size = n_embds // n_heads
        self.saheads = MultiHead(n_embds,head_size,block_size,n_heads,dropout)
        self.feedforward = FeedForward(n_embds,dropout)
        # the layer normalization
        self.norm1 = nn.LayerNorm(n_embds)
        self.norm2 = nn.LayerNorm(n_embds)
    
    def forward(self,x):
        # also added residual connection
        x = x + self.saheads(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x