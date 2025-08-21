import torch
from math import sqrt


class SelfAttention(torch.nn.Module):
    def __init__(self, token_dim: int, embedding_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.w_queries = [
            torch.nn.Parameter(torch.rand(1,token_dim, embedding_size))
            for i in range(self.num_heads)
        ]
        self.w_keys = [
            torch.nn.Parameter(torch.rand(1,token_dim, embedding_size))
            for i in range(self.num_heads)
        ]
        self.w_values = [
            torch.nn.Parameter(torch.rand(1,token_dim, embedding_size))
            for i in range(self.num_heads)
        ]
        self.w_agg = torch.nn.Parameter(torch.rand(embedding_size*self.num_heads, token_dim))
        self.embedding_size = embedding_size
        self.token_dim = token_dim
            
    def forward(self, x, attention_mask):
        attention_heads = []
        for i in range(self.num_heads):
            Q = x @ self.w_queries[i] 
            K = x @ self.w_keys[i] 
            V = x @ self.w_values[i]
            attention_heads.append(
                torch.softmax(
                    ((Q @ K.transpose(2,1))) / sqrt(self.embedding_size),
                    dim=1
                ) @ V)
        multiple_heads = torch.cat(attention_heads, dim=-1)
        return multiple_heads @ self.w_agg