import torch
from torch import  nn
from torch.nn import functional as F
from attention import SelfAttention

class Clip(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        # vocabulary size = 49408 from files
        # embedding dimension = 768
        # maximum sequence length = 77

        self.layers = nn.Module(
            [CLIPLayer(12, 768) for _ in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # torch.LongTensor 是一个具体的张量类，用于创建新的长整型张量
        # torch.long 是一个数据类型（dtype），用于指定张量的数据类型

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim=768)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output
    
# --------------------------------------------------------------------------

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        # 将离散的词索引转换为连续的向量表示
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        # nn.Embedding 层将每个整数替换为一个向量(维度为n_embed，这里是768)

        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))  # \in 模型的可训练参数

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim=768)
        x += self.position_embedding
        # 相当于： x += self.position_embedding[:Seq_Len, :]

        return x
    
# -------------------------------------------------------------------

class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
        # 信息瓶颈设计：这种"扩展-压缩"结构增加了模型的表达能力，有助于模型学习更紧凑和有效的特征表示

    def forward(self, x: torch.tensor) -> torch.tensor:
        # (Batch_Size, Seq_Len, Dim)
        # 见原理图

        residue = x

        """SELF ATTENTION"""

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=False)

        x += residue

        """FEEDFORWARD LAYER"""

        residue = x

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        x = self.linear_2(x)

        x += residue

        return x







            




        



