import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        """LLMs: embeddings are the kind of vectors capturing the meaning of the word.
           Diffusion: pixels represented by many channels capture the information about the pixel.
           d_embed 参数表示嵌入维度
           - 在语言模型中：表示每个词的向量表示维度
           - 在扩散模型中：表示每个像素/位置的特征通道数
           - 决定了模型能够编码的信息量
        """
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // self.n_heads

    def forward(self, x: torch.tensor, causal_mask=False):
        # x: (Batch_Size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape  # batch_size, seq_len, _ = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 生成 Q/K/V
        qkv = self.in_proj(x)  # (Batch_Size, Seq_Len, 3*Dim)
        q, k, v = qkv.chunk(3, dim=-1)  # 分割为 Q/K/V，每个形状：(Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        """each head will watch all the sequence, but only a part of the embedding."""

        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            # 创建一个上三角掩码矩阵（防止看到未来信息）
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # 将上三角部分的权重填充为负无穷
            weight.masked_fill_(mask, -torch.inf)

        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)    ## 缩放点积
        """Batch_Size: 批量大小，表示同时处理的样本数量
        H (Heads): 注意力头的数量，每个头学习不同的注意力模式
        Query_Seq_Len: 查询序列的长度（目标序列）
        Key_Seq_Len: 键序列的长度（源序列）"""

        weight = F.softmax(weight, dim=-1)  # 归一化
        """权重矩阵结构 (Batch_Size, H, Q_Seq, K_Seq)
            ┌───────────────┐
            │ Head 1        │
            │ Q1 -> [K1 K2] │  # Q1 对 K1/K2 的注意力权重需归一化
            │ Q2 -> [K1 K2] │
            ├───────────────┤
            │ Head 2        │
            │ Q1 -> [K1 K2] │
            │ Q2 -> [K1 K2] │
            └───────────────┘
        """

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H) 
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Dim / H, Seq_Len) 
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)    # 合并多头

        output = self.out_proj(output)  # 输出投影

        # (Batch_Size, Seq_Len, Dim)
        return output
    

class CrossAttention(nn.Module):
    def  __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)  # Query projection, from x
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Key projection, from y
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Value projection, from y
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)  # Output projection
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):    # x: query, y: key & values
        # x: (latent): (Batch_Size, Seq_Len, Dim_Q)
        # y: (context): (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape   # (Batch_Size, Seq_Len_Q, Dim_Q)
        batch_size, sequence_length, d_embed = input_shape

        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        # PyTorch会根据张量的总元素数量和其他维度的大小自动计算 -1 所在维度应该的大小
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)  

        # Multiply query by Wq
        q = self.q_proj(x)    # (Batch_Size, Seq_Len_Q, Dim_Q)
        k = self.k_proj(y)    # (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)    # (Batch_Size, Seq_Len_KV, Dim_Q)

        q = q.view(interim_shape).transpose(1, 2)  # (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)  # (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)  # (Batch_Size, H, Seq_Len_KV, Dim_Q / H)

        weight = q @ k.transpose(-1, -2)  # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = weight / math.sqrt(self.d_head)  
        weight = torch.softmax(weight, dim=-1)  # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)  # Attention weights

        output = weight @ v  # (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()  # (Batch_Size, Seq_Len, H, Dim_Q / H), 合并多头
        output = output.view(input_shape)  # (Batch_Size, Seq_Len_Q, Dim_Q)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        return output
        