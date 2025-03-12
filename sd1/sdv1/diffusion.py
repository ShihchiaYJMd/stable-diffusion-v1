import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

# latent (1,4,64,64) ──┐
#                      │
# context (1,77,768) ──┼──► UNet ──► (1,320,64,64) ──► OutputLayer ──► (1,4,64,64)
#                      │
# time (1,320) ────────┘
#       │
#       ▼
# TimeEmbedding
#       │
#       ▼
#    (1,1280)

class Diffusion(nn.Module):
    # U-Net
    def __init__(self):
        super().__init__()
        # 320: size of time embedding
        self.time_embedding = TimeEmbedding(320) # torch.Size([1, 1280])
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent: torch.tensor, context: torch.tensor, time: torch.tensor):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)  with noise!
        # context: (Batch_Size, Seq_Len, Dim=768)
        # time: (1, 320)  current timestep

        """时间步 -> 时间嵌入, 提供噪声级别信息"""
        # (1, 320) -> (1, 1280)  
        time = self.time_embedding(time)
        # like positional encoding of transformer model = a number * sin * cos to convey information about time
        # tells model which step we arrived in the denosification process

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        output = self.final(output)      

        """predicted noise"""
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return output
    

class UNet(nn.Module):
    """使用均方误差(MSE)损失
    比较UNet预测的噪声与实际添加的噪声
    公式: L = ||ε - ε_θ(x_t, t)||², 
    ε: 真实噪声
    ε_θ: 模型预测的噪声"""
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # 下采样：减小特征图尺寸，扩大感受野
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # 通道数翻倍：补偿空间分辨率的损失，8个头，每个头处理80维的特征
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # 继续下采样，增大感受野，让模型能够捕获更大范围的上下文信息
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNet_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),

            UNet_AttentionBlock(8, 160),

            UNet_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # skip connection 2560 = 1280(bottleneck) + 1280(encoder_last_layer)
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)
        # (Batch_Size, 4, Height, Width) -> (Batch_Size, 1280, Height / 32, Width / 32)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)    # SwitchSequential takes 3 args
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's
            """梯度流动 ：跳跃连接帮助梯度更好地流动，缓解深度网络的梯度消失问题
               特征保留 ：低层特征包含重要的空间细节（如边缘、纹理），高层特征包含语义信息。跳跃连接确保这些低层细节不会在上采样过程中丢失"""
            pop_out = skip_connections.pop()  # 从后往前取，先删除list最后一个元素，并捕获它
            x = torch.cat((x, pop_out), dim=1)   # 删除元素和x拼接, 沿通道维度合并特征 {x: (Batch_Size, 4, Height, Width)}
            x = layers(x, context, time)

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.tensor):
        # x: (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        # (1, 1280)
        return x
    

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)

    
class SwitchSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()  # 正确地将参数传递给父类
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.tensor, context: torch.tensor, time: torch.tensor) -> torch.tensor:
        # 路由机制
        for layer in self.layers:
            """根据层类型选择性地传递 context"""
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    

class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        self.groupnorm = nn.GroupNorm(32, in_channels)      # 此32非x的32（样本/图片数），而是分组归一化的32（in_channels/32）
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_times=1280): # n_times: 时间嵌入的特征维度
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_times, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height / 8, Width / 8)
        # time: (1, 1280)
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        """时间信息与特征融合
        时间步信息通过这种方式在整个网络中传播，使模型能够根据不同的噪声级别调整其预测"""
        # (1, n_times=1280) -> (1, out_channels)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        # relate latent with time embedding
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
    

class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed
        
        # - GroupNorm计算过程中需要除以标准差
        # - 当标准差接近零时，添加一个小常数可以避免除零错误
        # 添加eps可以使反向传播更加稳定
        # 防止舍入误差累积导致的不稳定性
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        # - 在不增加太多计算量的情况下增强模型的表达能力
        # - 简单说，它就是一个"特征混合器"，帮助模型找到不同特征之间更好的组合方式，从而提高模型的性能
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # no conv in transformer
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        # 总共有 Height * Width 个空间位置，对应 seq_len（序列元素）
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residue_short = x   # skip connection
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x   # skip connection

        # Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)

        """交叉注意力将文本特征与图像特征关联"""
        # Cross Attention
        x = self.attention_2(x, context)

        x += residue_short

        residue_short = x   # skip connection

        # Normalization + FeedForward layer with GeGLU and skip connection

        x = self.layernorm_3(x)

        # GEGLU
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = F.gelu(gate) * x

        x = self.linear_geglu_2(x)

        x += residue_short

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long






















