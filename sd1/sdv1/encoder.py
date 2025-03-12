import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# - VAE不是直接将图像编码为确定的潜在向量，而是编码为概率分布
# - 这个分布通常由均值(mean)和方差(variance)参数化，通过神经网络学习得到（mean, log_variance）
# - 从这个分布中采样得到最终的潜在向量
# 噪声参数的作用 ：

# - 提供随机性：使编码过程具有随机性，每次可以生成不同的潜在表示
# - 实现重参数化：通过公式 latent = mean + std * noise 实现可导的采样过程
# - 控制采样：通过提供相同或不同的噪声，可以控制生成结果的一致性或多样性

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            # 输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2*padding) / stride + 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # 将RGB图像(3通道)转换为128通道的特征图
            # 通过两个残差块进一步提取特征，保持空间维度不变
            # 作用：初步提取图像的基本特征
            
            # 第一次下采样和特征增强
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),            # keep increasing features while decreasing images

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),
            # - 将特征图尺寸减半 Height/2, Width/2
            # - 将通道数增加到256
            # - 作用：压缩空间信息，增加特征表示能力

            # 第二次下采样和特征增强
            # each pixel represents more information, but the number of pixels is reducing at every step.

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),            # keep increasing features while decreasing images

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            # - 再次将特征图尺寸减半 Height/4, Width/4
            # - 将通道数增加到512
            # - 作用：进一步压缩空间信息，增加特征复杂度

            # 第三次下采样和深度特征提取
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            # - 最后一次空间下采样 Height/8, Width/8
            # - 保持通道数512不变，进行深度特征提取
            # - 作用：获取最终的高级特征表示

            # 注意力处理
            # run a self-attention over each pixel. attention is a way to relate tokens to each other in a sentence.
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            # - 应用自注意力机制处理全局关系
            # - 作用：捕获特征图中的长距离依赖关系

            # 最终特征规范化和压缩
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            # - 对特征进行标准化和非线性变换
            # - 将512通道压缩到8通道
            # - 作用：生成最终的潜在表示
        )    


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height / 8, Width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):   # 下采样处理 ：对于步长为2的卷积层，先进行特殊的非对称填充(右侧和底部各填充1个像素)
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)   # 防止数值溢出或下溢，确保训练稳定性

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()   # 从对数方差计算实际方差

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()     # 对方差取平方根得到标准差

        # Z = N(0, 1) -> N(mean, var) = X?
        # X = mean + stdev * Z
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise    # 使用外部提供的标准正态分布噪声进行采样，将标准正态分布转换为目标分布N(mean, variance)，x = μ + σ * ε，其中ε是标准正态分布噪声
        # 目的 ：使采样过程可导，便于反向传播
        # VAE的核心思想是通过变分推断近似后验分布p(z|x)。通过学习均值和方差参数，编码器q(z|x)可以更好地近似这个后验分布

        # Scale the output by a constant
        x *= 0.18215
        # - 将8通道分为均值和对数方差（各4通道）
        # - 通过重参数化技巧生成潜在向量
        # - 作用：实现VAE的随机采样过程，确保可导性

        return x
    
        # ## 1. 尺寸对齐问题
        # 当使用步长为2的3×3卷积进行下采样时，如果输入尺寸为奇数，直接下采样会导致尺寸不对齐。例如：

        # - 输入尺寸为5×5
        # - 不填充时，下采样后变为2×2
        # - 而我们期望的是3×3（向上取整）
        # ## 2. 信息保留
        # 非对称填充可以确保在下采样过程中不会丢失边缘信息：

        # - 传统的对称填充会在两侧都添加像素
        # - 而这里只在右侧和底部填充，更符合图像处理的直觉
        # - 这样可以保证原始图像的左上角信息被完整保留
        # ## 3. 避免棋盘效应
        # 这种填充方式有助于减少下采样过程中可能出现的棋盘效应(checkerboard artifacts)：

        # - 棋盘效应是深度学习中常见的图像处理问题
        # - 非对称填充可以减轻这种效应的产生
        # ## 4. 与上采样对称
        # 在VAE的解码器中，上采样通常使用最近邻插值，这种非对称填充方式与之形成良好的对称性：

        # - 下采样时右下填充
        # - 上采样时左上插值
        # - 这种对称性有助于更好地重建原始图像