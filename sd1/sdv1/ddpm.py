import torch
import numpy as np

# DDPM：通过马尔可夫链逐步将数据扩散为噪声，反向过程通过神经网络预测噪声并迭代去噪
# x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1-alpha_t) * ε   (ε ~ N(0,I))

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        # 先平方根后线性插值再平方
        # +-------------------+--------------------------------------------------------------+
        # |      优点          |                          解释                                |
        # +-------------------+--------------------------------------------------------------+
        # | 非线性beta调度      | beta以二次函数形式增长，早期增长慢，后期增长快，精细控制噪声添加       |
        # |                   | 早期保留更多数据信息，后期加速破坏结构                              |
        # +-------------------+--------------------------------------------------------------+
        # | 调整alpha_bar衰减   | 使alpha_bar衰减更平缓，避免指数衰减过快，中间时间步保留更多信息       |
        # |                   | 有助于模型学习中间阶段的去噪任务                                   |
        # +-------------------+--------------------------------------------------------------+
        # | 类似改进调度效果     | 类似Improved DDPM的cosine调度，避免线性调度后期噪声过大的问题       |
        # |                   | 信噪比（SNR）变化更平滑，提升生成质量                              |
        # +-------------------+--------------------------------------------------------------+
        # | 数学推导示例        | beta_start=0.0001, beta_end=0.02, T=1000                     |
        # |                   | sqrt(beta_start)=0.01, sqrt(beta_end)≈0.1414                 |
        # |                   | beta_t = (0.01 + t/T*(0.1414-0.01))^2                        |
        # |                   | 结果：beta早期增长慢，alpha_bar衰减更平缓                         |
        # +-------------------+--------------------------------------------------------------+
        # | 总结               | 通过非线性调整beta，优化噪声添加动态过程，使训练更稳定、生成质量更高    |
        # |                   | 是经验性改进与数学设计的结合                                      |
        # +-------------------+--------------------------------------------------------------+
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)   # [alpha_0, alpha_0 * alpha_1, ..., alpha_0 * alpha_1 * ... * alpha_n]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, self.num_training_timesteps)[::-1].copy())

    
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # ex. num_training_steps = 1000, num_inference_steps = 50
        # 999, 998, 997, 996, ..., 0 = 1000 steps
        # 999, 999-20, 999-40, ..., 0 = 50 steps
        # final = [999, 979, 959, 939, ..., 19]
        step_ratio = self.num_training_timesteps // self.num_inference_steps    # 1000 // 50 = 20
        timesteps = ((self.num_training_timesteps - np.arange(0, self.num_training_steps, step_ratio))[:self.num_inference_steps]).astype(np.int64)
        # timesteps = (np.arange(0, self.num_inference_steps) * step_ratio)[::-1].copy().astype(np.int64)
        # ----------------------------------------------------------------------------------------------------------------------------
        # np.arange(0, self.num_training_steps=1000, step_ratio=20)  ->  [0, 20, 40, 60, ..., 980]
        # self.num_training_steps - np.arange(0, self.num_training_steps, step_ratio) -> 1000 - [0, 20, 40, 60, ..., 980] = [1000, 980, 960, 940, ..., 20]
        # [:self.num_inference_steps]  ->  [1000, 980, 960, 940, ..., 20][:50] = [1000, 980, 960, 940, ..., 20]
        # 处理边界情况: 如果num_training_steps不能被num_inference_steps整除, 生成的序列可能会比预期的长, 如step_ratio = 990 // 50 = 19, np.arange(0, 990, 19)会生成[0, 19, 38, ..., 969, 988], 共52个元素
        self.timesteps = torch.from_numpy(timesteps)

    
    def _get_previous_timestep(self, timestep: int) -> int:     # 内部使用，只被 step 方法内部调用
        # 计算上一个时间步
        prev_timestep = timestep - self.num_training_timesteps // self.num_inference_steps
        # 确保prev_timestep不小于0
        return prev_timestep

    def _get_variance(self, timestep: int) -> torch.FloatTensor:
        # 计算方差
        # 方差的计算公式为：
        # var = (1 - α̅_{t-1}) / (1 - α̅_t) * β_t
        prev_t = self._get_previous_timestep(timestep)
        
        alpha_prod_t = self.alpha_cumprod[timestep]        # 从0到t的所有alpha的累积乘积(α̅_t)
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one 
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        # computed using formula (7) of the DDPM paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)
        return variance

    def step(self, timestep: int, latents: torch.FloatTensor, model_output: torch.FloatTensor):
        # model_output: \epsilion_{\theta}(\vec{x}_t, t) predicted noise @ timestep t
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t]        # 从0到t的所有alpha的累积乘积(α̅_t)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one     # 是从0到t-1的所有alpha的累积乘积(α̅_{t-1})
        # 在DDPM中，信噪比可以表示为：SNR = α_t / (1 - α_t)，α_t 是累积乘积alpha_cumprod，表示原始信号保留的比例
        # 在扩散过程中，我们从高噪声(低SNR)逐渐转变为低噪声(高SNR)，最终目标是恢复原始无噪声图像
        # 最后一个时间步时，这里当 prev_t < 0 时，使用 self.one （值为1.0）作为alpha值
        # SNR = 1.0 / (1 - 1.0) = 1.0 / 0 = ∞，即纯信号没有噪声
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev  # 单步的α_t = α̅_t / α̅_{t-1}
        current_beta_t = 1 - current_alpha_t                # 单步的β_t = 1 - α_t

        # formula(15) or fomula(7)[x_0 is predicted(unknown), x_t is known]
        # compute the predicted original sample using formula(15) of the DDPM paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # compute the coefficients for pred_original_sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 只有在不是最后一个时间才添加方差，如果在最后一个时间步，没有噪声，不需要添加噪声
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0, I) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, I)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    
    def set_strength(self, strength: float = 1):
        # strength: 0~1
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def add_noise(self, original_sample: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """forward: add noise for input_image
        use formula in only one step"""
        # original_sample: [batch_size, channels, height, width]
        alphas_cumprod = self.alphas_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5   # PyTorch允许使用张量作为索引，这称为"高级索引"
        # [batch_size]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()         # flatten() 操作能确保结果始终是一维的
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5   # standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # x_t = √α_t · x_{t-1} + √(1-α_t) · ε
        # Z = N(0, 1) -> N(mean, var) = X?
        # X = mean + stdev * Z
        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, dtype=original_sample.dtype)
        noisy_samples = sqrt_one_minus_alpha_prod * original_sample + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    


    





