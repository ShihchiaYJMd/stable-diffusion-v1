import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = HEIGHT // 8  # 64
LATENT_HEIGHT = WIDTH // 8


def generate(prompt: str, 
              uncond_prompt: str = None, # Negative prompt or empty string
              input_image=None,
              strength=0.8, do_cfg=True, cfg_scale=7.5, 
              sampler_name="ddpm", 
              n_inference_steps=50, models={}, seed=None,
              device=None,
              idle_device=None,
              tokenizer=None
            ):
    # Strength: how much attention we pay to the images' noise when generating output image, less = more noise added, more creative 
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in (0, 1]")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        # classifier free guidance(combine output)
        # output = w * (output_{conditioned} - output_{unconditioned}) + output_{unconditioned}
        # w = cfg_scale \in [1, 14], higher = more attention to the prompt

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            """'max_length'=77`: GPU计算效率, 模型架构要求, 内存管理, 批处理一致性, 预训练模型兼容性"""
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len=77, Dim=768)    # 77 because `padding='max_length'=77`
            cond_context = clip(cond_tokens)

            # Convert the uncond prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim=768)
            uncond_context = clip(uncond_tokens)

            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim) = (2, 77, 768)
            context = torch.cat([uncond_context, cond_context])

        else:
            # Convert the prompt into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
        
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)      # 更新sampler.timesteps
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # (1, 4, 64, 64)
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size=1, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # run the image throught the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)     # 再次更新了一次 sampler.timesteps 
            latents = sampler.add_noise(latents, sampler.timesteps[0])  # 这里用更新的sampler.timesteps
            
            to_idle(encoder)
        
        else:
            # If we are doing text-to-image, start with random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        # train time: 999, ..., 0
        # inference = 50: 
        # 1000, 980, 960, 940, 920, 900, 880, 860, 840, 820, 0

        timesteps = tqdm(sampler.timesteps)     # 由于99行可能更新了 sampler.timesteps，确认更新
        for _, timestep in enumerate(timesteps):
            # (timestep)int -> (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height=64, Latents_Width=64)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                #  创建两个相同的潜在表示副本: 一个用于无条件生成(使用空提示), 一个用于有条件生成(使用实际提示)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the U-Net
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            """Remove noise predicted by the U-Net"""
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def  get_time_embedding(timestep):
    # use same frequencies of cosine and sine in transformer
    # 基于Transformer中位置编码的思想，使用不同频率的正弦和余弦函数来表示不同的位置（这里是时间步），频率按照指数递减，从高频到低频
    # 输入: timestep (例如 t=5)
    # |
    # v
    # +---------------------+
    # | 创建频率向量 freqs  |
    # | [f₁, f₂, ..., f₁₆₀] |
    # +---------------------+
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # 步骤 1: 将 timestep 转换为张量            步骤 2: 添加新维度
    # [timestep] --> tensor([5.0])          [5.0] --> [[5.0]]
    # Shape: (1,)                           Shape: (**1**, 1)
    # 
    # 步骤 3: 扩展 freqs 维度                步骤 4: 广播乘法
    # freqs: [f1, f2, ..., f160]            [[5.0]] * [[f1, f2, ..., f160]]
    # Shape: (160,) --> (**1**, 160)            => [[5*f1, 5*f2, ..., 5*f160]]
    #                                       Shape: (1, 160)
    # 
    # 最终 x 的结构:
    # +------------------+------------------+-----+------------------+
    # |       5*f1       |       5*f2       | ... |      5*f160      |
    # +------------------+------------------+-----+------------------+
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)




       









        


    
    
    
    
