import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time and condition embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Main convolution layers
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # Condition embedding projection
        self.cond_proj = nn.Linear(condition_emb_dim, out_channels)

        # Residual connection
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # First convolution
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # Add time and condition embeddings
        time_emb = F.silu(self.time_proj(time_emb))[:, :, None, None]
        cond_emb = F.silu(self.cond_proj(cond_emb))[:, :, None, None]
        x = x + time_emb + cond_emb

        # Second convolution
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block for better feature learning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention computation
        q = q.view(batch, channels, height * width).transpose(1, 2)
        k = k.view(batch, channels, height * width).transpose(1, 2)
        v = v.view(batch, channels, height * width).transpose(1, 2)

        # Compute attention
        scale = 1.0 / math.sqrt(channels)
        attention = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attention, v)

        # Reshape back
        out = out.transpose(1, 2).view(batch, channels, height, width)
        out = self.out(out)

        return out + residual


class ConditionEmbedding(nn.Module):
    """Embedding for multi-label conditions."""

    def __init__(self, num_classes: int = 24, emb_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Multi-label embedding with projection
        self.embedding = nn.Embedding(num_classes, emb_dim // 4)
        self.projection = nn.Sequential(
            nn.Linear(emb_dim // 4, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.SiLU()
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (batch_size, num_classes) one-hot or multi-hot
        batch_size = labels.shape[0]

        # Get embeddings for all classes
        all_embeddings = self.embedding.weight  # (num_classes, emb_dim//4)

        # Weighted sum based on label presence
        # (batch_size, emb_dim//4)
        weighted_emb = torch.matmul(labels.float(), all_embeddings)

        # Project to final dimension
        return self.projection(weighted_emb)


class UNet(nn.Module):
    """U-Net architecture for conditional DDPM."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, time_emb_dim: int = 256,
                 condition_emb_dim: int = 512, num_classes: int = 24):
        super().__init__()

        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)
        self.condition_embedding = ConditionEmbedding(
            num_classes, condition_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling)
        self.down1 = nn.ModuleList([
            ResBlock(base_channels, base_channels,
                     time_emb_dim, condition_emb_dim),
            ResBlock(base_channels, base_channels,
                     time_emb_dim, condition_emb_dim),
        ])
        self.down_pool1 = nn.Conv2d(
            base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels *
                     2, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 2, base_channels *
                     2, time_emb_dim, condition_emb_dim),
        ])
        self.down_pool2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.down3 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels *
                     4, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 4, base_channels *
                     4, time_emb_dim, condition_emb_dim),
        ])
        self.down_pool3 = nn.Conv2d(
            base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        # Bottleneck with attention
        self.bottleneck = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels *
                     8, time_emb_dim, condition_emb_dim),
            AttentionBlock(base_channels * 8),
            ResBlock(base_channels * 8, base_channels *
                     8, time_emb_dim, condition_emb_dim),
        ])

        # Decoder (upsampling)
        self.up_conv3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2)
        self.up3 = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels *
                     4, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 4, base_channels *
                     4, time_emb_dim, condition_emb_dim),
        ])

        self.up_conv2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels *
                     2, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 2, base_channels *
                     2, time_emb_dim, condition_emb_dim),
        ])

        self.up_conv1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2)
        self.up1 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels,
                     time_emb_dim, condition_emb_dim),
            ResBlock(base_channels, base_channels,
                     time_emb_dim, condition_emb_dim),
        ])

        # Output layers
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        # Embeddings
        time_emb = self.time_embedding(time)
        cond_emb = self.condition_embedding(condition)

        # Initial convolution
        x = self.init_conv(x)

        # Encoder
        skip_connections = []

        # Down 1
        skip_connections.append(x)
        for block in self.down1:
            x = block(x, time_emb, cond_emb)
        x = self.down_pool1(x)

        # Down 2
        skip_connections.append(x)
        for block in self.down2:
            x = block(x, time_emb, cond_emb)
        x = self.down_pool2(x)

        # Down 3
        skip_connections.append(x)
        for block in self.down3:
            x = block(x, time_emb, cond_emb)
        x = self.down_pool3(x)

        # Bottleneck
        x = self.bottleneck[0](x, time_emb, cond_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, time_emb, cond_emb)

        # Decoder
        # Up 3
        x = self.up_conv3(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for block in self.up3:
            x = block(x, time_emb, cond_emb)

        # Up 2
        x = self.up_conv2(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for block in self.up2:
            x = block(x, time_emb, cond_emb)

        # Up 1
        x = self.up_conv1(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for block in self.up1:
            x = block(x, time_emb, cond_emb)

        # Output
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)

        return x


class NoiseScheduler:
    """Noise scheduler for DDPM with linear or cosine schedule."""

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02,
                 schedule_type: str = "linear"):
        self.num_train_timesteps = num_train_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to the original samples according to the noise schedule."""
        device = original_samples.device

        # Move scheduler tensors to the same device as inputs
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device)

        sqrt_alpha_prod = sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps].reshape(
            -1, 1, 1, 1)

        noisy_samples = sqrt_alpha_prod * original_samples + \
            sqrt_one_minus_alpha_prod * noise
        return noisy_samples


class DDPM(nn.Module):
    """Conditional Denoising Diffusion Probabilistic Model."""

    def __init__(self, unet: UNet, noise_scheduler: NoiseScheduler):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler

    def forward(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = images.shape[0]
        device = images.device

        # Sample random timesteps
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                  (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Predict noise
        predicted_noise = self.unet(noisy_images, timesteps, conditions)

        return predicted_noise, noise

    @torch.no_grad()
    def sample(self, batch_size: int, conditions: torch.Tensor,
               device: torch.device, num_inference_steps: int = 50,
               guidance_scale: float = 1.0) -> torch.Tensor:
        """Sample images from the model."""
        # Start from random noise
        shape = (batch_size, 3, 64, 64)
        images = torch.randn(shape, device=device)

        # Create timesteps for inference
        timesteps = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0,
                                   num_inference_steps, device=device).long()

        for t in timesteps:
            # Predict noise
            t_batch = t.repeat(batch_size)
            predicted_noise = self.unet(images, t_batch, conditions)

            # Classifier-free guidance
            if guidance_scale > 1.0:
                uncond_conditions = torch.zeros_like(conditions)
                uncond_predicted_noise = self.unet(
                    images, t_batch, uncond_conditions)
                predicted_noise = uncond_predicted_noise + guidance_scale * \
                    (predicted_noise - uncond_predicted_noise)

            # Compute previous image
            alpha = self.noise_scheduler.alphas[t].to(device)
            alpha_cumprod = self.noise_scheduler.alphas_cumprod[t].to(device)
            beta = self.noise_scheduler.betas[t].to(device)

            if t > 0:
                noise = torch.randn_like(images)
                variance = self.noise_scheduler.posterior_variance[t].to(
                    device)
            else:
                noise = torch.zeros_like(images)
                variance = torch.tensor(0.0, device=device)

            images = (1 / torch.sqrt(alpha)) * (images - beta /
                                                torch.sqrt(1 - alpha_cumprod) * predicted_noise)
            images = images + torch.sqrt(variance) * noise

        return torch.clamp(images, -1, 1)
