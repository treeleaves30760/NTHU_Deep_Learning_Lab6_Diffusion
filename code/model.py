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


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for condition integration."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.to_out(out)

        return out + residual


class SpatialTransformer(nn.Module):
    """Spatial transformer with cross-attention for better conditioning."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)

        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(channels),
                nn.MultiheadAttention(channels, num_heads, batch_first=True),
                nn.LayerNorm(channels),
                CrossAttentionBlock(channels, context_dim, num_heads),
                nn.LayerNorm(channels),
                nn.Sequential(
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels)
                )
            ]) for _ in range(1)
        ])

        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        x = self.proj_in(x)

        # Reshape for transformer
        x = x.view(batch, channels, height * width).transpose(1, 2)

        for norm1, self_attn, norm2, cross_attn, norm3, ff in self.transformer_blocks:
            # Self-attention
            x_norm = norm1(x)
            attn_out, _ = self_attn(x_norm, x_norm, x_norm)
            x = x + attn_out

            # Cross-attention
            x = x + cross_attn(norm2(x), context)

            # Feed-forward
            x = x + ff(norm3(x))

        # Reshape back
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.proj_out(x)

        return x + residual


class ResBlock(nn.Module):
    """Enhanced residual block with time and condition embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_emb_dim: int, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Main convolution layers
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Condition embedding projection
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_emb_dim, out_channels)
        )

        # Residual connection
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Optional spatial transformer for better conditioning
        if use_attention:
            self.spatial_transformer = SpatialTransformer(
                out_channels, condition_emb_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # First convolution
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # Add time and condition embeddings
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        cond_emb_proj = self.cond_proj(cond_emb)[:, :, None, None]
        x = x + time_emb + cond_emb_proj

        # Second convolution
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = x + residual

        # Apply spatial transformer if enabled
        if self.use_attention:
            # Expand condition embedding for cross-attention
            cond_expanded = cond_emb.unsqueeze(1)  # (batch, 1, emb_dim)
            x = self.spatial_transformer(x, cond_expanded)

        return x


class AttentionBlock(nn.Module):
    """Enhanced self-attention block with better normalization."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

        # Position encoding
        self.pos_emb = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        # Add positional embedding
        x = x + self.pos_emb

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(batch, self.num_heads, self.head_dim,
                   height * width).transpose(2, 3)
        k = k.view(batch, self.num_heads, self.head_dim,
                   height * width).transpose(2, 3)
        v = v.view(batch, self.num_heads, self.head_dim,
                   height * width).transpose(2, 3)

        # Compute attention with proper scaling
        scale = self.head_dim ** -0.5
        attention = torch.softmax(torch.matmul(
            q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attention, v)

        # Reshape back
        out = out.transpose(2, 3).contiguous().view(
            batch, channels, height, width)
        out = self.out(out)

        return out + residual


class ConditionEmbedding(nn.Module):
    """Enhanced embedding for multi-label conditions with better representation."""

    def __init__(self, num_classes: int = 24, emb_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Individual class embeddings
        self.class_embeddings = nn.Embedding(num_classes, emb_dim // 2)

        # Position encodings for better multi-label representation
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_classes, emb_dim // 2) * 0.02)

        # Multi-layer projection with residual connections
        self.projection = nn.Sequential(
            nn.Linear(emb_dim // 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        # Final projection for better representation
        self.final_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (batch_size, num_classes) one-hot or multi-hot
        batch_size = labels.shape[0]
        device = labels.device

        # Get all class embeddings and add positional encoding
        class_embs = self.class_embeddings.weight + \
            self.pos_encoding.squeeze(0)  # (num_classes, emb_dim//2)

        # Weighted combination based on label presence
        # (batch_size, emb_dim//2)
        weighted_emb = torch.matmul(labels.float(), class_embs)

        # Normalize by number of active classes to prevent scale issues
        num_active = labels.sum(dim=1, keepdim=True).clamp(min=1)
        weighted_emb = weighted_emb / num_active.sqrt()

        # Project to final dimension
        emb = self.projection(weighted_emb)
        emb = self.final_proj(emb)

        return emb


class UNet(nn.Module):
    """Enhanced U-Net architecture with better attention and conditioning."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, time_emb_dim: int = 256,
                 condition_emb_dim: int = 512, num_classes: int = 24):
        super().__init__()

        # Enhanced time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.condition_embedding = ConditionEmbedding(
            num_classes, condition_emb_dim)

        # Initial convolution with better initialization
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.init_conv.weight)

        # Encoder (downsampling) with progressive attention
        self.down1 = nn.ModuleList([
            ResBlock(base_channels, base_channels,
                     time_emb_dim, condition_emb_dim),
            ResBlock(base_channels, base_channels, time_emb_dim,
                     condition_emb_dim, use_attention=True),
        ])
        self.down_pool1 = nn.Conv2d(
            base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels *
                     2, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 2, base_channels * 2,
                     time_emb_dim, condition_emb_dim, use_attention=True),
        ])
        self.down_pool2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.down3 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels *
                     4, time_emb_dim, condition_emb_dim),
            ResBlock(base_channels * 4, base_channels * 4,
                     time_emb_dim, condition_emb_dim, use_attention=True),
        ])
        self.down_pool3 = nn.Conv2d(
            base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        # Enhanced bottleneck with multiple attention layers
        self.bottleneck = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 8,
                     time_emb_dim, condition_emb_dim, use_attention=True),
            AttentionBlock(base_channels * 8, num_heads=8),
            ResBlock(base_channels * 8, base_channels * 8,
                     time_emb_dim, condition_emb_dim, use_attention=True),
            AttentionBlock(base_channels * 8, num_heads=8),
            ResBlock(base_channels * 8, base_channels *
                     8, time_emb_dim, condition_emb_dim),
        ])

        # Decoder (upsampling) with attention
        self.up_conv3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2)
        self.up3 = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 4,
                     time_emb_dim, condition_emb_dim, use_attention=True),
            ResBlock(base_channels * 4, base_channels *
                     4, time_emb_dim, condition_emb_dim),
        ])

        self.up_conv2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 2,
                     time_emb_dim, condition_emb_dim, use_attention=True),
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

        # Enhanced output layers
        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # Zero initialization for output layer
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

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

        # Enhanced bottleneck
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, time_emb, cond_emb)

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
    """Enhanced noise scheduler with improved beta schedules."""

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085, beta_end: float = 0.012,
                 schedule_type: str = "cosine"):
        self.num_train_timesteps = num_train_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        elif schedule_type == "scaled_linear":
            # Better linear schedule
            self.betas = torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps)**2
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

        # For DDIM sampling
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Improved cosine schedule with better parameters."""
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
    """Enhanced Conditional Denoising Diffusion Probabilistic Model."""

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
               guidance_scale: float = 1.0, use_ddim: bool = True) -> torch.Tensor:
        """Enhanced sampling with DDIM option and better guidance."""
        # Start from random noise
        shape = (batch_size, 3, 64, 64)
        images = torch.randn(shape, device=device)

        if use_ddim:
            return self._ddim_sample(images, conditions, num_inference_steps, guidance_scale)
        else:
            return self._ddpm_sample(images, conditions, num_inference_steps, guidance_scale)

    def _ddim_sample(self, images: torch.Tensor, conditions: torch.Tensor,
                     num_inference_steps: int, guidance_scale: float) -> torch.Tensor:
        """DDIM sampling for faster and better quality generation."""
        device = images.device

        # Create timesteps for DDIM
        step_ratio = self.noise_scheduler.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long()
        timesteps = torch.flip(timesteps, [0]).to(device)

        eta = 0.0  # DDIM parameter (0 = deterministic)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(images.shape[0])

            # Predict noise
            predicted_noise = self.unet(images, t_batch, conditions)

            # Classifier-free guidance
            if guidance_scale > 1.0:
                uncond_conditions = torch.zeros_like(conditions)
                uncond_predicted_noise = self.unet(
                    images, t_batch, uncond_conditions)
                predicted_noise = uncond_predicted_noise + guidance_scale * \
                    (predicted_noise - uncond_predicted_noise)

            # DDIM step
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t].to(device)

            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[timesteps[i + 1]].to(
                    device)
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x0
            pred_x0 = (images - torch.sqrt(1 - alpha_cumprod_t) *
                       predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta **
                                2 * (1 - alpha_cumprod_t_prev)) * predicted_noise

            # Random noise component
            noise = eta * torch.sqrt(1 - alpha_cumprod_t_prev) * \
                torch.randn_like(images) if eta > 0 else 0

            # Update images
            images = torch.sqrt(alpha_cumprod_t_prev) * \
                pred_x0 + dir_xt + noise

        return torch.clamp(images, -1, 1)

    def _ddpm_sample(self, images: torch.Tensor, conditions: torch.Tensor,
                     num_inference_steps: int, guidance_scale: float) -> torch.Tensor:
        """Original DDPM sampling."""
        device = images.device

        # Create timesteps for inference
        timesteps = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0,
                                   num_inference_steps, device=device).long()

        for t in timesteps:
            # Predict noise
            t_batch = t.repeat(images.shape[0])
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
