import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1, 2).reshape(-1, self.channels, *size)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()

        if use_attention:
            self.attention = AttentionBlock(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.norm1(self.conv1(x)))

        # Add time embedding
        time_emb = self.act1(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb

        h = self.act2(self.norm2(self.conv2(h)))

        # Apply attention if specified
        if self.use_attention:
            h = self.attention(h)

        # Residual connection
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.residual_block = ResidualBlock(
            in_ch, out_ch, time_emb_dim, use_attention)
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = self.residual_block(x, t)
        return self.downsample(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.residual_block = ResidualBlock(
            in_ch, out_ch, time_emb_dim, use_attention)
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = self.residual_block(x, t)
        return self.upsample(x)


class ConditionalUNet(nn.Module):
    """
    A conditional U-Net model that takes a noisy image and a timestep as input,
    and returns the noise added to the image.
    """

    def __init__(self, in_channels=3, model_channels=128, out_channels=3, num_classes=24,
                 time_dim=512, condition_dim=256):
        super().__init__()

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Condition embedding - INCREASED STRENGTH
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_classes, condition_dim * 2),  # Increased size
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )

        # Combined embedding dimension
        combined_dim = time_dim + condition_dim

        # Define channel dimensions for each level
        channels = [model_channels, model_channels *
                    2, model_channels*4, model_channels*8]

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Downsampling blocks
        self.down1 = DownBlock(channels[0], channels[0], combined_dim)
        self.down2 = DownBlock(
            channels[0], channels[1], combined_dim, use_attention=True)
        self.down3 = DownBlock(channels[1], channels[2], combined_dim)
        self.down4 = DownBlock(
            channels[2], channels[3], combined_dim, use_attention=True)

        # Middle blocks - ADDED ONE MORE BLOCK
        self.mid1 = ResidualBlock(
            channels[3], channels[3], combined_dim, use_attention=True)
        self.mid2 = ResidualBlock(
            channels[3], channels[3], combined_dim, use_attention=True)
        self.mid3 = ResidualBlock(
            channels[3], channels[3], combined_dim, use_attention=True)  # Added block

        # Upsampling blocks with proper skip connection channels
        self.up1 = UpBlock(channels[3] + channels[3],
                           channels[2], combined_dim, use_attention=True)
        self.up2 = UpBlock(channels[2] + channels[2],
                           channels[1], combined_dim)
        self.up3 = UpBlock(channels[1] + channels[1],
                           channels[0], combined_dim, use_attention=True)
        self.up4 = UpBlock(channels[0] + channels[0],
                           channels[0], combined_dim)

        # Final output layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t, condition):
        """
        Apply the model to an input batch.
        """
        # Get batch size
        batch_size = x.shape[0]

        # Process time embedding
        t_emb = self.time_mlp(t)
        if t_emb.shape[0] != batch_size:
            t_emb = t_emb.expand(batch_size, -1)

        # Process condition embedding
        c_emb = self.condition_mlp(condition)
        if c_emb.shape[0] != batch_size:
            c_emb = c_emb.expand(batch_size, -1)

        # Combined embedding
        emb = torch.cat([t_emb, c_emb], dim=1)

        # Initial convolution
        x0 = self.init_conv(x)

        # Downsampling path with explicit skip connections
        x1 = self.down1(x0, emb)
        x2 = self.down2(x1, emb)
        x3 = self.down3(x2, emb)
        x4 = self.down4(x3, emb)

        # Middle blocks - UPDATED WITH NEW BLOCK
        x = self.mid1(x4, emb)
        x = self.mid2(x, emb)
        x = self.mid3(x, emb)  # New block

        # Upsampling path with explicit skip connections
        x = self.up1(torch.cat([x, x4], dim=1), emb)
        x = self.up2(torch.cat([x, x3], dim=1), emb)
        x = self.up3(torch.cat([x, x2], dim=1), emb)
        x = self.up4(torch.cat([x, x1], dim=1), emb)

        # Final convolution
        return self.final_conv(x)


class GaussianDiffusion:
    """
    Gaussian diffusion model for image generation.
    """

    def __init__(self, betas, img_size=64, device="cuda"):
        self.betas = betas.to(device)
        self.img_size = img_size
        self.device = device

        # Define schedule parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1. / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * \
            torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - forward diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(
            -1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_0, t, condition, noise=None):
        """
        Calculate the loss for training
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = denoise_model(x_noisy, t, condition)
        # Ensure predicted and target noise have the same spatial dimensions
        if predicted_noise.shape[2:] != noise.shape[2:]:
            predicted_noise = F.interpolate(
                predicted_noise,
                size=noise.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Hybrid loss: combination of MSE and L1 loss
        mse_loss = F.mse_loss(predicted_noise, noise)
        l1_loss = F.l1_loss(predicted_noise, noise)

        # Dynamically adjust loss weights based on timestep
        # For earlier timesteps (smaller t), prioritize MSE loss
        # For later timesteps (larger t), gradually increase L1 weight
        t_weight = t.float() / self.betas.shape[0]
        t_weight = t_weight.view(-1, 1, 1, 1)
        hybrid_weight = 0.9 - 0.4 * t_weight.mean()  # Ranges from 0.9 to 0.5

        loss = hybrid_weight * mse_loss + (1 - hybrid_weight) * l1_loss
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, condition, guidance_scale=3.0):
        """
        Sample from p(x_{t-1} | x_t) - reverse diffusion process with classifier-free guidance
        """
        batch_size = x.shape[0]
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(
            1. / self.alphas[t]).reshape(-1, 1, 1, 1)

        # Classifier-free guidance implementation
        # Predict noise with conditional and unconditional inputs
        uncond = torch.zeros_like(condition).to(self.device)

        # Get both predictions
        noise_cond = model(x, t, condition)
        noise_uncond = model(x, t, uncond)

        # Apply guidance (combination of conditional and unconditional)
        guided_noise = noise_uncond + \
            guidance_scale * (noise_cond - noise_uncond)

        # Equation 11 in the paper with guided noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * guided_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(
                -1, 1, 1, 1)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition, n_steps=1000, guidance_scale=3.0):
        """
        Generate samples from the model using DDPM sampling with classifier-free guidance
        """
        device = next(model.parameters()).device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, n_steps)):
            img = self.p_sample(
                model,
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                condition,
                guidance_scale=guidance_scale
            )
            imgs.append(img.cpu())

        return imgs

    @torch.no_grad()
    def sample(self, model, condition, n_samples=1, n_steps=1000, guidance_scale=3.0):
        """
        Generate n_samples from the model with classifier-free guidance
        """
        # Prepare condition: ensure it's at least 2D
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        # Repeat or broadcast condition to match n_samples
        if condition.shape[0] != n_samples:
            condition = condition.expand(n_samples, -1)
        return self.p_sample_loop(
            model,
            shape=(n_samples, 3, self.img_size, self.img_size),
            condition=condition,
            n_steps=n_steps,
            guidance_scale=guidance_scale
        )


def create_cosine_schedule(n_steps=1000, s=0.008):
    """
    Create a cosine noise schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'
    """
    steps = torch.linspace(0, n_steps, n_steps + 1)
    x = steps / n_steps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def create_linear_schedule(n_steps=1000, beta_start=1e-4, beta_end=2e-2):
    """
    Create a linear beta schedule for diffusion
    """
    return torch.linspace(beta_start, beta_end, n_steps)


def create_diffusion_model(img_size=64, device="cuda", schedule_type="cosine"):
    """
    Create the UNet model and Gaussian diffusion process

    Args:
        img_size: Size of the images
        device: Device to use
        schedule_type: Type of noise schedule to use, either "cosine" or "linear"
    """
    # Create beta schedule based on the specified type
    n_steps = 1000
    if schedule_type == "cosine":
        betas = create_cosine_schedule(n_steps=n_steps).to(device)
    elif schedule_type == "linear":
        betas = create_linear_schedule(n_steps=n_steps).to(device)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Create model with improved architecture
    model = ConditionalUNet(in_channels=3,
                            model_channels=128,  # Increased from 64 to 128
                            out_channels=3,
                            num_classes=24,
                            time_dim=512,       # Increased from 256 to 512
                            condition_dim=256   # Increased from 128 to 256
                            ).to(device)

    # Create diffusion process
    diffusion = GaussianDiffusion(betas, img_size=img_size, device=device)

    return model, diffusion
