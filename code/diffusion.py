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


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class ConditionalUNet(nn.Module):
    """
    A conditional U-Net model that takes a noisy image and a timestep as input,
    and returns the noise added to the image.
    """

    def __init__(self, in_channels=3, model_channels=64, out_channels=3, num_classes=24,
                 time_dim=256, condition_dim=128):
        super().__init__()

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Condition embedding
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_classes, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU()
        )

        # Combined embedding dimension
        combined_dim = time_dim + condition_dim

        # Initial projection
        self.conv1 = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([
            Block(model_channels, model_channels, combined_dim),
            Block(model_channels, model_channels*2, combined_dim),
            Block(model_channels*2, model_channels*4, combined_dim),
            Block(model_channels*4, model_channels*4, combined_dim)
        ])

        # Middle
        self.middle_block1 = Block(
            model_channels*4, model_channels*4, combined_dim)
        self.middle_block2 = Block(
            model_channels*4, model_channels*4, combined_dim)

        # Upsampling: use full concatenated channel count for each Block
        self.ups = nn.ModuleList([
            # block0: concat h(256) + skip3(256) = 512 in, out 256
            Block(model_channels*4 + model_channels*4,
                  model_channels*4, combined_dim, up=True),
            # block1: concat h(256) + skip2(128) = 384 in, out 128
            Block(model_channels*4 + model_channels*2,
                  model_channels*2, combined_dim, up=True),
            # block2: concat h(128) + skip1(64)  = 192 in, out 64
            Block(model_channels*2 + model_channels,
                  model_channels, combined_dim, up=True),
            # block3: concat h(64) + skip0(64)   = 128 in, out 64
            Block(model_channels + model_channels,
                  model_channels, combined_dim, up=True)
        ])

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t, condition):
        """
        Apply the model to an input batch.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            t: Time embedding tensor of shape [batch_size]
            condition: Conditional embedding tensor of shape [batch_size, num_classes]

        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Process time embedding
        t_emb = self.time_mlp(t)

        # Process condition embedding
        c_emb = self.condition_mlp(condition)

        # Combine time and condition embeddings
        temb_cond = torch.cat([t_emb, c_emb], dim=1)

        # Initial conv
        h = self.conv1(x)

        # Store skip connections
        skips = [h]

        # Downsample
        for i, down_block in enumerate(self.downs):
            h = down_block(h, temb_cond)
            if i < len(self.downs) - 1:  # Don't add last layer to skip connections
                skips.append(h)

        # Middle
        h = self.middle_block1(h, temb_cond)
        h = self.middle_block2(h, temb_cond)

        # Upsample with skip connections
        for i, up_block in enumerate(self.ups):
            # Ensure we're not out of bounds with skip connections
            skip_idx = len(skips) - i - 1
            if skip_idx >= 0:
                skip = skips[skip_idx]

                # Ensure spatial dimensions match
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='nearest')

                # Ensure batch dimensions match
                if h.shape[0] != skip.shape[0]:
                    # Use the smaller batch size
                    min_batch_size = min(h.shape[0], skip.shape[0])
                    h = h[:min_batch_size]
                    skip = skip[:min_batch_size]

                # Concatenate current feature map with skip connection
                h = torch.cat([h, skip], dim=1)

            # Apply up block
            h = up_block(h, temb_cond)

        # Final convolution
        output = self.final_conv(h)

        return output


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

        # Simple MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, condition):
        """
        Sample from p(x_{t-1} | x_t) - reverse diffusion process
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(
            1. / self.alphas[t]).reshape(-1, 1, 1, 1)

        # Equation 11 in the paper
        # Use predicted noise to compute x_0
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, condition) /
            sqrt_one_minus_alphas_cumprod_t
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
    def p_sample_loop(self, model, shape, condition, n_steps=1000):
        """
        Generate samples from the model using DDPM sampling
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
                condition
            )
            imgs.append(img.cpu())

        return imgs

    @torch.no_grad()
    def sample(self, model, condition, n_samples=1, n_steps=1000):
        """
        Generate n_samples from the model
        """
        return self.p_sample_loop(
            model,
            shape=(n_samples, 3, self.img_size, self.img_size),
            condition=condition,
            n_steps=n_steps
        )


def create_diffusion_model(img_size=64, device="cuda"):
    """
    Create the UNet model and Gaussian diffusion process
    """
    # Create beta schedule
    n_steps = 1000
    betas = torch.linspace(1e-4, 0.02, n_steps).to(device)

    # Create model
    model = ConditionalUNet(in_channels=3, out_channels=3,
                            num_classes=24).to(device)

    # Create diffusion process
    diffusion = GaussianDiffusion(betas, img_size=img_size, device=device)

    return model, diffusion
