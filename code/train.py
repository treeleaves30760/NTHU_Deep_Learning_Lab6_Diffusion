import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import math
import random

from model import UNet, NoiseScheduler, DDPM
from evaluator import evaluation_model


class ICLEVRDataset(Dataset):
    """Enhanced dataset class for iCLEVR conditional generation."""

    def __init__(self, root_dir: str, json_file: str, objects_file: str,
                 transform=None, image_size: int = 64):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size

        # Load annotations
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        # Load object mappings
        with open(objects_file, 'r') as f:
            self.object_to_idx = json.load(f)

        self.num_classes = len(self.object_to_idx)
        self.filenames = list(self.annotations.keys())

        # Filter out missing files
        valid_filenames = []
        for filename in self.filenames:
            image_path = self.root_dir / filename
            if image_path.exists():
                valid_filenames.append(filename)
            else:
                print(f"Warning: Missing file {image_path}")

        self.filenames = valid_filenames
        print(f"Loaded {len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Load image
        image_path = self.root_dir / filename
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Fallback to a dummy image if file not found
            image = Image.new(
                'RGB', (self.image_size, self.image_size), color='black')

        if self.transform:
            image = self.transform(image)

        # Create condition vector
        objects = self.annotations[filename]
        condition = torch.zeros(self.num_classes)
        for obj in objects:
            if obj in self.object_to_idx:
                condition[self.object_to_idx[obj]] = 1.0

        return image, condition


def create_transforms(image_size: int = 64):
    """Create enhanced image transforms for training and validation."""
    train_transform = transforms.Compose([
        # Slightly larger for random crop
        transforms.Resize((image_size + 8, image_size + 8)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.RandomRotation(degrees=5),
        # Random erasing for better generalization
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # Normalize to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, val_transform


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


class EMA:
    """Enhanced Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.9999, update_after_step=0, update_every=1):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.shadow = {}
        self.backup = {}
        self.step = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1

        if self.step <= self.update_after_step:
            return

        if self.step % self.update_every != 0:
            return

        # Dynamic decay based on step
        decay = min(self.decay, (1 + self.step) / (10 + self.step))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + \
                    decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.step_count - self.warmup_steps) / \
                (self.total_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + \
                (1 - self.min_lr_ratio) * 0.5 * \
                (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale


def compute_enhanced_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                          timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Enhanced loss function with timestep-aware weighting."""
    # Basic MSE loss
    mse_loss = F.mse_loss(predicted_noise, target_noise, reduction='none')

    # Timestep weighting - focus more on challenging timesteps
    device = timesteps.device
    alphas_cumprod = alphas_cumprod.to(device)

    # SNR weighting (Signal-to-Noise Ratio)
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
    weight = 1.0 / (snr + 1).sqrt()  # Emphasize difficult timesteps
    weight = weight.view(-1, 1, 1, 1)

    weighted_loss = mse_loss * weight
    return weighted_loss.mean()


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1] range."""
    return (images + 1.0) / 2.0


def normalize_for_evaluator(images: torch.Tensor) -> torch.Tensor:
    """Normalize images for the evaluator (from [0, 1] to evaluator's expected range)."""
    return (images - 0.5) / 0.5


@torch.no_grad()
def generate_sample_images(model: DDPM, device: torch.device, object_to_idx: dict,
                           num_samples: int = 16, guidance_scale: float = 3.0,
                           num_inference_steps: int = 50) -> torch.Tensor:
    """Generate sample images for monitoring training progress."""
    model.eval()

    # Create diverse conditions for sampling
    conditions = []
    sample_objects = [
        ["red sphere"],
        ["blue cube"],
        ["yellow cylinder"],
        ["green sphere"],
        ["red cube", "blue sphere"],
        ["cyan cylinder", "purple cube"],
        ["yellow sphere", "green cube"],
        ["red cylinder"],
        ["blue sphere", "yellow cube"],
        ["green cylinder"],
        ["purple sphere"],
        ["cyan cube"],
        ["brown cylinder", "gray sphere"],
        ["red sphere", "blue cube", "yellow cylinder"],
        ["green sphere", "purple cube"],
        ["cyan cylinder", "brown cube"]
    ]

    # Take only the number of samples we need
    sample_objects = sample_objects[:num_samples]

    for objects in sample_objects:
        condition = torch.zeros(len(object_to_idx))
        for obj in objects:
            if obj in object_to_idx:
                condition[object_to_idx[obj]] = 1.0
        conditions.append(condition)

    # Pad with random conditions if needed
    while len(conditions) < num_samples:
        condition = torch.zeros(len(object_to_idx))
        # Randomly select 1-3 objects
        selected_objects = random.sample(
            list(object_to_idx.keys()), random.randint(1, 3))
        for obj in selected_objects:
            condition[object_to_idx[obj]] = 1.0
        conditions.append(condition)

    conditions = torch.stack(conditions).to(device)

    # Generate images with DDIM sampling
    sample_images = model.sample(
        batch_size=num_samples,
        conditions=conditions,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_ddim=True
    )

    model.train()
    return sample_images


def evaluate_samples(model: DDPM, evaluator: evaluation_model, device: torch.device,
                     test_conditions: List[torch.Tensor], num_samples: int = 32) -> float:
    """Evaluate model on test conditions."""
    model.eval()

    # Take subset of test conditions
    test_subset = test_conditions[:num_samples]
    test_subset = torch.stack(test_subset).to(device)

    # Generate images
    generated_images = model.sample(
        batch_size=len(test_subset),
        conditions=test_subset,
        device=device,
        num_inference_steps=50,
        guidance_scale=3.0,
        use_ddim=True
    )

    # Prepare for evaluator
    images_eval = denormalize_images(generated_images)
    images_eval = normalize_for_evaluator(images_eval)

    if torch.cuda.is_available():
        images_eval = images_eval.cuda()
        test_subset = test_subset.cuda()

    # Evaluate
    accuracy = evaluator.eval(images_eval, test_subset)
    model.train()
    return accuracy


def train_epoch(model: DDPM, dataloader: DataLoader, optimizer: optim.Optimizer,
                scheduler: WarmupCosineScheduler, ema: EMA, scaler: GradScaler,
                device: torch.device, epoch: int, gradient_accumulation_steps: int = 1) -> float:
    """Enhanced training for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, conditions) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast('cuda'):
            # Forward pass
            predicted_noise, target_noise = model(images, conditions)

            # Enhanced loss computation
            timesteps = torch.randint(0, model.noise_scheduler.num_train_timesteps,
                                      (images.shape[0],), device=device).long()
            loss = compute_enhanced_loss(predicted_noise, target_noise,
                                         timesteps, model.noise_scheduler.alphas_cumprod)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update learning rate
            scheduler.step()

            # Update EMA
            ema.update()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Update progress bar
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'lr': f"{current_lr:.2e}"
        })

        # Log to wandb
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            wandb.log({
                'train_loss_step': loss.item() * gradient_accumulation_steps,
                'learning_rate': current_lr,
                'epoch': epoch,
                'step': epoch * len(dataloader) + batch_idx
            })

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model: DDPM, ema: EMA, optimizer: optim.Optimizer,
                    scheduler: WarmupCosineScheduler, scaler: GradScaler,
                    epoch: int, loss: float, save_dir: str, is_best: bool = False):
    """Save enhanced model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state': {
            'step_count': scheduler.step_count,
            'warmup_steps': scheduler.warmup_steps,
            'total_steps': scheduler.total_steps,
            'base_lrs': scheduler.base_lrs
        },
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)

    # Save latest checkpoint
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

    logging.info(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(checkpoint_path: str, model: DDPM, ema: EMA,
                    optimizer: optim.Optimizer, scheduler: WarmupCosineScheduler,
                    scaler: GradScaler):
    """Load enhanced model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    ema.shadow = checkpoint['ema_shadow']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state' in checkpoint:
        scheduler_state = checkpoint['scheduler_state']
        scheduler.step_count = scheduler_state['step_count']
        scheduler.warmup_steps = scheduler_state['warmup_steps']
        scheduler.total_steps = scheduler_state['total_steps']
        scheduler.base_lrs = scheduler_state['base_lrs']

    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logging.info(f'Checkpoint loaded from epoch {epoch}')
    return epoch, loss


def load_test_conditions(json_file: str, objects_file: str) -> List[torch.Tensor]:
    """Load test conditions from JSON file."""
    with open(json_file, 'r') as f:
        test_data = json.load(f)

    with open(objects_file, 'r') as f:
        object_to_idx = json.load(f)

    conditions = []
    num_classes = len(object_to_idx)

    for objects in test_data:
        condition = torch.zeros(num_classes)
        for obj in objects:
            if obj in object_to_idx:
                condition[object_to_idx[obj]] = 1.0
        conditions.append(condition)

    return conditions


def main():
    parser = argparse.ArgumentParser(
        description='Conditional DDPM Training on iCLEVR')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing dataset')
    parser.add_argument('--image_dir', type=str,
                        default='../iclevr', help='Directory containing images')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/baseline', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int,
                        default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='Weight decay')
    parser.add_argument('--num_timesteps', type=int,
                        default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        choices=['linear', 'cosine', 'scaled_linear'], help='Noise schedule type')
    parser.add_argument('--base_channels', type=int,
                        default=32, help='Base number of channels in UNet')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_freq', type=int, default=10,
                        help='Generate samples every N epochs')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        default=2, help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int,
                        default=10, help='Number of warmup epochs')
    parser.add_argument('--guidance_scale', type=float,
                        default=3.0, help='Guidance scale for sampling')
    parser.add_argument('--wandb_run_name', type=str,
                        default='baseline', help='Custom wandb run name')
    parser.add_argument('--mixed_precision', action='store_true',
                        default=True, help='Use mixed precision training')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Setup logging
    setup_logging(args.output_dir)
    logging.info(f"Starting training with args: {args}")

    # Initialize wandb
    wandb.init(
        project='lab6_diffusion',
        name=args.wandb_run_name,
        config=vars(args)
    )

    # Create transforms
    train_transform, _ = create_transforms()

    # Create dataset and dataloader
    dataset = ICLEVRDataset(
        root_dir=args.image_dir,
        json_file=os.path.join(args.data_dir, 'train.json'),
        objects_file=os.path.join(args.data_dir, 'objects.json'),
        transform=train_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Load object mappings for sampling
    with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
        object_to_idx = json.load(f)

    # Load test conditions for evaluation
    test_conditions = load_test_conditions(
        os.path.join(args.data_dir, 'test.json'),
        os.path.join(args.data_dir, 'objects.json')
    )

    # Create enhanced model
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type
    )

    unet = UNet(
        base_channels=args.base_channels,
        num_classes=dataset.num_classes
    )

    model = DDPM(unet, noise_scheduler).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Setup enhanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # Better for transformers
        eps=1e-8
    )

    # Calculate total steps for scheduler
    total_steps = args.num_epochs * \
        len(dataloader) // args.gradient_accumulation_steps
    warmup_steps = args.warmup_epochs * \
        len(dataloader) // args.gradient_accumulation_steps

    # Setup enhanced scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01
    )

    # Setup enhanced EMA
    ema = EMA(model, decay=0.9999, update_after_step=warmup_steps)

    # Setup mixed precision
    scaler = GradScaler() if args.mixed_precision else None

    # Setup evaluator
    evaluator = evaluation_model()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0

    if args.resume:
        start_epoch, last_loss = load_checkpoint(
            args.resume, model, ema, optimizer, scheduler, scaler
        )
        start_epoch += 1

    # Create sample output directory
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    # Training loop
    logging.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, ema, scaler,
            device, epoch, args.gradient_accumulation_steps
        )

        # Log epoch metrics
        wandb.log({
            'train_loss_epoch': avg_loss,
            'epoch': epoch,
            'learning_rate': scheduler.optimizer.param_groups[0]['lr']
        })

        logging.info(
            f'Epoch {epoch}: Average Loss = {avg_loss:.6f}, LR = {scheduler.optimizer.param_groups[0]["lr"]:.2e}')

        # Generate samples and evaluate
        if (epoch + 1) % args.sample_freq == 0:
            logging.info(f"Generating samples at epoch {epoch}")

            # Apply EMA weights for sampling
            ema.apply_shadow()

            # Generate sample images
            sample_images = generate_sample_images(
                model, device, object_to_idx, num_samples=16,
                guidance_scale=args.guidance_scale
            )

            # Create grid and save
            sample_grid = make_grid(
                denormalize_images(sample_images),
                nrow=4, padding=2, normalize=False
            )
            sample_path = os.path.join(
                sample_dir, f'samples_epoch_{epoch:03d}.png')
            save_image(sample_grid, sample_path)

            # Log to wandb
            wandb.log({
                'samples': wandb.Image(sample_path, caption=f'Samples at epoch {epoch}'),
                'epoch': epoch
            })

            # Evaluate on test set
            if (epoch + 1) % args.eval_freq == 0:
                test_accuracy = evaluate_samples(
                    model, evaluator, device, test_conditions)
                wandb.log({
                    'test_accuracy': test_accuracy,
                    'epoch': epoch
                })
                logging.info(
                    f'Epoch {epoch}: Test Accuracy = {test_accuracy:.4f}')

                # Update best accuracy
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy

            # Restore original weights
            ema.restore()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, ema, optimizer, scheduler, scaler, epoch, avg_loss,
                args.output_dir, is_best
            )

    # Save final checkpoint
    save_checkpoint(
        model, ema, optimizer, scheduler, scaler, args.num_epochs - 1, avg_loss,
        args.output_dir
    )

    logging.info("Training completed!")
    logging.info(f"Best test accuracy achieved: {best_accuracy:.4f}")
    wandb.finish()


if __name__ == '__main__':
    main()
