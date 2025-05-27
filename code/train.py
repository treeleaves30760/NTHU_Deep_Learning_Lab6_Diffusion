import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple

from model import UNet, NoiseScheduler, DDPM
from evaluator import evaluation_model


class ICLEVRDataset(Dataset):
    """Dataset class for iCLEVR conditional generation."""

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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Load image
        image_path = self.root_dir / filename
        try:
            image = Image.open(image_path).convert('RGB')
        except:
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
    """Create image transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
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
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                    self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1] range."""
    return (images + 1.0) / 2.0


def normalize_for_evaluator(images: torch.Tensor) -> torch.Tensor:
    """Normalize images for the evaluator (from [0, 1] to evaluator's expected range)."""
    return (images - 0.5) / 0.5


@torch.no_grad()
def generate_sample_images(model: DDPM, device: torch.device, object_to_idx: dict,
                           num_samples: int = 16, guidance_scale: float = 2.0,
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
        import random
        selected_objects = random.sample(
            list(object_to_idx.keys()), random.randint(1, 3))
        for obj in selected_objects:
            condition[object_to_idx[obj]] = 1.0
        conditions.append(condition)

    conditions = torch.stack(conditions).to(device)

    # Generate images
    sample_images = model.sample(
        batch_size=num_samples,
        conditions=conditions,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
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
        guidance_scale=2.0
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
                scheduler, ema: EMA, device: torch.device, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, conditions) in enumerate(progress_bar):
        images = images.to(device)
        conditions = conditions.to(device)

        optimizer.zero_grad()

        # Forward pass
        predicted_noise, target_noise = model(images, conditions)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, target_noise)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update EMA
        ema.update()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

        # Log to wandb
        wandb.log({
            'train_loss_step': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'step': epoch * len(dataloader) + batch_idx
        })

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model: DDPM, ema: EMA, optimizer: optim.Optimizer,
                    scheduler, epoch: int, loss: float, save_dir: str,
                    is_best: bool = False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
                    optimizer: optim.Optimizer, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    ema.shadow = checkpoint['ema_shadow']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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
        description='Train Conditional DDPM on iCLEVR')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing dataset')
    parser.add_argument('--image_dir', type=str, default='../iclevr',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2000000,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2.5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Noise schedule type')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Base number of channels in UNet')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_freq', type=int, default=20,
                        help='Generate samples every N epochs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--wandb_run_name', type=str, default='baseline',
                        help='Custom wandb run name')

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
        drop_last=True
    )

    # Load object mappings for sampling
    with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
        object_to_idx = json.load(f)

    # Load test conditions for evaluation
    test_conditions = load_test_conditions(
        os.path.join(args.data_dir, 'test.json'),
        os.path.join(args.data_dir, 'objects.json')
    )

    # Create model
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type
    )

    unet = UNet(
        base_channels=args.base_channels,
        num_classes=dataset.num_classes
    )

    model = DDPM(unet, noise_scheduler).to(device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # Setup EMA
    ema = EMA(model, decay=0.9999)

    # Setup evaluator
    evaluator = evaluation_model()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        start_epoch, last_loss = load_checkpoint(
            args.resume, model, ema, optimizer, scheduler
        )
        start_epoch += 1
        best_loss = last_loss

    # Create sample output directory
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    # Training loop
    logging.info("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, ema, device, epoch
        )

        # Log epoch metrics
        wandb.log({
            'train_loss_epoch': avg_loss,
            'epoch': epoch
        })

        logging.info(f'Epoch {epoch}: Average Loss = {avg_loss:.6f}')

        # Generate samples every sample_freq epochs
        if (epoch + 1) % args.sample_freq == 0:
            logging.info(f"Generating samples at epoch {epoch}")

            # Apply EMA weights for sampling
            ema.apply_shadow()

            # Generate sample images
            sample_images = generate_sample_images(
                model, device, object_to_idx, num_samples=16
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
            test_accuracy = evaluate_samples(
                model, evaluator, device, test_conditions)
            wandb.log({
                'test_accuracy': test_accuracy,
                'epoch': epoch
            })
            logging.info(f'Epoch {epoch}: Test Accuracy = {test_accuracy:.4f}')

            # Restore original weights
            ema.restore()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, ema, optimizer, scheduler, epoch, avg_loss,
                args.output_dir, is_best
            )

    # Save final checkpoint
    save_checkpoint(
        model, ema, optimizer, scheduler, args.num_epochs - 1, avg_loss,
        args.output_dir
    )

    logging.info("Training completed!")
    wandb.finish()


if __name__ == '__main__':
    main()
