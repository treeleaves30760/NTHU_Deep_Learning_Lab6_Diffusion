import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from pathlib import Path
import math

from dataset import get_dataloader, ICLEVRDataset
from diffusion import create_diffusion_model
from evaluator import evaluation_model


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("../images", exist_ok=True)

    # Create data loaders with num_workers
    train_dataloader = get_dataloader(
        json_path=args.train_json,
        image_dir=args.train_image_dir,
        batch_size=args.batch_size,
        is_train=True
    )

    # Create model and diffusion process
    model, diffusion = create_diffusion_model(
        img_size=args.img_size, device=device)

    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr * 0.1,  # Start with 10x smaller learning rate
        weight_decay=0.0001,  # Further reduced weight decay
        betas=(0.9, 0.999)
    )

    # Create learning rate scheduler with warmup
    num_warmup_steps = len(train_dataloader) * 2  # 2 epochs of warmup
    num_training_steps = len(train_dataloader) * args.epochs

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * float(current_step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize mixed precision training with dynamic loss scaling
    scaler = torch.amp.GradScaler(
        init_scale=2**10,
        growth_factor=1.1,
        backoff_factor=0.5,
        growth_interval=100
    )

    # Training loop
    best_loss = float('inf')
    global_step = 0
    accumulation_steps = 4  # Accumulate gradients for 4 steps

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader))

        for step, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, args.diffusion_steps,
                              (batch_size,), device=device).long()

            # Calculate loss with mixed precision
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = diffusion.p_losses(model, images, t, labels)
                loss = loss / accumulation_steps  # Normalize loss for accumulation

            # Check for NaN loss before backprop
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, step {step}")
                optimizer.zero_grad()
                continue

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()

            # Update weights if we've accumulated enough steps
            if (step + 1) % accumulation_steps == 0:
                # Check for NaN gradients before unscaling
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"NaN gradients detected at epoch {epoch+1}, step {step}")
                    optimizer.zero_grad()
                    continue

                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update learning rate
                scheduler.step()
                global_step += 1

            # Update epoch loss (multiply by accumulation_steps to get true loss)
            epoch_loss += loss.item() * accumulation_steps

            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Print epoch stats
        avg_loss = epoch_loss/len(train_dataloader)
        print(
            f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint if loss improved
        if avg_loss < best_loss and not torch.isnan(torch.tensor(avg_loss)):
            best_loss = avg_loss
            checkpoint_path = f"checkpoints/model_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Generate samples
            if args.generate_samples:
                generate_samples(model, diffusion, device, epoch=epoch+1)


def denormalize(images):
    """Convert normalized images back to RGB"""
    return (images * 0.5 + 0.5).clamp(0, 1)


def generate_samples(model, diffusion, device, epoch=None, n_samples=8):
    """Generate and save a grid of samples"""
    model.eval()

    # Generate condition for visualization
    # Generate one-hot encoding for "red sphere"
    with open('objects.json', 'r') as f:
        import json
        object_dict = json.load(f)

    red_sphere_idx = object_dict["red sphere"]
    condition = torch.zeros(1, 24).to(device)
    condition[0, red_sphere_idx] = 1

    # Generate samples
    samples = diffusion.sample(
        model, condition, n_samples=n_samples, n_steps=100)

    # Get final image and denormalize
    final_sample = samples[-1]
    final_sample = denormalize(final_sample)

    # Create and save grid
    grid = torchvision.utils.make_grid(final_sample, nrow=4)
    if epoch:
        save_path = f"../images/samples_epoch_{epoch}.png"
    else:
        save_path = f"../images/samples.png"

    torchvision.utils.save_image(grid, save_path)
    print(f"Samples saved to {save_path}")


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    model, diffusion = create_diffusion_model(device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, diffusion


def test(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs("../images", exist_ok=True)

    # Load model
    model, diffusion = load_model(args.checkpoint, device)
    model.eval()

    # Load evaluator
    evaluator = evaluation_model()

    # Create test data loaders
    test_dataloader = get_dataloader(
        json_path=args.test_json,
        batch_size=args.batch_size,
        is_train=False,
        shuffle=False
    )

    # Testing loop
    total_acc = 0
    all_samples = []
    all_conditions = []
    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for step, (images, labels) in progress_bar:
        labels = labels.to(device)
        batch_size = labels.shape[0]

        # Generate samples
        samples = diffusion.sample(
            model, labels, n_samples=1, n_steps=args.diffusion_steps)
        final_samples = samples[-1]  # Get the last step samples

        # Evaluate with classifier
        acc = evaluator.eval(final_samples, labels)
        total_acc += acc * batch_size

        # Store for visualization
        all_samples.append(final_samples.cpu())

        # Update progress bar
        progress_bar.set_description(f"Testing, Batch Acc: {acc:.4f}")

    # Calculate overall accuracy
    avg_acc = total_acc / len(test_dataloader.dataset)
    print(f"Test Accuracy: {avg_acc:.4f}")

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)

    # Create grid of samples
    n_rows = (all_samples.shape[0] + 7) // 8  # Ceiling division
    grid = torchvision.utils.make_grid(denormalize(all_samples), nrow=8)
    save_path = f"../images/test_samples.png"
    torchvision.utils.save_image(grid, save_path)
    print(f"Test samples saved to {save_path}")

    # Save individual samples
    for i, sample in enumerate(all_samples):
        sample_path = f"../images/test_sample_{i}.png"
        torchvision.utils.save_image(denormalize(sample), sample_path)

    return avg_acc


def visualize_denoising_process(args):
    """Visualize the denoising process for specific conditions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, diffusion = load_model(args.checkpoint, device)
    model.eval()

    # Specified conditions to visualize
    condition_labels = ["red sphere", "cyan cylinder", "cyan cube"]

    # Create dataset to get the one-hot encodings
    dataset = ICLEVRDataset(args.test_json)

    # Convert conditions to one-hot encoding
    conditions = torch.zeros(len(condition_labels), dataset.num_classes)
    for i, label in enumerate(condition_labels):
        conditions[i, dataset.object_dict[label]] = 1.0

    conditions = conditions.to(device)

    # Generate samples with intermediate steps
    sample_steps = diffusion.p_sample_loop(
        model,
        shape=(len(condition_labels), 3, args.img_size, args.img_size),
        condition=conditions,
        n_steps=args.diffusion_steps
    )

    # Select steps to visualize (8 steps from noise to clear)
    step_indices = np.linspace(0, len(sample_steps)-1, 8, dtype=int)
    steps_to_vis = [sample_steps[i] for i in step_indices]

    # Denormalize images
    steps_to_vis = [denormalize(step) for step in steps_to_vis]

    # Create figure
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    for i, condition in enumerate(condition_labels):
        for j, step_img in enumerate(steps_to_vis):
            img = step_img[i].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            if j == 0:
                axes[i, j].set_ylabel(condition)

            if i == 0:
                axes[i, j].set_title(f"Step {step_indices[j]}")

    plt.tight_layout()
    save_path = f"../images/denoising_process.png"
    plt.savefig(save_path, dpi=300)
    print(f"Denoising process visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and test DDPM")

    # Common arguments
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--diffusion_steps', type=int,
                        default=1000, help='Number of diffusion steps')

    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Train arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train_json', type=str,
                              default='train.json', help='Path to training json')
    train_parser.add_argument(
        '--train_image_dir', type=str, default='../iclevr', help='Path to training images')
    train_parser.add_argument('--batch_size', type=int,
                              default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int,
                              default=100, help='Number of epochs')
    train_parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--save_every', type=int,
                              default=10, help='Save checkpoint every N epochs')
    train_parser.add_argument(
        '--generate_samples', action='store_true', help='Generate samples during training')

    # Test arguments
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--checkpoint', type=str,
                             required=True, help='Path to model checkpoint')
    test_parser.add_argument('--test_json', type=str,
                             default='test.json', help='Path to test json')
    test_parser.add_argument('--batch_size', type=int,
                             default=32, help='Batch size')

    # Visualization arguments
    vis_parser = subparsers.add_parser(
        'visualize', help='Visualize denoising process')
    vis_parser.add_argument('--checkpoint', type=str,
                            required=True, help='Path to model checkpoint')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'visualize':
        visualize_denoising_process(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
