import os
import json
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import List, Dict

from model import UNet, NoiseScheduler, DDPM
from evaluator import evaluation_model


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1] range."""
    return (images + 1.0) / 2.0


def normalize_for_evaluator(images: torch.Tensor) -> torch.Tensor:
    """Normalize images for the evaluator (from [0, 1] to evaluator's expected range)."""
    # The evaluator expects normalization with (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # which maps [0, 1] to [-1, 1]
    return (images - 0.5) / 0.5


def load_model(checkpoint_path: str, device: torch.device,
               base_channels: int = 128, num_classes: int = 24,
               num_timesteps: int = 1000, schedule_type: str = 'cosine',
               use_ema: bool = True) -> DDPM:
    """Load trained model from checkpoint."""
    # Create model
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=num_timesteps,
        schedule_type=schedule_type
    )

    unet = UNet(
        base_channels=base_channels,
        num_classes=num_classes
    )

    model = DDPM(unet, noise_scheduler).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Apply EMA weights if available and requested
    if use_ema and 'ema_shadow' in checkpoint:
        ema_shadow = checkpoint['ema_shadow']
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data = ema_shadow[name].to(device)
        logging.info("Applied EMA weights")

    model.eval()
    return model


@torch.no_grad()
def generate_images(model: DDPM, conditions: List[torch.Tensor],
                    device: torch.device, num_inference_steps: int = 50,
                    guidance_scale: float = 2.0, batch_size: int = 8) -> torch.Tensor:
    """Generate images for given conditions."""
    model.eval()

    all_images = []

    # Process conditions in batches
    for i in tqdm(range(0, len(conditions), batch_size), desc="Generating images"):
        batch_conditions = conditions[i:i + batch_size]
        batch_conditions = torch.stack(batch_conditions).to(device)

        # Generate images
        generated_images = model.sample(
            batch_size=len(batch_conditions),
            conditions=batch_conditions,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        all_images.append(generated_images.cpu())

    return torch.cat(all_images, dim=0)


@torch.no_grad()
def generate_denoising_process(model: DDPM, conditions: torch.Tensor,
                               device: torch.device, num_steps: int = 8) -> torch.Tensor:
    """Generate images showing the denoising process."""
    model.eval()

    batch_size = conditions.shape[0]
    shape = (batch_size, 3, 64, 64)

    # Start from random noise
    images = torch.randn(shape, device=device)

    # Create timesteps for visualization
    total_timesteps = model.noise_scheduler.num_train_timesteps
    step_size = total_timesteps // num_steps
    timesteps = torch.arange(total_timesteps - 1, -
                             1, -step_size, device=device).long()

    # Ensure we have exactly num_steps
    if len(timesteps) > num_steps:
        timesteps = timesteps[:num_steps]
    elif len(timesteps) < num_steps:
        timesteps = torch.cat([timesteps, torch.tensor([0], device=device)])

    denoising_images = []
    denoising_images.append(images.clone().cpu())

    for i, t in enumerate(timesteps[:-1]):
        # Predict noise
        t_batch = t.repeat(batch_size)
        predicted_noise = model.unet(images, t_batch, conditions)

        # Compute previous image
        alpha = model.noise_scheduler.alphas[t].to(device)
        alpha_cumprod = model.noise_scheduler.alphas_cumprod[t].to(device)
        beta = model.noise_scheduler.betas[t].to(device)

        next_t = timesteps[i + 1]
        if next_t > 0:
            noise = torch.randn_like(images)
            variance = model.noise_scheduler.posterior_variance[t].to(device)
        else:
            noise = 0
            variance = 0

        images = (1 / torch.sqrt(alpha)) * (images - beta /
                                            torch.sqrt(1 - alpha_cumprod) * predicted_noise)
        images = images + torch.sqrt(variance) * noise

        denoising_images.append(images.clone().cpu())

    # Shape: [batch, steps, channels, height, width]
    return torch.stack(denoising_images, dim=1)


def evaluate_accuracy(images: torch.Tensor, conditions: List[torch.Tensor],
                      evaluator: evaluation_model) -> float:
    """Evaluate accuracy using the provided evaluator."""
    # Prepare images for evaluator
    images_eval = denormalize_images(images)  # Convert to [0, 1]
    # Convert to evaluator's expected range
    images_eval = normalize_for_evaluator(images_eval)

    # Resize to 64x64 if needed
    if images_eval.shape[-1] != 64 or images_eval.shape[-2] != 64:
        images_eval = F.interpolate(images_eval, size=(
            64, 64), mode='bilinear', align_corners=False)

    # Prepare labels
    labels = torch.stack(conditions)

    # Move to cuda if available
    if torch.cuda.is_available():
        images_eval = images_eval.cuda()
        labels = labels.cuda()

    # Evaluate
    accuracy = evaluator.eval(images_eval, labels)
    return accuracy


def create_image_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Create image grid for visualization."""
    images = denormalize_images(images)  # Convert to [0, 1]
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    return grid


def create_denoising_grid(denoising_images: torch.Tensor) -> torch.Tensor:
    """Create grid showing denoising process."""
    batch_size, num_steps, channels, height, width = denoising_images.shape

    # Reshape to create grid: each row is one sample's denoising process
    images_flat = denoising_images.view(-1, channels, height, width)
    images_flat = denormalize_images(images_flat)

    grid = make_grid(images_flat, nrow=num_steps, padding=2, normalize=False)
    return grid


def main():
    parser = argparse.ArgumentParser(
        description='Test Conditional DDPM on iCLEVR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--base_channels', type=int, default=256,
                        help='Base number of channels in UNet')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Noise schedule type')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps for sampling')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA weights if available')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Setup logging
    setup_logging()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_model(
        args.checkpoint, device, args.base_channels, 24,
        args.num_timesteps, args.schedule_type, args.use_ema
    )

    # Load evaluator
    evaluator = evaluation_model()

    # Test on test.json
    logging.info("Testing on test.json")
    test_conditions = load_test_conditions(
        os.path.join(args.data_dir, 'test.json'),
        os.path.join(args.data_dir, 'objects.json')
    )

    test_images = generate_images(
        model, test_conditions, device,
        args.num_inference_steps, args.guidance_scale, args.batch_size
    )

    # Evaluate accuracy
    test_accuracy = evaluate_accuracy(test_images, test_conditions, evaluator)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")

    # Save test images grid
    test_grid = create_image_grid(test_images, nrow=8)
    save_image(test_grid, os.path.join(args.output_dir, 'test_images.png'))

    # Test on new_test.json
    logging.info("Testing on new_test.json")
    new_test_conditions = load_test_conditions(
        os.path.join(args.data_dir, 'new_test.json'),
        os.path.join(args.data_dir, 'objects.json')
    )

    new_test_images = generate_images(
        model, new_test_conditions, device,
        args.num_inference_steps, args.guidance_scale, args.batch_size
    )

    # Evaluate accuracy
    new_test_accuracy = evaluate_accuracy(
        new_test_images, new_test_conditions, evaluator)
    logging.info(f"New test accuracy: {new_test_accuracy:.4f}")

    # Save new test images grid
    new_test_grid = create_image_grid(new_test_images, nrow=8)
    save_image(new_test_grid, os.path.join(
        args.output_dir, 'new_test_images.png'))

    # Generate denoising process visualization
    logging.info("Generating denoising process visualization")

    # Create conditions for specific objects: ["red sphere", "cyan cylinder", "cyan cube"]
    with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
        object_to_idx = json.load(f)

    demo_objects = ["red sphere", "cyan cylinder", "cyan cube"]
    demo_conditions = []

    for obj in demo_objects:
        condition = torch.zeros(24)
        if obj in object_to_idx:
            condition[object_to_idx[obj]] = 1.0
        demo_conditions.append(condition)

    demo_conditions = torch.stack(demo_conditions).to(device)

    denoising_images = generate_denoising_process(
        model, demo_conditions, device, num_steps=8
    )

    # Create and save denoising grid
    denoising_grid = create_denoising_grid(denoising_images)
    save_image(denoising_grid, os.path.join(
        args.output_dir, 'denoising_process.png'))

    # Save results summary
    results = {
        'test_accuracy': test_accuracy,
        'new_test_accuracy': new_test_accuracy,
        'checkpoint': args.checkpoint,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'use_ema': args.use_ema
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {args.output_dir}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    logging.info(f"New test accuracy: {new_test_accuracy:.4f}")


if __name__ == '__main__':
    main()
