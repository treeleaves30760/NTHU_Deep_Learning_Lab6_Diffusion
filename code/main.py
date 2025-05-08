import os
import torch
import argparse
from tqdm import tqdm
import torchvision.utils as vutils

from dataset import get_dataloader
from diffusion import create_diffusion_model
from evaluator import evaluation_model
from train import load_model, denormalize, visualize_denoising_process


def run_evaluation(model, diffusion, dataloader, evaluator, device, output_dir, prefix="test", guidance_scale=3.0):
    """Run evaluation on a dataset and save images"""
    total_acc = 0
    batch_samples = []
    all_samples = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in progress_bar:
        labels = labels.to(device)
        batch_size = labels.shape[0]

        # Generate samples
        samples = diffusion.sample(
            model, labels, n_samples=1, n_steps=1000, guidance_scale=guidance_scale)
        final_samples = samples[-1]  # Get the last step

        # Evaluate accuracy
        acc = evaluator.eval(final_samples, labels)
        total_acc += acc * batch_size

        # Store for visualization
        all_samples.append(final_samples.cpu())

        # Store batches for grid visualization
        if len(batch_samples) < 32:  # Collect up to 32 samples for grid visualization
            batch_samples.append(final_samples)

        # Save individual images
        for i in range(final_samples.shape[0]):
            img_idx = step * dataloader.batch_size + i
            img_path = os.path.join(output_dir, f"{prefix}_{img_idx}.png")
            vutils.save_image(denormalize(final_samples[i]), img_path)

        progress_bar.set_description(f"Batch Acc: {acc:.4f}")

    # Combine all batch samples
    if batch_samples:
        combined_batch = torch.cat(batch_samples, dim=0)
        # Ensure we only use max 32 samples
        combined_batch = combined_batch[:32]

        # Create and save grid
        grid = vutils.make_grid(denormalize(combined_batch), nrow=8)
        grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
        vutils.save_image(grid, grid_path)

    # Calculate final accuracy
    avg_acc = total_acc / len(dataloader.dataset)
    print(f"{prefix.capitalize()} Accuracy: {avg_acc:.4f}")

    return avg_acc


def load_model(checkpoint_path, device, schedule_type=None):
    """Load model from checkpoint, with optional schedule override"""
    # If a specific schedule type is provided, use it
    if schedule_type:
        print(f"Using specified {schedule_type} noise schedule")
        model, diffusion = create_diffusion_model(
            device=device, schedule_type=schedule_type)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Use the default method that reads from checkpoint
        from train import load_model as train_load_model
        model, diffusion = train_load_model(checkpoint_path, device)

    return model, diffusion


def main():
    parser = argparse.ArgumentParser(
        description="DDPM Model for ICLEVaR Dataset")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_100.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--test_json', type=str, default='../test.json',
                        help='Path to the test JSON file')
    parser.add_argument('--new_test_json', type=str, default='../new_test.json',
                        help='Path to the new test JSON file')
    parser.add_argument('--output_dir', type=str, default='../images',
                        help='Directory to save output images')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the denoising process')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Guidance scale for classifier-free guidance')
    parser.add_argument('--schedule_type', type=str, choices=['cosine', 'linear'],
                        help='Type of noise schedule to use (overrides checkpoint setting)')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, diffusion = load_model(args.checkpoint, device, args.schedule_type)
    model.eval()

    # Load evaluator
    evaluator = evaluation_model()

    # Load test dataloaders
    test_dataloader = get_dataloader(
        args.test_json, batch_size=args.batch_size, is_train=False, shuffle=False)
    new_test_dataloader = get_dataloader(
        args.new_test_json, batch_size=args.batch_size, is_train=False, shuffle=False)

    # Run evaluation on test set
    print("Evaluating on test.json...")
    test_acc = run_evaluation(model, diffusion, test_dataloader,
                              evaluator, device, args.output_dir, prefix="test", guidance_scale=args.guidance_scale)

    # Run evaluation on new test set
    print("Evaluating on new_test.json...")
    new_test_acc = run_evaluation(
        model, diffusion, new_test_dataloader, evaluator, device, args.output_dir, prefix="new_test", guidance_scale=args.guidance_scale)

    # Print final results
    print(f"\nFinal Results:")
    print(f"test.json Accuracy: {test_acc:.4f}")
    print(f"new_test.json Accuracy: {new_test_acc:.4f}")

    # Visualize denoising process if requested
    if args.visualize:
        print("Generating denoising visualization...")

        class Args:
            def __init__(self):
                self.checkpoint = args.checkpoint
                self.test_json = args.test_json
                self.img_size = 64
                self.diffusion_steps = 1000
                self.guidance_scale = args.guidance_scale

        visualize_denoising_process(Args())


if __name__ == "__main__":
    main()
