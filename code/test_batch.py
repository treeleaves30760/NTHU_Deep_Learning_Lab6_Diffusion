import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from diffusion import create_diffusion_model

# Add debug wrapper around UNet's forward method


def add_debug_wrapper(model):
    original_forward = model.forward

    def debug_forward(x, t, condition):
        print("\n--- Debug UNet Forward Pass ---")
        print(
            f"Input shapes: x={x.shape}, t={t.shape}, condition={condition.shape}")

        # Embed time
        t_embedded = model.time_mlp(t)
        print(f"Time embedding shape: {t_embedded.shape}")

        # Embed condition
        c_embedded = model.condition_mlp(condition)
        print(f"Condition embedding shape: {c_embedded.shape}")

        # Combine
        t_c = torch.cat([t_embedded, c_embedded], dim=1)
        print(f"Combined embedding shape: {t_c.shape}")

        # Initial conv
        x_conv = model.conv1(x)
        print(f"Initial conv output shape: {x_conv.shape}")

        # Cache residuals
        residuals = [x_conv]

        # Downsampling
        print("\nDownsampling blocks:")
        for i, layer in enumerate(model.downs):
            try:
                x_conv = layer(x_conv, t_c)
                print(f"  Down block {i} output shape: {x_conv.shape}")
                residuals.append(x_conv)
            except Exception as e:
                print(f"  Error in down block {i}: {e}")
                raise e

        # Middle blocks
        print("\nMiddle blocks:")
        try:
            x_conv = model.middle_block1(x_conv, t_c)
            print(f"  Middle block 1 output shape: {x_conv.shape}")
            x_conv = model.middle_block2(x_conv, t_c)
            print(f"  Middle block 2 output shape: {x_conv.shape}")
        except Exception as e:
            print(f"  Error in middle blocks: {e}")
            raise e

        # Upsampling with skip connections
        print("\nUpsampling blocks with skip connections:")
        residuals = list(reversed(residuals))
        for i, (layer, residual) in enumerate(zip(model.ups, residuals)):
            try:
                print(
                    f"  Before concat - x: {x_conv.shape}, residual: {residual.shape}")
                x_conv = torch.cat((x_conv, residual), dim=1)
                print(f"  After concat shape: {x_conv.shape}")
                x_conv = layer(x_conv, t_c)
                print(f"  Up block {i} output shape: {x_conv.shape}")
            except Exception as e:
                print(f"  Error in up block {i}: {e}")
                raise e

        # Final conv
        try:
            print(
                f"\nBefore final concat - x: {x_conv.shape}, residual: {residuals[0].shape}")
            x_conv = torch.cat((x_conv, residuals[0]), dim=1)
            print(f"After final concat shape: {x_conv.shape}")
            result = model.final_conv(x_conv)
            print(f"Final output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"Error in final conv: {e}")
            raise e

    model.forward = debug_forward
    return model


def test_batch_processing():
    print("Testing batch processing...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloader
    train_dataloader = get_dataloader(
        json_path="train.json",
        image_dir="../iclevr",
        batch_size=4,  # Use a small batch size for testing
        is_train=True
    )

    # Create model and diffusion process
    model, diffusion = create_diffusion_model(img_size=64, device=device)

    # Add debug wrapper to model
    model = add_debug_wrapper(model)

    # Get a single batch
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        print(f"Batch {batch_idx} shapes:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Sample timesteps
        batch_size = images.shape[0]
        t = torch.randint(0, 1000, (batch_size,), device=device).long()

        print(f"Timestep t shape: {t.shape}")

        # Try forward pass with noise prediction
        try:
            # Add noise to images
            noise = torch.randn_like(images)
            x_noisy = diffusion.q_sample(images, t, noise)

            print(f"Noisy images shape: {x_noisy.shape}")

            # Predict noise
            with torch.no_grad():
                print("Trying to predict noise...")
                predicted_noise = model(x_noisy, t, labels)
                print(f"Predicted noise shape: {predicted_noise.shape}")
                print("Forward pass successful!")
        except Exception as e:
            print(f"Error in forward pass: {e}")

        # Only process one batch
        break


if __name__ == "__main__":
    test_batch_processing()
