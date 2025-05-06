# Conditional DDPM for Image Generation

This repository contains an implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM) for image generation based on text conditions.

## Project Structure

- `code/`: Contains all Python code files
  - `dataset.py`: Dataset and dataloader implementations
  - `diffusion.py`: DDPM model implementation
  - `train.py`: Training and evaluation functions
  - `main.py`: Main script for running evaluation
  - `requirements.txt`: List of dependencies
- `images/`: Directory for output images

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the following files in the root directory:
   - `train.json`: Training data with labels
   - `test.json`: Test data with labels
   - `new_test.json`: New test data with labels
   - `objects.json`: Object dictionary
   - `evaluator.py`: Evaluation script
   - `checkpoint.pth`: Pretrained evaluator checkpoint

## Training

To train the model:

```bash
cd code
python train.py train --train_json train.json --train_image_dir ../iclevr --batch_size 32 --epochs 100 --lr 1e-4 --save_every 10 --generate_samples
```

Arguments:

- `--train_json`: Path to the training JSON file
- `--train_image_dir`: Path to the directory containing training images
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs to train
- `--lr`: Learning rate
- `--save_every`: Save model checkpoint every N epochs
- `--generate_samples`: Generate and save sample images during training

## Testing

To test the model:

```bash
cd code
python train.py test --checkpoint checkpoints/model_epoch_100.pth --test_json ../test.json
```

Arguments:

- `--checkpoint`: Path to the model checkpoint
- `--test_json`: Path to the test JSON file

## Visualization

To visualize the denoising process:

```bash
cd code
python train.py visualize --checkpoint checkpoints/model_epoch_100.pth
```

## Full Evaluation

To run evaluation on both test.json and new_test.json and generate all required visualizations:

```bash
cd code
python main.py --checkpoint checkpoints/model_epoch_100.pth --visualize
```

Arguments:

- `--checkpoint`: Path to the model checkpoint
- `--test_json`: Path to the test JSON file (default: test.json)
- `--new_test_json`: Path to the new test JSON file (default: new_test.json)
- `--output_dir`: Directory to save output images (default: ../images)
- `--batch_size`: Batch size for testing (default: 4)
- `--visualize`: Visualize the denoising process

## Model Architecture

The model uses a conditional U-Net architecture with:

- Time embeddings using sinusoidal positional encoding
- Condition embeddings using a simple MLP
- Skip connections between encoder and decoder
- Linear noise schedule

## Sampling Method

The model uses the standard DDPM sampling algorithm for generation, which gradually denoises an image from pure noise to a clean image based on the conditioning labels.

## Results

The model performance is evaluated using a pretrained classifier, which computes the accuracy by checking if the generated images contain the objects specified in the input labels.
