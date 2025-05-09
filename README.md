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
python train.py train --train_json train.json --train_image_dir ../iclevr --batch_size 64 --epochs 2000 --lr 1e-4 --save_every 10 --generate_samples --schedule_type cosine --guidance_scale 3.0
```

Arguments:

- `--train_json`: Path to the training JSON file
- `--train_image_dir`: Path to the directory containing training images
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs to train
- `--lr`: Learning rate (default: 1e-4)
- `--save_every`: Save model checkpoint every N epochs
- `--generate_samples`: Generate and save sample images during training
- `--schedule_type`: Noise schedule type, either "cosine" or "linear" (default: cosine)
- `--guidance_scale`: Guidance scale for classifier-free guidance (default: 3.0)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)

## Testing

To test the model:

```bash
cd code
python train.py test --checkpoint checkpoints/model_epoch_1500.pth --test_json ./test.json --guidance_scale 3.0 --schedule_type cosine
```

Arguments:

- `--checkpoint`: Path to the model checkpoint
- `--test_json`: Path to the test JSON file
- `--guidance_scale`: Guidance scale for classifier-free guidance (default: 3.0)
- `--schedule_type`: Override noise schedule type (optional)

## Visualization

To visualize the denoising process:

```bash
cd code
python train.py visualize --checkpoint checkpoints/model_epoch_100.pth --guidance_scale 3.0
```

Arguments:
- `--checkpoint`: Path to the model checkpoint
- `--guidance_scale`: Guidance scale for classifier-free guidance (default: 3.0)
- `--schedule_type`: Override noise schedule type (optional)

## Full Evaluation

To run evaluation on both test.json and new_test.json and generate all required visualizations:

```bash
cd code
python main.py --checkpoint checkpoints/model_epoch_100.pth --visualize --guidance_scale 3.0
```

Arguments:

- `--checkpoint`: Path to the model checkpoint
- `--test_json`: Path to the test JSON file (default: test.json)
- `--new_test_json`: Path to the new test JSON file (default: new_test.json)
- `--output_dir`: Directory to save output images (default: ../images)
- `--batch_size`: Batch size for testing (default: 4)
- `--visualize`: Visualize the denoising process
- `--guidance_scale`: Guidance scale for classifier-free guidance (default: 3.0)
- `--schedule_type`: Override noise schedule type (optional)

## Model Architecture

The model uses a conditional U-Net architecture with:

- Time embeddings using sinusoidal positional encoding
- Enhanced condition embeddings with increased dimensionality
- Three middle residual blocks with attention for increased capacity
- Skip connections between encoder and decoder
- Option for either cosine or linear noise schedules
- Weight decay regularization (1e-5)

## Sampling Method

The model uses classifier-free guidance for sampling, which combines conditional and unconditional predictions to improve quality:

1. For each diffusion step, the model makes two predictions:
   - A conditional prediction based on the given label
   - An unconditional prediction with no label
   
2. These predictions are combined using the guidance scale (γ):
   - `prediction = unconditional + γ * (conditional - unconditional)`
   
3. Higher guidance scale values (3.0-5.0) produce more distinct objects that better match the conditions, though values that are too high can cause artifacts.

4. The model uses 250 denoising steps during training visualization and 1000 steps during evaluation for high-quality results.

## Results

The model performance is evaluated using a pretrained classifier, which computes the accuracy by checking if the generated images contain the objects specified in the input labels.

## Troubleshooting

If generated images appear noisy or don't show clear objects:

1. Increase the guidance scale (try values between 3.0 and 7.0)
2. Try the linear noise schedule instead of cosine
3. Increase the number of sampling steps
4. Train for more epochs with a lower learning rate
