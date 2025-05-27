# Conditional DDPM on iCLEVR

This repository implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for generating 64×64 images of multi-object scenes from the iCLEVR dataset.

## Prerequisites

- Linux (tested on Ubuntu 20.04) with NVIDIA GPU and CUDA drivers
- conda
- Python 3.9+
- Git

## Setup

```bash
# Clone repository and navigate to project root
git clone <repo_url>
cd NTHU_Deep_Learning_Lab6_Diffusion

# Create and activate conda environment
conda create -n llm python=3.9 -y
conda activate llm

# Install Python dependencies
pip install -r requirements.txt
```

## Directory structure

```
.
├── code/
│   ├── train.py            # Training script
│   ├── test.py             # Testing & generation script
│   ├── run.py              # Run multiple experiments in parallel
│   ├── model.py            # Model definitions (UNet, DDPM, scheduler)
│   ├── evaluator.py        # Pretrained ResNet-18 evaluator
│   ├── train.json          # Training annotations
│   ├── test.json           # Test annotations
│   ├── new_test.json       # New test annotations
│   ├── objects.json        # Object label mapping
│   └── outputs/            # Default output folder for experiments
├── iclevr/                 # iCLEVR dataset images (RGB PNG)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Data

- Place the iCLEVR RGB images under the `iclevr/` directory.
- Annotation JSON files (`train.json`, `test.json`, `new_test.json`, `objects.json`) are in the `code/` directory by default. You can override their locations using `--data_dir` or `--image_dir` flags.

## Training

### Single experiment

```bash
cd code
python train.py \
  --data_dir . \
  --image_dir ../iclevr \
  --output_dir ./outputs/baseline \
  --batch_size 32 \
  --num_epochs 200 \
  --learning_rate 2.5e-5 \
  --weight_decay 1e-6 \
  --num_timesteps 1000 \
  --schedule_type cosine \
  --base_channels 128 \
  --save_freq 1 \
  --sample_freq 20 \
  --device cuda \
  --num_workers 4 \
  --wandb_run_name baseline
```

- Checkpoints are saved every `--save_freq` epochs to `output_dir`.
- Sample grids are saved every `--sample_freq` epochs under `output_dir/samples`.
- Training logs are written to `output_dir/train.log`.

### Multiple experiments

Use `run.py` to launch 3 predefined experiments in parallel (requires ≥3 GPUs):

```bash
cd code
# Dry run to view configurations
python run.py --data_dir . --image_dir ../iclevr --dry_run

# Launch experiments
python run.py --data_dir . --image_dir ../iclevr
```

- Logs are written to `run_experiments.log`.
- Each experiment writes to its own subfolder under `outputs/expX_*`.

## Testing & Generation

```bash
cd code
python test.py \
  --checkpoint ./outputs/my_experiment/checkpoint_epoch_019.pth \
  --data_dir . \
  --output_dir ./test_results \
  --base_channels 128 \
  --num_timesteps 1000 \
  --schedule_type cosine \
  --num_inference_steps 50 \
  --guidance_scale 2.0 \
  --batch_size 8 \
  --use_ema \
  --device cuda
```

- Generates:
  - `test_images.png` grid for `test.json`
  - `new_test_images.png` grid for `new_test.json`
  - `denoising_process.png` (denoising steps for ["red sphere", "cyan cylinder", "cyan cube"])
  - `results.json` containing accuracy scores and settings.

## Logs & Outputs

- Training logs: `<output_dir>/train.log`
- Experiment logs: `run_experiments.log`
- Model checkpoints: `<output_dir>/checkpoint_epoch_{epoch}.pth`
- Sample images: `<output_dir>/samples/*.png`
- Test results: `<test_output_dir>/test_images.png`, `new_test_images.png`, `denoising_process.png`, `results.json`

## Notes

- Ensure you are in the `llm` conda environment: `conda activate llm`.
- Adjust `--device` (`cuda`, `cpu`, or `cuda:0`) as needed.
- For best performance, use a GPU and increase `--batch_size` and `--num_workers` if memory allows. 