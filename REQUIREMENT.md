# Lab6 Code Requirements and Limitations

## Requirements

### 1. Dataset Preparation

* **Implement data loader** to read and preprocess training and testing data (`train.json`, `test.json`, `new_test.json`)
* **Define multi-label condition embedding** method to encode object labels (e.g., "red sphere", "cyan cube")

### 2. Conditional DDPM Implementation

* **Model architecture**: choose a conditional DDPM design (any architecture/library is allowed; document details and cite references)
* **Noise schedule & time embeddings**: design a noise schedule and appropriate time embedding scheme
* **Sampling method**: implement the sampling procedure; you may integrate the pretrained evaluator as classifier guidance
* **Loss functions**: select and implement loss (e.g., reparameterization tricks)
* **Training & Testing functions**:

  * Training loop with labeled conditions
  * Testing loop to generate samples under specified labels
* **Evaluation**:

  * Use the provided `evaluator.py` (ResNet‑18 based) to compute classification accuracy on `test.json` and `new_test.json`
  * Generate **synthetic image grids** for each model × each test file (8 images/row × 4 rows)
  * Produce a **denoising process grid** (≥8 steps) for label set `["red sphere", "cyan cylinder", "cyan cube"]`

## Limitations

* **Data usage**: only use provided files (no external/background images or additional data)
* **Evaluator**:

  * Do **not** modify `evaluator.py` or pretrained weights
  * You may subclass/inherit the evaluator class for extra functionality
* **Normalization**:

  * Input to evaluator must use `transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
  * Apply appropriate **denormalization** when saving generated RGB images
* **Resolution**:

  * Evaluator input resolution = **64×64**
  * You may output any resolution but **resize** to 64×64 for evaluation
* **Visualization**:

  * Use `torchvision.utils.make_grid` for image grids
  * Layouts: 8 images per row, 4 rows for testing files; at least 8 images in a row for denoising
* **Implementation reporting**:

  * Document all libraries and implementation details in your report

---
