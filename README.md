# Super-Resolution Project: SRCNN, SRGAN, and Attentive ESRGAN

## Overview

This project implements and compares three deep learning models for single-image super-resolution (SISR):

1. **SRCNN Baseline** - A PSNR-oriented convolutional neural network with 3 layers
2. **SRGAN Baseline** - A generative adversarial network using SRResNet generator with BatchNorm, Binary Cross-Entropy loss, and VGG perceptual loss
3. **Attentive ESRGAN** - An enhanced super-resolution GAN with channel attention, no BatchNorm, and Relativistic average Least Squares GAN (RaLSGAN) loss

The goal is to upscale low-resolution images by a factor of 4x while maintaining or improving perceptual quality compared to traditional bicubic interpolation.

## Authors

- A. B. Bahi
- Felix Floberg
- Simon Que
- 黎裴方東

## Prerequisites

### Coding Environment

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows (tested on Kaggle notebooks)
- **GPU**: Recommended for training (CUDA-compatible GPU with at least 8GB VRAM)

### Package Versions

Install the required packages using:

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.18.0
- NumPy >= 1.21.0
- OpenCV-Python >= 4.5.0
- Matplotlib >= 3.5.0
- Protobuf 3.20.*

## Dataset

This project uses the **DIV2K** dataset:
- Training set: 800 high-resolution images
- Validation set: 100 high-resolution images

The dataset should be organized as follows:
```
DIV2K_root/
├── DIV2K_train_HR/
│   └── DIV2K_train_HR/
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
└── DIV2K_valid_HR/
    └── DIV2K_valid_HR/
        ├── 0801.png
        ├── 0802.png
        └── ...
```

## Usage

### 1. Setup

Open the notebook `dip-final-project-srgan.ipynb` in your environment (Jupyter, Kaggle, or Google Colab).

### 2. Configure Paths

Update the following paths in the notebook:

```python
DIV2K_ROOT = "/path/to/div2k-high-resolution-images"
SRCNN_PRETRAINED_PATH = "/path/to/srcnn_baseline_model.h5"
SRRESNET_WARMUP_PATH = "/path/to/srresnet_warmup.h5"
ATTENTIVE_WARMUP_PATH = "/path/to/attentive_generator_warmup.h5"
```

### 3. Training

#### SRCNN Baseline
```python
srcnn = build_srcnn()
# Train with MSE loss
srcnn.fit(train_data, epochs=100, ...)
```

#### SRGAN Baseline
```python
srgan_gen = build_srgan_generator(scale=4, num_res_blocks=16)
srgan_disc = build_srgan_discriminator(input_shape=(128, 128, 3))
vgg = build_vgg(hr_shape=(128, 128, 3))

srgan_history = train_srgan_baseline(
    generator=srgan_gen,
    discriminator=srgan_disc,
    srgan=srgan_combined,
    vgg=vgg,
    train_loader=train_gen,
    val_loader=val_gen,
    epochs=30,
    steps_per_epoch=50
)
```

#### Attentive ESRGAN
```python
att_gen = build_attentive_generator(scale=4, num_res_blocks=16)
att_disc = build_relativistic_discriminator(input_shape=(128, 128, 3))
vgg_esr = build_vgg(hr_shape=(128, 128, 3))

att_history = train_attentive_esrgan(
    generator=att_gen,
    discriminator=att_disc,
    vgg=vgg_esr,
    train_loader=train_gen,
    val_loader=val_gen,
    epochs=100,
    steps_per_epoch=50
)
```

### 4. Inference

Evaluate models on test images:

```python
# SRCNN
predict_srcnn_full_image(srcnn_model, image_path, scale_factor=4)

# SRGAN
predict_srgan_full_image(srgan_gen, image_path, scale_factor=4)

# Attentive ESRGAN
predict_attentive_full_image(att_gen, image_path, scale_factor=4)

# Compare two models
compare_two_gan_models(
    gen_a=srgan_gen,
    gen_b=att_gen,
    label_a="SRGAN Baseline",
    label_b="Attentive ESRGAN",
    image_path=image_path,
    scale_factor=4
)
```

## Hyperparameters

### Global Configuration
- **Batch Size**: 16
- **HR Crop Size**: 128×128 pixels
- **Upscale Factor**: 4×
- **LR Crop Size**: 32×32 pixels (HR_CROP_SIZE / UPSCALE)

### SRCNN
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss**: Mean Squared Error (MSE)
- **Architecture**: 3 convolutional layers (64→32→3 filters)

### SRGAN Baseline
- **Generator Learning Rate**: 1e-4
- **Discriminator Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Residual Blocks**: 16
- **Loss Weights**: 
  - Adversarial loss: 1e-3
  - VGG perceptual loss: 1.0
- **Discriminator Loss**: Binary Cross-Entropy (BCE)

### Attentive ESRGAN
- **Generator Learning Rate**: 1e-4
- **Discriminator Learning Rate**: 5e-5
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Gradient Clipping**: 1.0
- **Residual Blocks**: 16
- **Residual Scaling**: 0.2
- **Loss Weights**:
  - VGG perceptual loss: 0.006
  - Adversarial loss (RaLSGAN): 5e-3
  - L1 pixel loss: 1e-2
- **Channel Attention Ratio**: 16

## Experiment Results

### Training Metrics

The training process tracks:
- **Discriminator Loss**: Measures how well the discriminator distinguishes real from fake images
- **Generator Loss**: Combined adversarial and perceptual loss
- **Validation PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **Validation SSIM**: Structural Similarity Index (closer to 1.0 is better)

### Example Results (from training logs)

**Attentive ESRGAN Training** (first few epochs):
- Epoch 1: D_loss=2.11, G_loss=0.14, Val_PSNR=21.24 dB, Val_SSIM=0.5260
- Epoch 2: D_loss=0.54, G_loss=0.13, Val_PSNR=21.16 dB, Val_SSIM=0.5388
- Epoch 3: D_loss=0.53, G_loss=0.13, Val_PSNR=21.23 dB, Val_SSIM=0.5218

### Model Comparison

The project includes visualization tools to compare:
- Bicubic interpolation (baseline)
- SRCNN output
- SRGAN output
- Attentive ESRGAN output

Metrics are computed for each method:
- **PSNR** (dB): Measures pixel-level accuracy
- **SSIM**: Measures perceptual similarity

## Architecture Details

### SRCNN
- Simple 3-layer CNN optimized for PSNR
- Input: Bicubic upsampled LR image
- Output: Refined HR image

### SRGAN Generator (SRResNet)
- Initial convolution (9×9, 64 filters)
- 16 residual blocks with BatchNorm
- Skip connection
- 2× upsampling blocks (PixelShuffle)
- Final convolution with tanh activation
- Output range: [-1, 1]

### Attentive ESRGAN Generator
- Similar structure to SRResNet but:
  - **No BatchNorm** in residual blocks
  - **Channel Attention** modules after convolutions
  - **Residual scaling** (0.2) for stable training
  - Uses PixelShuffle for upsampling

### Discriminators
- **SRGAN**: Standard discriminator with BCE loss
- **ESRGAN**: Relativistic discriminator with RaLSGAN loss (no sigmoid, outputs logits)

## Loss Functions

### SRGAN
- **Adversarial Loss**: Binary Cross-Entropy
- **Perceptual Loss**: MSE on VGG19 features (block5_conv4)

### Attentive ESRGAN
- **Adversarial Loss**: Relativistic average Least Squares GAN (RaLSGAN)
- **Perceptual Loss**: MSE on VGG19 features
- **Pixel Loss**: L1 loss between HR and SR images

## File Structure

```
dip-final-project-srgan/
├── dip-final-project-srgan.ipynb  # Main notebook with all code
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── (model checkpoints)             # Saved during training
    ├── srgan_generator_epoch_*.keras
    └── attentive_esrgan_epoch_*.keras
```

## Notes

- Models are saved after each epoch during training
- Training history can be saved as JSON for later analysis
- The code includes visualization functions for comparing results
- Pre-trained warm-up weights are recommended for GAN training stability

## References

- SRCNN: "Image Super-Resolution Using Deep Convolutional Networks" (Dong et al., 2014)
- SRGAN: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (Ledig et al., 2017)
- ESRGAN: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (Wang et al., 2018)
- RCAN: Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., and Fu, Y., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", arXiv e-prints, Art. no. arXiv:1807.02758, 2018. doi:10.48550/arXiv.1807.02758.

## License

This project is for educational purposes.

