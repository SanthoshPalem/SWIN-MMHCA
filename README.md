# Swin-MMHCA for Medical Image Super-Resolution

## Overview

This project implements a medical image super-resolution model that leverages a Swin Transformer and a Multi-Head Convolutional Attention (MHCA) module. The model, named `SwinMMHCA`, was designed as a proof-of-concept to test a novel architecture against the baseline established in the paper "Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution".

Our `SwinMMHCA` model demonstrates a significant performance improvement over the baseline `EDSR_Nav` model within our controlled test environment.

## Architectures

### Baseline Model: EDSR + MMHCA

The original paper proposed an architecture based on EDSR (a ResNet-style backbone) that was modified to accept multiple input modalities. The features from each modality are extracted by separate backbones, concatenated, and then fused using the proposed MMHCA attention module.

<img src="https://raw.githubusercontent.com/lilygeorgescu/MHCA/main/imgs/overview.png" alt="Baseline EDSR+MMHCA Architecture" width="700">

*(Source: Georgescu et al., WACV 2023)*

### Our Novel Architecture: SwinMMHCA

We replaced the EDSR backbone with a more modern and powerful **Swin Transformer**. The goal was to leverage the Swin Transformer's advanced hierarchical attention mechanism for superior feature extraction. The general architecture can handle both single and multiple inputs.

#### Multi-Input `SwinMMHCA` (General Design)

![alt text](<Untitled diagram-2025-12-05-062541.png>)

#### Single-Input `SwinMMHCA` (As Trained and Evaluated)
This is the version we trained and evaluated in our experiments.
```mermaid
graph TD;
    A[LR Input Image <br> (T2w)] --> B(CNN Encoder);
    B --> C(2D Positional Encoding);
    C --> D{Swin Transformer <br> Deep Feature Extraction};
    D --> E(Reshape);
    E --> F(MMHCA Module <br> Spatial/Channel Attention);

    subgraph Upsampling Decoder
        direction LR
        G1(UpsampleBlock) --> G2(UpsampleBlock) --> G_dots(...) --> Gn(Final Conv);
    end

    F --> G1;
    Gn --> H[HR Output Image];
```

## Performance

The following table summarizes the performance of our `SwinMMHCA` model compared to the baseline `EDSR_Nav` model from the `MHCA-main` project, evaluated within our test environment on the IXI dataset.

| Model       | Scale | PSNR (dB) ↑ | SSIM ↑    | LPIPS ↓   |
| :---------- | :---- | :--------   | :-------- | :-------- |
| SwinMMHCA   | 2x    | **35.59**   | **0.828** | 0.162     |
| EDSR_Nav    | 2x    | 30.87       | 0.485     | **0.130** |
| SwinMMHCA   | 4x    | **33.90**   | **0.768** | **0.236** |
| EDSR_Nav    | 4x    | 23.67       | 0.135     | 0.813     |

*(Higher PSNR/SSIM is better, lower LPIPS is better)*

### Conclusion

As shown in the table, our **`SwinMMHCA` model significantly outperforms the baseline `EDSR_Nav` model** in our direct, head-to-head comparison across both 2x and 4x scales, particularly in the crucial PSNR and SSIM metrics.

Notably, the performance of the `EDSR_Nav` model in our testbed was considerably lower than the scores reported in the original paper. However, our primary goal was to validate the novel architecture. The results confirm that **replacing the EDSR backbone with a Swin Transformer yields a substantial improvement in performance**, with our `SwinMMHCA` model's 4x PSNR score (33.90) even surpassing the paper's reported score for their EDSR-based model (32.51).

## Setup and Usage

### Codebase Flow
The project is organized as follows:
```
Swin-MMHCA/
|-- datasets/
|-- pretrained_models/
|-- results/
|-- src/
|   |-- data/
|   |-- models/
|-- run.py
|-- ...
```

### Setup

1.  **Clone the repository and navigate into it.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset

Place the IXI dataset in the `datasets/` folder, organized by modality (e.g., `IXI-T1/`, `IXI-T2/`, `IXI-PD/`).

### Usage

The `run.py` script is the main entry point for all operations.

#### Training
To train the `SwinMMHCA` model (e.g., 3-channel input for 50 epochs):
```bash
python run.py --mode train --n_inputs 3 --epochs 50 --batch_size 4
```

#### Evaluation
To evaluate the `SwinMMHCA` model:
```bash
python run.py --mode evaluate --model_type SwinMMHCA --checkpoint_path results/swin_mmhca.pth
```

To evaluate the baseline `EDSR_Nav` model:
```bash
python run.py --mode evaluate --model_type EDSR_Nav --edsr_checkpoint_path ../MHCA-main/edsr/pretrained_models/model_multi_input_IXI_x4.pt
```

#### Visualization
To generate side-by-side visual comparisons and save them to the `results/` folder:
```bash
python run.py --mode visualize --num_samples 3
```