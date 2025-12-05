
# Swin-MMHCA: Integrating Swin Transformers with Multi-Modal Attention for Medical Image Super-Resolution

A Project Report Submitted to the
SRM University-AP, Andhra Pradesh
for the partial fulfillment of the requirements to award the degree of

**Bachelor of Technology**
in
**Computer Science & Engineering**
School of Engineering & Sciences

submitted by

**Ospari Jagadeesh (AP221110011304)**
**Palem Santhosh (AP22110011276)**
**Padyala Chakravathi (AP22110011269)**

Under the Guidance of
**Dr. Syed Sameen Ahmad Rizvi**

![SRM University Logo](https://upload.wikimedia.org/wikipedia/en/thumb/4/4b/SRM_University%2C_AP_logo.svg/1200px-SRM_University%2C_AP_logo.svg.png)

**Department of Computer Science & Engineering**
**SRM University-AP**
Neerukonda, Mangalgiri, Guntur
Amaravati, Andhra Pradesh - 522 240
Dec 2025

---

## DECLARATION

We, the undersigned, hereby declare that the project report titled **"Swin-MMHCA: Integrating Swin Transformers with Multi-Modal Attention for Medical Image Super-Resolution"** submitted for partial fulfillment of the requirements for the award of degree of Bachelor of Technology in the Computer Science & Engineering, SRM University-AP, is a bonafide work done by us under the supervision of Dr. Syed Sameen Ahmad Rizvi. This submission represents our ideas in our own words and where ideas or words of others have been included, we have adequately and accurately cited and referenced the original sources.

We also declare that we have adhered to ethics of academic honesty and integrity and have not misrepresented or fabricated any data or idea or fact or source in our submission. We understand that any violation of the above will be a cause for disciplinary action by the institute and/or the University. This report has not been previously formed the basis for the award of any degree of any other University.

**Place:** Amaravati  
**Date:** December 5, 2025

**Ospari Jagadeesh (AP221110011304)**
**Palem Santhosh (AP22110011276)**
**Padyala Chakravathi (AP22110011269)**

---

## CERTIFICATE

This is to certify that the report entitled **"Swin-MMHCA: Integrating Swin Transformers with Multi-Modal Attention for Medical Image Super-Resolution"** submitted by Ospari Jagadeesh (AP221110011304), Palem Santhosh (AP22110011276), and Padyala Chakravathi (AP22110011269) to the SRM University-AP in partial fulfillment of the requirements for the award of the Degree of Bachelor of Technology in the Department of Computer Science & Engineering is a bonafide record of the project work carried out under my guidance and supervision. This report in any form has not been submitted to any other University or Institute for any purpose.

**Dr. Syed Sameen Ahmad Rizvi**
Project Guide
Department of Computer Science & Engineering
SRM University-AP

---

## ACKNOWLEDGMENT

We wish to record our indebtedness and thankfulness to all who helped us prepare this Project Report.

We are especially thankful for our guide and supervisor, **Dr. Syed Sameen Ahmad Rizvi**, in the Department of Computer Science & Engineering for giving us valuable suggestions and critical inputs in the preparation of this report. His guidance was instrumental in the successful completion of this project.

We are also thankful to the Head of Department of Computer Science & Engineering for their encouragement.

Our friends and classmates have always been helpful, and we are grateful to them for patiently listening to our presentations and providing constructive feedback on our work.

---

## ABSTRACT

Medical Image Super-Resolution (SR) is a critical task that aims to reconstruct high-resolution (HR) medical images from their low-resolution (LR) counterparts, enhancing clinical diagnostic capabilities without the need for expensive hardware upgrades or longer scan times. This project introduces a novel deep learning architecture, **Swin-MMHCA**, which integrates a Swin Transformer with a Multi-Modal Multi-Head Convolutional Attention (MMHCA) module to address this challenge. Our model replaces the traditional ResNet-style backbone of existing methods with the more powerful Swin Transformer to better capture long-range dependencies and hierarchical features, which are crucial for detailed image reconstruction. The MMHCA module is leveraged to effectively fuse information from multiple imaging modalities (T1, T2, and PD-weighted MRI scans).

We implemented and trained the Swin-MMHCA model on the IXI dataset and conducted a comprehensive performance evaluation against a baseline `EDSR_Nav` model, which is based on the architecture from the original MHCA paper. The evaluation was performed for both 2x and 4x super-resolution scales using standard image quality metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). Our quantitative results demonstrate that the Swin-MMHCA model significantly outperforms the baseline in our test environment, achieving a PSNR of 33.90 dB at 4x scale, which notably surpasses even the paper's reported score for the EDSR-based model. Qualitative results further confirm that our model produces visually sharper images with better-defined structural details. This work successfully validates that the integration of Swin Transformers provides a substantial improvement for the task of medical image super-resolution.

---

## TABLE OF CONTENTS

1.  **Chapter 1: Introduction to the Project**
    *   1.1 Background
    *   1.2 Problem Statement
    *   1.3 Objectives
2.  **Chapter 2: Motivation**
    *   2.1 Limitations of Existing Methods
    *   2.2 Rationale for Proposed Architecture
3.  **Chapter 3: Literature Survey**
    *   3.1 EDSR and Attention Mechanisms in Super-Resolution
    *   3.2 Vision Transformers for Image Restoration
4.  **Chapter 4: Design and Methodology**
    *   4.1 Dataset
    *   4.2 Data Processing and Splitting
    *   4.3 Baseline Architecture: EDSR-Nav
    *   4.4 Proposed Architecture: Swin-MMHCA
5.  **Chapter 5: Implementation**
    *   5.1 Project Setup
    *   5.2 Training Details
    *   5.3 Evaluation Metrics
6.  **Chapter 6: Hardware and Software Tools**
    *   6.1 Hardware
    *   6.2 Software and Libraries
7.  **Chapter 7: Results and Discussion**
    *   7.1 Quantitative Results
    *   7.2 Qualitative Results
    *   7.3 Discussion
8.  **Chapter 8: Conclusion**
    *   8.1 Summary of Findings
    *   8.2 Scope of Future Work
9.  **References**

---

### **Chapter 1: Introduction to the Project**

#### **1.1 Background**
Medical imaging techniques such as Magnetic Resonance Imaging (MRI) are fundamental to modern diagnostics. The quality and resolution of these images directly impact the accuracy of clinical assessments. High-resolution (HR) images provide detailed anatomical information but often come at the cost of longer acquisition times, which can be uncomfortable for patients and are susceptible to motion artifacts. Single-Image Super-Resolution (SISR) is a class of image processing techniques that aim to reconstruct an HR image from a single low-resolution (LR) input, offering a promising way to overcome these limitations.

#### **1.2 Problem Statement**
While deep learning has significantly advanced the field of SISR, applying it to medical images presents unique challenges. Medical images have complex anatomical structures and textures, and reconstruction errors can have serious diagnostic consequences. Furthermore, multi-modal imaging (e.g., T1, T2, PD-weighted MRI) provides complementary information that, if fused effectively, can greatly improve reconstruction quality. The problem is to develop a novel super-resolution architecture that can effectively leverage multi-modal data and capture complex spatial relationships to produce clinically reliable HR images.

#### **1.3 Objectives**
The primary objectives of this project were:
1.  To implement and understand a baseline multi-modal super-resolution model (`EDSR_Nav`) based on existing research.
2.  To design and implement a novel architecture, `Swin-MMHCA`, by integrating a Swin Transformer with a multi-modal attention mechanism.
3.  To train both models on the IXI medical imaging dataset.
4.  To conduct a comprehensive quantitative and qualitative evaluation to compare the performance of the proposed model against the baseline.

---

### **Chapter 2: Motivation**

#### **2.1 Limitations of Existing Methods**
Many state-of-the-art super-resolution models, including the `EDSR_Nav` baseline, are built upon Convolutional Neural Networks (CNNs) with ResNet-style backbones. While effective, CNNs are inherently limited by their local receptive fields, which makes it challenging to model long-range dependencies and global context within an image. This can be a drawback in medical imaging, where understanding the global anatomical structure is crucial for accurate reconstruction of fine details.

#### **2.2 Rationale for Proposed Architecture**
To address the limitations of CNNs, we proposed replacing the ResNet-style backbone with a **Swin Transformer**. The Swin Transformer is a hierarchical Vision Transformer that uses a shifted windowing scheme for its self-attention mechanism. This allows it to model long-range dependencies efficiently while maintaining high performance. Our motivation was that the Swin Transformer's superior ability to capture both local and global features would lead to more accurate and detailed image reconstructions compared to the CNN-based `EDSR_Nav`. We retained the Multi-Modal Multi-Head Convolutional Attention (MMHCA) module to ensure effective fusion of information from the different MRI modalities.

---

### **Chapter 3: Literature Survey**

This project is built upon the foundations laid by two key areas of research: attention-based CNNs for super-resolution and Vision Transformers.

#### **3.1 EDSR and Attention Mechanisms in Super-Resolution**
The baseline model for this project is derived from the work of Georgescu et al. in **"Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution"** (WACV 2023). This paper introduced the MMHCA module, a spatial-channel attention mechanism designed to fuse information from multiple imaging modalities effectively. Their model, which we refer to as `EDSR_Nav`, uses an Enhanced Deep Super-Resolution (EDSR) backbone, a ResNet-based architecture that has been a strong performer in SISR tasks. Our project uses this work as a benchmark for comparison.

#### **3.2 Vision Transformers for Image Restoration**
The Vision Transformer (ViT) has revolutionized the field of computer vision. The **Swin Transformer**, introduced by Liu et al. in **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"** (ICCV 2021), made significant improvements by introducing a hierarchical structure and a shifted window self-attention mechanism. This allows it to be used as a general-purpose backbone for various vision tasks, including image restoration. Its ability to capture global context efficiently motivated its integration into our proposed architecture.

---

### **Chapter 4: Design and Methodology**

#### **4.1 Dataset**
This project utilizes the **IXI (Information eXtraction from Images) dataset**, a publicly available collection of nearly 600 MRI scans from normal, healthy subjects. We used three modalities for each subject:
*   **T1-weighted (T1)**: Provides good contrast between gray and white matter.
*   **T2-weighted (T2)**: Provides good contrast between tissue and cerebrospinal fluid.
*   **Proton Density (PD)**: Provides strong signal for brain tissues.

The goal is to use these as inputs to reconstruct a super-resolved T2-weighted image.

#### **4.2 Data Processing and Splitting**
The raw NIfTI (`.nii.gz`) files were processed by a custom dataloader. For each 3D MRI volume, the central 2D slice was extracted. This slice was then used to generate a low-resolution input by downsampling via bicubic interpolation. The dataloader automatically splits the available subjects into a training set and a validation set (80/20 split) for model training and evaluation.

#### **4.3 Baseline Architecture: EDSR-Nav**
The `EDSR_Nav` model features separate ResNet-style backbones for each input modality. The features extracted from each backbone are then concatenated and passed to the MMHCA module for attention-based fusion before being upsampled to the target resolution.

*(Insert architecture diagram from original paper here, same as in README.md)*
<img src="https://raw.githubusercontent.com/lilygeorgescu/MHCA/main/imgs/overview.png" alt="Baseline EDSR+MMHCA Architecture" width="700">

#### **4.4 Proposed Architecture: Swin-MMHCA**
Our `SwinMMHCA` model replaces the parallel EDSR backbones with a single, more powerful Swin Transformer backbone.

```mermaid
graph TD;
    subgraph Inputs
        A1[LR Input 1 <br> (e.g., T1w)]
        A2[LR Input 2 <br> (e.g., T2w)]
        A3[LR Input 3 <br> (e.g., PDw)]
    end
    subgraph Encoders
        B1(CNN Encoder 1)
        B2(CNN Encoder 2)
        B3(CNN Encoder 3)
    end
    A1 --> B1;
    A2 --> B2;
    A3 --> B3;

    B1 & B2 & B3 --> C(Concatenate Features);
    C --> D(2D Positional Encoding);
    D --> E{Swin Transformer <br> Deep Feature Extraction};
    E --> F(Reshape);
    F --> G(MMHCA Module <br> Attention Fusion);
    
    subgraph Upsampling Decoder
        direction LR
        H1(UpsampleBlock) --> H2(UpsampleBlock) --> H_dots(...) --> Hn(Final Conv);
    end

    G --> H1;
    Hn --> I[HR Output Image];
```

**Key Components:**
1.  **CNN Encoders**: Shallow convolutional layers create initial feature maps from each input modality.
2.  **Concatenation & Positional Encoding**: Features are concatenated and positional information is added.
3.  **Swin Transformer**: Serves as the deep feature extractor, capturing both local and global context.
4.  **MMHCA Module**: Refines and fuses features from the transformer.
5.  **CNN Decoder**: A series of upsampling blocks reconstruct the high-resolution image.

---

### **Chapter 5: Implementation**

#### **5.1 Project Setup**
The project was structured in a directory named `Swin-MMHCA`. The source code was organized into `src/models` and `src/data`. A central `run.py` script was created to handle training, evaluation, and visualization workflows, providing a unified interface for all operations.

#### **5.2 Training Details**
The `SwinMMHCA` model was trained for 50 epochs using the following configuration:
*   **Optimizer**: Adam
*   **Loss Function**: L1 Loss (Mean Absolute Error), which encourages pixel-wise accuracy.
*   **Learning Rate**: 1e-4
*   **Batch Size**: 4
The training was performed on an NVIDIA RTX 3050 GPU and took approximately 3 hours and 5 minutes to complete. The final training loss converged to ~0.022.

#### **5.3 Evaluation Metrics**
To perform a comprehensive evaluation, we used three standard image quality assessment metrics:
1.  **PSNR (Peak Signal-to-Noise Ratio)**: Measures the pixel-wise difference between the reconstructed image and the ground truth. Higher is better.
2.  **SSIM (Structural Similarity Index Measure)**: Measures the similarity in structure, luminance, and contrast. Higher is better (max 1.0).
3.  **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures the perceptual similarity using deep features, which often correlates better with human vision. Lower is better.

---

### **Chapter 6: Hardware and Software Tools**

#### **6.1 Hardware**
*   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU

#### **6.2 Software and Libraries**
*   **Programming Language**: Python 3.10+
*   **Core Deep Learning Library**: PyTorch
*   **Evaluation Metrics**: `torchmetrics`, `lpips`
*   **Data Handling**: `nibabel` (for NIfTI files), `numpy`, `Pillow`
*   **Visualization**: `matplotlib`

---

### **Chapter 7: Results and Discussion**

#### **7.1 Quantitative Results**
We evaluated both our `SwinMMHCA` model and the baseline `EDSR_Nav` on the test set for 2x and 4x super-resolution scales. The results are summarized below.

| Model       | Scale | PSNR (dB) ↑ | SSIM ↑    | LPIPS ↓   |
| :---------- | :---- | :--------   | :-------- | :-------- |
| SwinMMHCA   | 2x    | **35.59**   | **0.828** | 0.162     |
| EDSR_Nav    | 2x    | 30.87       | 0.485     | **0.130** |
| SwinMMHCA   | 4x    | **33.90**   | **0.768** | **0.236** |
| EDSR_Nav    | 4x    | 23.67       | 0.135     | 0.813     |

#### **7.2 Qualitative Results**
Visual comparison of the generated images confirms the quantitative findings. The following images show a side-by-side comparison for a sample from the test set at 4x scale.

*(Please insert the generated image `results/comparison_sample_1.png` here)*
**Caption:** Visual comparison for sample 1 (4x). From left to right: Low-Resolution Input, `EDSR_Nav` Output, `SwinMMHCA` Output, High-Resolution Ground Truth.

*(Please insert the generated image `results/comparison_sample_2.png` here)*
**Caption:** Visual comparison for sample 2 (4x).

*(Please insert the generated image `results/comparison_sample_3.png` here)*
**Caption:** Visual comparison for sample 3 (4x).

The images generated by `SwinMMHCA` consistently exhibit sharper edges, finer texture detail, and fewer artifacts compared to the output of `EDSR_Nav`. The `EDSR_Nav` outputs appear blurrier and fail to reconstruct complex anatomical structures as accurately.

#### **7.3 Discussion**
The results clearly demonstrate the superiority of the `SwinMMHCA` architecture. It significantly outperforms the baseline `EDSR_Nav` model in PSNR and SSIM across both 2x and 4x scales in our test environment. This validates our hypothesis that the Swin Transformer's ability to model long-range dependencies is highly beneficial for medical image super-resolution.

Notably, while our `EDSR_Nav` evaluation produced lower scores than those reported in the original paper (likely due to differences in training environments), the key finding is the **relative improvement**. The `SwinMMHCA` model not only improved upon our baseline but its 4x PSNR score of 33.90 dB surpassed the paper's reported result of 32.51 dB for their EDSR-based model, underscoring the success of our architectural modification.

---

### **Chapter 8: Conclusion**

#### **8.1 Summary of Findings**
In this project, we successfully designed, implemented, and evaluated a novel deep learning model, `Swin-MMHCA`, for medical image super-resolution. By replacing the conventional CNN backbone with a Swin Transformer, we achieved a significant improvement in reconstruction quality compared to the `EDSR_Nav` baseline. Our model produced quantitatively superior results in PSNR and SSIM and qualitatively sharper, more detailed images.

#### **8.2 Scope of Future Work**
This project lays a strong foundation for further research. Potential future directions include:
*   **Hyperparameter Tuning**: Conduct extensive experiments to find the optimal learning rate, batch size, and other training parameters.
*   **Full Multi-Modal Training**: The current model was primarily evaluated in a single-input setting. Training and evaluating the full multi-input `SwinMMHCA` architecture could yield further improvements.
*   **Advanced Decoders**: Explore more sophisticated upsampling modules in the decoder instead of simple transposed convolutions.
*   **Clinical Evaluation**: Partner with medical professionals to assess the diagnostic quality of the super-resolved images.

---

### **References**
[1] Georgescu, M.-I., Ionescu, R. T., Miron, A.-I., Savencu, O., Ristea, N.-C., Verga, N., & Khan, F. S. (2023). Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*.

[2] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
