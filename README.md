# Hybrid Deep Learning Model: U-Net + MobileNetV2 for Brain Tumor Segmentation

This repository contains a hybrid deep learning model, **UMobileNetV2**, combining the spatial localization capabilities of U-Net and the efficient feature extraction of MobileNetV2. The model is designed for precise brain tumor segmentation in MRI images and demonstrates strong improvements in segmentation accuracy, loss, and computational efficiency.

## Table of Contents

- [Project Overview]
- [Architecture]
  - [Base U-Net]
  - [UMobileNetV2]
-[Workflow]
- [Dataset]
- [Installation]
- [Usage]
  - [Data Preparation]
  - [Training]
  - [Evaluation]
- [Results]
- [Model Comparison]
- [Future Work]
- [References]

## Project Overview

**Goal:** Develop a robust and efficient model for pixel-wise brain tumor segmentation in MRI scans, leveraging transfer learning and architectural innovations for improved clinical applicability.

Traditional segmentation models (e.g., U-Net) struggle with limited data and complex tumor shapes. By integrating MobileNetV2 as a pre-trained encoder into U-Net's structure, UMobileNetV2 achieves better feature representation, faster convergence, and higher segmentation accuracy.[1]

## Architecture
! [Final Architecture](images/architecture.png)

### Base U-Net

- Encoder-decoder structure with skip connections
- Custom blocks: HAC (Hierarchical Attention Convolution) and CASPP (Context Aggregation Spatial Pyramid Pooling) to enhance multi-scale feature extraction and contextual understanding
- Trained from scratch, requiring more data and computation
! [Base Architecture](images/base-architecture.png)
### UMobileNetV2

- **Encoder:** MobileNetV2 pre-trained on ImageNet for efficient, scalable, and robust feature extraction
- **Decoder:** U-Net style, with strategic skip connections from MobileNetV2's intermediate layers
- **Multi-scale feature fusion:** Preserves both fine-grained and semantic details
- **Output:** 1×1 convolution with sigmoid activation for binary segmentation
- **Loss Functions:** Combined binary cross-entropy, Dice loss, and Tversky loss (handles class imbalance in medical imaging)
- **Data Augmentation:** Rotation, translation, scaling, flipping, elastic deformation, brightness/contrast adjustment, noise, and cropping/padding[2][1]

## Workflow
! [Work Flow](images/work-flow.png)

## Dataset

-https://www.kaggle.com/datasets/divyaiyer123/brain-tumor-segmentation-with-mask?select=archive
- Each image file must have a corresponding mask file with the same name

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install Python libraries:
   ```bash
   pip install tensorflow numpy opencv-python scikit-learn matplotlib
   ```

## Usage

### Data Preparation

- Organize MRI images and masks in the specified folders.
- Ensure all image-mask pairs are correctly matched and resized to 256x256.

### Training

By default, the script will automatically load data, apply augmentation, and start training if the model file does not already exist.
- Edit `PATH` in the notebook to point to the dataset directory.
- Run the notebook `munet1.ipynb` to train and save the model:
  - If a trained model file (`munet_segmentation_model.h5`) exists, it will be loaded instead of retraining.
  

*Key hyperparameters:*
- Epochs: 30
- Batch size: 8
- Learning rate: 1e-4
- Input size: 256
 ! [Training_results](images/training-results.png) 

### Evaluation

Model evaluation metrics include:
- Accuracy
- Dice Coefficient
- Tversky Loss

To evaluate:
- Run the evaluation cells in the notebook.
- Results are printed after completion:
  ```
  Test Loss: 0.1489
  Test Dice Coefficient: 0.8249
  Test Accuracy: 0.9946
  ```

### Visualization

Prediction visualizations for MRI images are included for quantitative and qualitative analysis, showing improvements in boundary and small lesion segmentation.
! [Base Architecture visualization](images/base-architecture-visualization.png)
! [Final Architecture Visualization](images/final-architecture-Visualization.png)

## Results

UMobileNetV2 outperforms the base U-Net in both segmentation accuracy and computational efficiency:

| Model            | Dice Coefficient | Test Loss | Accuracy |
|------------------|-----------------|-----------|----------|
| U-Net            | 0.7556          | 0.2077    | 0.9927   |
| UMobileNetV2     | **0.8249**      | **0.1489**| **0.9946** |

UMobileNetV2 converges faster during training (406 ms/step vs 555 ms/step for U-Net) and is robust on unseen data due to transfer learning.[2][1]

## Model Comparison

| Factor                      | U-Net           | UMobileNetV2          |
|-----------------------------|-----------------|-----------------------|
| Training Data Requirement   | High            | Low (transfer learning)[1]|
| Convergence Speed           | Slow            | Fast                  |
| Dice Coefficient            | Lower           | Higher                |
| Computational Cost          | Higher          | Lower                 |
| Segmentation Quality        | Good            | **Superior**          |

## Future Work

Integrating attention gates or transformer-based blocks.

Multi-modal MRI analysis (T1, T2, FLAIR).

Deployment as a web/edge application for real-time clinical use.

Explainability with Grad-CAM for medical trustworthiness.

## References

1. Shelke, S.M. & Mohod, S.W. (2018). Automated Segmentation and Detection of Brain Tumor from MRI. IEEE.
2. Acharya, M. et al. (2020). MRI-based Diagnosis of Brain Tumors Using Deep Neural Networks. IEEE.
***
