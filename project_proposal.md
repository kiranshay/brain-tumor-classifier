# CS 475/675 Project Proposal

**Kiran Shay**

kshay1

## Abstract

Brain tumor diagnosis from MRI scans requires expert radiological interpretation that is time-consuming and subject to inter-observer variability. We propose a two-stage deep learning pipeline that classifies brain MRI scans into eight tumor categories and conditionally subtypes gliomas into four subcategories using transfer learning with EfficientNet-B0. A central methodological contribution is a controlled investigation of evaluation contamination in widely-used brain tumor MRI benchmarks, where patient-level data leakage inflates reported accuracy by up to 11.6 percentage points for specific classes.

## 1 Introduction

Automated classification of brain tumors from MRI is a well-studied problem in medical image analysis, but most published results on popular Kaggle benchmarks use image-level train/test splits that ignore patient identity. When a patient contributes multiple MRI slices, random splitting places near-duplicate images in both sets, inflating test accuracy by an unknown amount. This project addresses that gap by building a classifier with honest evaluation methodology alongside a controlled leakage experiment that quantifies the inflation.

The input to the system is a single brain MRI slice. Stage 1 classifies it into one of eight categories: glioma, meningioma, pituitary, schwannoma, neurocytoma, carcinoma, papilloma, or no tumor. When Stage 1 predicts glioma, Stage 2 conditionally fires to identify the subtype as astrocytoma, ependymoma, glioblastoma, or oligodendroglioma. Both stages produce full softmax probability distributions, and Grad-CAM heatmaps provide interpretability by visualizing which brain regions the model attends to.

## 2 Dataset and Features

Training combines three publicly available Kaggle datasets totaling over 15,000 MRI images. The Brain Tumor MRI Dataset [1] provides 7,023 images across 4 classes (glioma, meningioma, pituitary, no tumor) with a pre-defined train/test split. The BRISC 2025 dataset [2] adds 6,000 images in axial, sagittal, and coronal orientations for the same 4 classes. The 44-class Brain Tumor MRI dataset [3] contributes fine-grained tumor types across T1, T2, and T1C+ MRI sequences, supplying the four rare classes (schwannoma, neurocytoma, carcinoma, papilloma) and all glioma subtype training data.

Preprocessing includes MRI-specific black border cropping (to remove scanner bezels), resizing to 224x224, and ImageNet mean/std normalization. Data augmentation applies random cropping from 256, horizontal and vertical flips, rotation (+/-20 degrees), random affine transforms (translation 0.1, scale 0.9-1.1), color jitter, random grayscale, and random erasing.

A key methodological concern is patient-level data leakage. Both the BRISC and 44-class datasets contain multiple slices per patient. Standard image-level random splits place correlated slices in both train and test, overstating generalization. We construct a clean dataset variant using inferred patient IDs and GroupShuffleSplit to produce a proper train/validation/test partition with no patient overlap, enabling a controlled comparison of leaked vs. clean evaluation.

## 3 Methods

Both stages use EfficientNet-B0 [4] pretrained on ImageNet with a replaced classifier head (Dropout followed by a Linear layer). Models are fully fine-tuned using AdamW optimization (lr=5e-5, weight decay=0.01) with cosine annealing learning rate scheduling over 20 epochs. The loss function is cross-entropy. The Stage 1 model uses dropout 0.3 with 8 output classes; Stage 2 uses dropout 0.4 with 4 output classes. Both are trained on Google Colab with T4 GPUs.

For the contamination experiment, we train identical architectures on two dataset variants: (a) the standard image-level random split and (b) the patient-aware GroupShuffleSplit. Comparing test accuracy between the two isolates the effect of leakage. We further analyze whether the four rare classes sourced exclusively from the 44-class dataset benefit from source confound (the model recognizing dataset-specific scanner signatures rather than tumor morphology) by examining whether these classes show systematically higher accuracy despite maximum confound opportunity.

Models are evaluated using per-class precision, recall, and F1-score, confusion matrices, and Wilson 95% confidence intervals for all per-class metrics. Wilson intervals are critical for rare classes with small test sample sizes (e.g., papilloma n=35, carcinoma n=37), where standard accuracy reporting gives a misleading sense of precision. Grad-CAM [5] heatmaps are generated via hooks on the final convolutional block to provide visual interpretability.

## 4 Deliverables

### 4.1 Must accomplish

(a) Train and evaluate the 8-class Stage 1 EfficientNet-B0 classifier on the combined dataset with image-level splits, reporting per-class precision, recall, F1, and confusion matrix.

(b) Build the clean dataset variant with patient-aware splits and train an identical model to quantify the accuracy inflation from data leakage.

(c) Compute Wilson 95% confidence intervals for all per-class metrics and report honest accuracy bounds, particularly for rare classes.

### 4.2 Expect to accomplish

(a) Train and evaluate the Stage 2 glioma subtype classifier (4-class) and integrate it into a conditional cascade pipeline.

(b) Implement Grad-CAM interpretability and analyze whether the model attends to clinically plausible brain regions across tumor types.

(c) Perform the source confound analysis: test whether rare classes sourced from a single dataset achieve high accuracy via scanner signature recognition rather than tumor morphology.

### 4.3 Would like to accomplish

(a) Deploy the full pipeline as a web application (FastAPI backend, vanilla JavaScript frontend, Supabase database) with prediction history, aggregate statistics, and on-demand Grad-CAM generation.

(b) Explore class-weighted loss or weighted random sampling to address severe class imbalance between common classes (glioma, meningioma) and rare classes (papilloma, carcinoma).

(c) Develop the contamination analysis into a standalone research contribution: a reproducible benchmark showing how widely-used brain tumor MRI datasets overstate classifier performance when patient identity is ignored.

## References

[1] Nickparvar, M. (2021). Brain Tumor MRI Dataset. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

[2] BRISC Dataset (2025). Brain tumor MRI classification, multi-orientation. Kaggle. https://www.kaggle.com/datasets/briscdataset/brisc2025

[3] Feltrin, F. (2022). Brain Tumor MRI Images 44 Classes. Kaggle. https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c

[4] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the 36th International Conference on Machine Learning (pp. 6105-6114).

[5] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (pp. 618-626).
