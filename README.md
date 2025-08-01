# DFSSD: Deep Feature-aware Semi-Supervised Distillation

**DFSSD** is a deep learning-based framework for **industrial defect detection** under **semi-supervised learning** conditions. It leverages the concept of **teacher-student distillation** to train lightweight student models from large teacher models, achieving both high detection accuracy and low-latency inference.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Training and Testing](#training-and-testing)

## Introduction

**DFSSD** (Deep Feature-aware Semi-Supervised Distillation) is designed for industrial defect detection, where annotated data is limited. By employing **distillation-based training**, it trains a compact student model from a teacher model with significantly fewer parameters while maintaining robust detection accuracy.

Key features:
- **Teacher-student distillation**: Compresses a large teacher model (YOLOv8s) to a smaller student model (YOLOv8n) while preserving high performance.
- **Semi-supervised learning**: Uses a minimal amount of labeled data and pseudo-labeled data generated by the teacher model.
- **High efficiency**: Focuses on low-latency inference, making it suitable for real-time applications.

## Installation

### Prerequisites

- Python >= 3.6
- PyTorch >= 1.7
- CUDA (optional for GPU acceleration)

### Installing Dependencies 
```bash
   pip install -r requirements.txt
```
## Usage

### 1. Dataset

This repository uses the **CR7-DET** dataset for training and testing. You can download the dataset from the following link:

- [CR7-DET Dataset](<INSERT_YOUR_DATASET_LINK_HERE>)

Make sure to download and place the dataset in the appropriate directory for training and testing.

### 2. Data Splitting

The **CR7-DET** dataset is divided into **three subsets**:
- **Training**: 80% of the dataset is used for training the model.
- **Validation**: 10% of the dataset is reserved for validating the model during training.
- **Testing**: 10% of the dataset is used for evaluating the final model after training is complete.

In addition, the **training set** is split into **labeled** and **unlabeled** data:
- **Labeled Data**: This subset contains manually annotated data, which will be used for supervised learning.
- **Unlabeled Data**: This subset contains data without annotations, which will be used for **semi-supervised learning** with the help of pseudo-labeling generated by the teacher model.



## Training and Testing

To pre-train the teacher model, use the following command:
```bash
python main.py --teacher_model <path_to_teacher_model>  --dataset <path_to_dataset>
```

To train the distillation model, use the following command:
```bash
python KD.py --teacher_model <path_to_teacher_model> --student_model <path_to_student_model> --dataset <path_to_dataset> --distillation_type <distillation_method> --model_choice <teacher_or_student>
```

To train the semi-supervised distillation model, use the following command:
```bash
python SSKD.py --teacher_model <path_to_teacher_model> --student_model <path_to_student_model> --dataset <path_to_dataset> --threshold <pseudo_label_threshold>
```

To test the model, use the following command:
```bash
python test.py --model <path_to_student_model> --dataset <path_to_test_dataset>
```
