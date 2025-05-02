ML-Arch: Machine Learning for Archaeological Classification
Authors: Deniz Kayıkçı (0000-0002-1311-4272) and Prof. Dr. Juan Antonio Barceló
Institution: Universitat Autònoma de Barcelona – Quantitative Archaeology Lab
Study: Archaeological Classification of Small Datasets Using Meta- and Transfer Learning


Overview
This repository contains the code and methodological implementation for our paper on classifying fragmented Hittite stele fragments using advanced machine learning methods. The key challenge addressed is working with very small and degraded archaeological datasets, a common issue in archaeology.

We explore and compare the following techniques:

Classical Machine Learning (SVM, KNN, RF, etc.)

Transfer Learning using ResNet18

Simple CNN with Few-Shot Learning (FSL)

A Hybrid Meta-Learning model using MAML + ResNet18 + FSL

Our models aim to predict the geographical provenance of stelae from four Hittite cities (Alacahöyük, Arslantepe, Karkamış, and Sakçagözü) based purely on image data.

Key Features
📉 Few-Shot Learning (FSL) and Meta-Learning (MAML) strategies for low-resource archaeological classification tasks

🧠 Comparison with human expert classification performance (62.5% accuracy benchmark)

🏛️ Dataset sourced from Ankara Anatolian Civilizations Museum

📊 Evaluation via 3-fold cross-validation

📦 Pretrained models with PyTorch and Scikit-learn

📂 Organized directory structure for support/query sets per class

Installation
We recommend using a virtual environment and conda for dependency management.

PyTorch (tested on 2.0+)

torchvision

scikit-learn

numpy

matplotlib

PIL

tqdm

Note: Please ensure your system has CUDA-enabled GPU if training deep learning models with GPU acceleration.

Directory Structure
bash
ML-Arch/
│
├── data/
│   ├── Alacahöyük/
│   ├── Arslantepe/
│   ├── Karkamış/
│   └── Sakçagözü/
│
├── notebooks/
│   ├── meta_learning_maml.ipynb
│   ├── transfer_resnet18.ipynb
│   └── classical_ml.ipynb
│
├── models/
├── utils/
│   ├── data_loader.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
Reproducibility
Please ensure to change local file paths in .ipynb scripts to match your directory.

Set a consistent random seed (random.seed(42), torch.manual_seed(42), etc.) for reproducibility.

Model configurations (learning rate, batch size, epochs) are defined in each notebook.

Data augmentation was not applied intentionally to simulate real-world data quality issues.

Dataset
Images (88 originals, 412 cropped, 136 used for training/testing) are not yet uploaded due to museum data usage restrictions.

If you are interested in replicating the study:

Please contact 1592874@uab.cat for access or alternatives.

We recommend generating your own image dataset with similar constraints (small, noisy, deformed).

Evaluation
We used:

Accuracy, Precision, Recall, F1 Score

Confusion matrices per class

Cross-validation (3-fold)

Human expert comparison

