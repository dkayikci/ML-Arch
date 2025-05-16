
ML-Arch: Machine Learning for Archaeological Classification

ML-Arch: Machine Learning for Archaeological Classification
Authors: Deniz Kayıkçı (0000-0002-1311-4272), Prof. Dr. Juan Antonio Barceló and Iban Berganzo-Besga 
Institution: Universitat Autònoma de Barcelona – Quantitative Archaeology Lab and Barcelona Supercomputing Centre
Study: Archaeological Classification of Small Datasets Using Meta- and Transfer Learning

Archaeological Classification of Hittite Stele Fragments Using Meta- and Transfer Learning
Show Image
Project Overview
This repository contains code and resources for classifying fragmented Hittite stele artifacts according to their geographical provenance (Alacahöyük, Arslantepe, Karkamış, or Sakçagözü) based on visual characteristics. The research addresses a fundamental challenge in archaeological classification: how to achieve accurate results with small, non-standardized, and highly degraded datasets.
Show Image
Research Problem
Archaeological datasets typically suffer from several limitations:

Small sample sizes (low-source-data)
Fragmented or deteriorated objects
Non-standardized and heterogeneous documentation

This research demonstrates that advanced machine learning approaches can effectively overcome these constraints and outperform traditional classification methods, and in some cases, match human expert performance.
Dataset
The dataset consists of Hittite stele images from four archaeological sites in Anatolia (modern-day Turkey):

Alacahöyük
Arslantepe
Karkamış
Sakçagözü

Two dataset configurations were evaluated:

Initial Dataset: 136 samples (112 training, 24 testing)
Enhanced Dataset: 208 samples (152 training, 56 testing)

Images were captured under controlled conditions at the Ankara Anatolian Civilizations Museum and were preprocessed to simulate real-world archaeological classification challenges.
Methodology
We compared four machine learning approaches:

Conventional Machine Learning Algorithms:

Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Random Forests (RF)
Logistic Regression (LR)
Decision Trees (DT)
Naive Bayes (NB)


Transfer Learning:

Pre-trained ResNet18 model with fine-tuning


Simple CNN with Few-Shot Learning:

Custom convolutional neural network with FSL integration


Hybrid Approach:

Combination of Model-Agnostic Meta-Learning (MAML)
Few-Shot Learning (FSL)
Transfer Learning (ResNet18)



Performance was evaluated against human expert classification as a benchmark.
Key Findings

The hybrid model achieved 82.74% accuracy on the enhanced dataset, comparable to human expert performance (85.7%)
Conventional ML methods showed poor generalization, with misclassification rates near 50%
Transfer learning (ResNet18) demonstrated high consistency across validation folds
Neural networks outperformed traditional methods even with extremely limited training data
Human experts and AI models showed complementary classification strengths





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

