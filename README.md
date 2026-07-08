# 🌍 Satellite Land Cover Classification using Swin Transformer

## Overview

Satellite imagery has become one of the most valuable sources of information for understanding Earth's surface. Accurate land cover classification helps governments, researchers, urban planners, environmental agencies, and agricultural organizations monitor environmental changes, plan infrastructure, assess natural resources, and respond to disasters.

This project implements a **Satellite Land Cover Classification system** using the **Swin Transformer**, a state-of-the-art Vision Transformer architecture that combines hierarchical feature extraction with shifted window self-attention. Unlike traditional CNNs, Swin Transformers are capable of capturing both local and global contextual information, making them highly effective for remote sensing image analysis.

The model is fine-tuned using transfer learning on a satellite image dataset containing multiple land cover categories. After training, the model is deployed through an interactive **Streamlit web application**, allowing users to upload satellite images and receive instant predictions.

---

## live demo
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://satelitelandcoverclassification-lbce9dauhbgvtmrg24toyn.streamlit.app/)

# Project Objectives

* Build a high-performance land cover classification model using a Vision Transformer.
* Apply transfer learning to reduce training time while improving accuracy.
* Compare transformer-based image classification with traditional CNN approaches.
* Develop a user-friendly web interface for real-time predictions.
* Demonstrate an end-to-end deep learning workflow from data preprocessing to deployment.

---

# Features

* Swin Transformer backbone
* Transfer Learning
* Fine-tuning on satellite imagery
* Image augmentation
* Automatic preprocessing pipeline
* Real-time prediction
* Streamlit web application
* GPU-compatible training
* Clean and modular codebase

---

# Dataset

The project uses a publicly available **Satellite Land Cover Dataset** consisting of RGB satellite images belonging to different land cover classes.

Typical classes include:

* Annual Crop
* Forest
* Herbaceous Vegetation
* Highway
* Industrial Area
* Pasture
* Permanent Crop
* Residential Area
* River
* Sea/Lake

Each image represents a fixed geographical area captured from satellite imagery.

---

# Model Architecture

## Swin Transformer

The project utilizes the **Swin Transformer**, introduced by Microsoft Research.

Unlike Vision Transformers (ViT), Swin Transformer computes self-attention within local windows while gradually merging image patches into hierarchical feature representations.

### Key Characteristics

* Hierarchical architecture
* Shifted Window Multi-head Self Attention (SW-MSA)
* Linear computational complexity
* Multi-scale feature learning
* Excellent performance on image classification tasks

### Why Swin Transformer?

Traditional CNNs have limited receptive fields and require deeper architectures to capture global context.

Swin Transformer offers:

* Better contextual understanding
* Strong feature extraction
* Improved generalization
* Higher accuracy on remote sensing datasets

---

# Transfer Learning

Instead of training from scratch, a pretrained Swin Transformer model is used.

Benefits include:

* Faster convergence
* Reduced computational cost
* Better feature extraction
* Improved performance with limited data

The pretrained classification head is replaced with a custom classifier matching the number of land cover classes.

---

# Data Preprocessing

Before training, images undergo several preprocessing steps.

## Training Transformations

* Random Resized Crop
* Random Horizontal Flip
* Random Rotation
* Color Jitter
* Tensor Conversion
* Normalization

These augmentations improve model robustness and reduce overfitting.

## Validation Transformations

* Resize
* Tensor Conversion
* Normalization

---

# Training Pipeline

The training workflow consists of:

1. Loading the dataset
2. Applying preprocessing
3. Initializing pretrained Swin Transformer
4. Replacing the classifier layer
5. Defining loss function
6. Configuring optimizer
7. Training the network
8. Monitoring validation accuracy
9. Saving the best-performing model

---

# Loss Function

Cross Entropy Loss is used for multi-class classification.

```python
nn.CrossEntropyLoss()
```

---

# Optimizer

The model is optimized using **AdamW**, which provides better regularization for transformer architectures.

```python
torch.optim.AdamW()
```

---

# Learning Rate Scheduler

A learning rate scheduler is used to gradually reduce the learning rate during training, improving convergence and helping the model reach better performance.

---

# Evaluation Metrics

The model is evaluated using:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss
* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1 Score

---

# Project Structure

```
Satellite-Land-Cover-Classifier/
│
├── dataset/
│
├── notebooks/
│     ├── training.ipynb
│     └── inference.ipynb
│
├── models/
│     └── best_model.pth
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
│
├── assets/
│     └── screenshots
│
└── utils/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Lavanya-Manila/satellite-land-cover-classifier.git
```

Navigate to the project folder

```bash
cd Satellite-Land-Cover-Classifier
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit app using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# How to Use

1. Open the Streamlit application.
2. Upload a satellite image.
3. The image is automatically preprocessed.
4. The trained Swin Transformer predicts the land cover category.
5. The predicted class and confidence score are displayed.

---

# Sample Workflow

```
Satellite Image
        │
        ▼
Image Preprocessing
        │
        ▼
Swin Transformer
        │
        ▼
Feature Extraction
        │
        ▼
Classification Head
        │
        ▼
Predicted Land Cover Class
```

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Torchvision
* NumPy
* Pillow
* Matplotlib
* Scikit-learn
* Streamlit
* Google Colab

---

# Applications

This project can be applied in:

* Urban planning
* Environmental monitoring
* Agricultural analysis
* Forest management
* Water resource monitoring
* Climate change studies
* Disaster management
* Geographic Information Systems (GIS)
* Remote sensing research

---

# Future Improvements

* Integrate Grad-CAM for model explainability.
* Support larger satellite imagery through tiling.
* Add batch image prediction.
* Deploy the application on Streamlit Community Cloud or Hugging Face Spaces.
* Experiment with Swin Transformer V2 and other transformer architectures.
* Add support for geospatial image formats such as GeoTIFF.
* Incorporate active learning for continuous model improvement.

---

# Results

The fine-tuned Swin Transformer demonstrates strong performance in distinguishing multiple land cover categories by leveraging hierarchical attention mechanisms and transfer learning. It achieves robust feature extraction from satellite imagery while maintaining efficient inference, making it well suited for real-world remote sensing applications.

> **Note:** Update this section with your final training metrics, such as training accuracy, validation accuracy, F1-score, and confusion matrix, once available.

---

# Learning Outcomes

This project demonstrates practical experience with:

* Vision Transformers
* Swin Transformer architecture
* Transfer Learning
* Fine-tuning pretrained models
* Image preprocessing and augmentation
* PyTorch deep learning workflows
* Model evaluation
* Deployment using Streamlit
* End-to-end computer vision project development

---

# Acknowledgements

This project builds upon the open-source deep learning ecosystem, including PyTorch, Torchvision, Hugging Face Transformers, Streamlit, and the contributors who provide publicly available satellite imagery datasets for research and education.

---

# License

This project is released under the MIT License.

You are free to use, modify, and distribute this project for educational and research purposes.

---

# Author

**Lavanya Manila**

Computer Science Student | AI & Machine Learning Enthusiast

Interested in:

* Computer Vision
* Deep Learning
* Remote Sensing
* Machine Learning
* Vision Transformers
* Intelligent Systems

If you found this project helpful, consider giving the repository a ⭐ to support future development.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://satelitelandcoverclassification-lbce9dauhbgvtmrg24toyn.streamlit.app/)