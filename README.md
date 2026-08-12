# MNIST Digit Classification (CNN vs RNN)

This project implements and compares **Convolutional Neural Network (CNN)** and **Recurrent Neural Network (RNN - LSTM)** models for handwritten digit classification using the **MNIST dataset**.  
It highlights how different deep learning architectures perform on the same dataset, with evaluation metrics and visualizations.

---

## Features
- CNN model with 3 convolutional layers + fully connected layers
- RNN model using LSTM
- Training & evaluation on MNIST dataset
- Confusion matrix visualization (Seaborn heatmap)
- Classification report (precision, recall, F1-score)

---

## 📊 Dataset
- **Source**: MNIST (via `torchvision.datasets`)  
- **Training samples**: 60,000  
- **Test samples**: 10,000  
- **Image size**: 28×28 grayscale  
- **Classes**: Digits 0–9  

---

## Model Architectures

### CNN
- Conv → ReLU → MaxPool (×3 layers)  
- Fully connected layers (256 → 10 classes)  

### RNN (LSTM)
- Input size: 28 (pixels per row)  
- Hidden size: 128  
- Fully connected layers (64 → 10 classes)  

---

##  Training Setup
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Epochs: 5  
- Batch Size: 64  

---

##  Results
- **CNN Accuracy**: ~98%  
- **RNN Accuracy**: ~92–95%  
- CNN outperforms RNN due to spatial feature extraction.  

---

##  Visualizations
- Confusion matrices plotted using **Seaborn heatmaps**  
- Classification reports generated using **scikit-learn**  
