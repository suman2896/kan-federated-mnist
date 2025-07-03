# 🔐 Federated Learning with Kolmogorov–Arnold Networks (KAN) on MNIST

> Privacy-preserving and interpretable deep learning using Federated Learning + B-spline activations.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-yellow)

---

## 📌 Overview

This project implements **Federated Learning (FL)** using **Kolmogorov–Arnold Networks (KAN)** with **B-spline activation functions** on the MNIST digit classification task. The goal is to combine **privacy**, **accuracy**, and **interpretability** in a single deep learning framework.

The model is trained across multiple **non-IID clients** without sharing raw data, using the **Federated Averaging (FedAvg)** algorithm.

---

## ✨ Key Features

- ✅ Federated training across simulated clients  
- 🔄 Kolmogorov–Arnold Networks with B-spline activation  
- 🔍 Learnable and interpretable activations  
- 📉 Graphs for accuracy & loss per round  
- 🖼️ Digit prediction demo on unseen test images  
- 🔒 Preserves client data privacy  

---


## 🧠 How It Works

1. The server initializes a **global KAN model**.  
2. Each client receives a copy and trains it locally on **non-IID MNIST digits**.  
3. Clients send back **model weights only** (no data).  
4. The server uses **FedAvg** to aggregate all models.  
5. The process repeats for multiple rounds.  
6. Final model is evaluated on global test set.  

---

## 🔧 Setup Instructions

### ✅ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- torchvision  
- matplotlib  

### ⚙️ Installation

```bash
git clone https://github.com/suman2896/kan-federated-mnist.git
cd kan-federated-mnist
pip install -r requirements.txt
```

---


## 📈 Example Output

<p align="center">
  <img src="assets/loss_accuracy_plot.png" width="450"/>
</p>

Accuracy improves steadily across communication rounds.  
Spline activations adapt to edge patterns and shape details.

---

## 🖼️ Sample Predictions

```python
Predicted: 3, Actual: 3
Predicted: 7, Actual: 7
```

---

## 🧪 Dataset

- MNIST Digits (0–9), 28×28 grayscale images  
- 60,000 training samples / 10,000 test samples  
- Simulated **non-IID split** across 5 clients  

---

## 📬 Contributing

Contributions are welcome and appreciated! If you’d like to help improve this project:

1. **Fork** the repository  
2. **Clone** your fork  
   ```bash
   git clone https://github.com/suman2896/kan-federated-mnist.git
   ```
3. **Create a new branch**  
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit them  
5. **Push to your fork** and submit a **Pull Request**

Make sure your code follows clean coding standards and is well-documented.  
If you're contributing a new feature or improvement, please also update the documentation or usage instructions as needed.

