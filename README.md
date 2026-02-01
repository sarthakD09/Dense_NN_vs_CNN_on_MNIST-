Dense_NN_vs_CNN_on_MNIST-




Handwritten Digit Recognition — Dense NN vs CNN on MNIST

This project explores how different neural network architectures perform on the MNIST handwritten digit dataset.

We first build a simple Dense Neural Network, evaluate its weaknesses, and then improve performance using a Convolutional Neural Network (CNN).

---
Dataset

MNIST: 60,000 training and 10,000 testing images of handwritten digits (28x28 grayscale).

---
Model 1 — Dense Neural Network

Architecture:
- Flatten (784)
- Dense (128, ReLU)
- Dense (10, Softmax)

Results

- Test Accuracy: ~97%
- Common Confusions: 5 vs 3, 9 vs 4

![Confusion Matrix Dense](images/confusion_dense.png)

Observations

The model struggles when digits are spatially shifted or written messily because flattening destroys spatial information.

---

Model 2 — Convolutional Neural Network (CNN)

Architecture:
- Conv2D → MaxPool
- Conv2D → MaxPool
- Dense → Output

Results

- Test Accuracy: ~99%

![CNN Filters](images/filters_cnn.png)

Observations

CNN preserves spatial structure and learns edge/shape detectors, leading to better generalization.

---

Key Learning

Dense networks treat images as numbers.  
CNN treats images as shapes.

---

How to Run

```bash
pip install -r requirements.txt
jupyter notebook
