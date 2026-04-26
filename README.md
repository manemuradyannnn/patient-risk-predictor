**Patient Risk Predictor (PyTorch)**
This project implements a simple neural network using PyTorch to classify patient health risk based on vital signs. The model takes in basic patient data such as age, heart rate, oxygen level, and temperature, and predicts whether the patient is low risk (0) or high risk (1).
The goal of this project is to demonstrate fundamental machine learning concepts, including data representation, model construction, loss computation, and training using gradient-based optimization.

**Features**
- Binary classification using a neural network
- Built with PyTorch
- Uses real-world inspired medical features
- End-to-end pipeline: data → model → training → prediction
- Demonstrates core ML concepts such as loss, backpropagation, and optimization
- Input Data

Each patient is represented by a vector of four features:
1. Age (years)
2. Heart Rate (beats per minute)
3. Oxygen Level (%)
4. Body Temperature (°C)

Example input:
[70, 110, 90, 38.5]

**Output**
The model outputs a probability between 0 and 1:
Closer to 0 → Low Risk
Closer to 1 → High Risk

_A threshold of 0.5 is used to convert probabilities into class predictions._

**Model Architecture**
The model is a simple feedforward neural network:
Linear layer (4 → 1)
Sigmoid activation function
Mathematically, the model learns a function of the form:
risk = w1*x1 + w2*x2 + w3*x3 + w4*x4 + bias
followed by a sigmoid transformation to produce a probability.

**Training**
The model is trained using:
- Loss Function: Binary Cross Entropy Loss (nn.BCELoss)
- Optimizer: Adam (torch.optim.Adam)
- Training loop with forward pass, loss computation, backpropagation, and parameter updates

During training, the loss decreases as the model improves its predictions.

**Results**
The trained model produces probabilities that are very close to 0 or 1 for the training data, indicating high confidence in its predictions. When converted to class labels, the predictions closely match the true labels.
