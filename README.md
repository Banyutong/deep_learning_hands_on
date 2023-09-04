# 交大密院Deep learning上手手册


# Session 1: Simple Deep Learning Tutorial with PyTorch

## Table of Contents

- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Building a Simple Neural Network](#building-a-simple-neural-network)
- [Training and Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)

---

## Introduction

This tutorial aims to walk you through the basics of setting up a Deep Learning environment and building a simple Neural Network model using Python and PyTorch.

---

## System Requirements

- Operating System: Windows, macOS, or Linux
- Python 3.7 or above
- Minimum 4GB RAM (8GB recommended)

---

## Environment Setup

### Step 1: Install Python

- Download Python from the [official website](https://www.python.org/downloads/).
- Follow the installation guide according to your OS.

### Step 2: Install Pip (If not installed)

- Most Python installations come with `pip` pre-installed. You can check its installation by running `pip --version`.

### Step 3: Create a Virtual Environment (Optional but Recommended)

- Run `python3 -m venv myenv` to create a virtual environment named `myenv`.
- Activate it:
  - On Windows: `myenv\Scripts\Activate`
  - On macOS and Linux: `source myenv/bin/activate`

### Step 4: Install Required Packages

Install the following packages:
- PyTorch: For building and training neural networks.
- NumPy: For numerical operations.

```bash
pip install torch numpy
```

---

## Building a Simple Neural Network

### Step 1: Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
```

### Step 2: Prepare Data

We'll use synthetic data for this example.

```python
# Training Data
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Testing Data
X_test = torch.tensor([[0.1, 0.0], [0.0, 0.9], [1.1, 0.0], [0.9, 1.0]], dtype=torch.float32)
```

### Step 3: Define the Model

In this step, we'll define the architecture of the neural network we're building. Here's how we'll break it down:

#### Initialize the Model Class
We define a Python class called `SimpleNN` that inherits from `nn.Module`, which is the base class for all neural network modules provided by PyTorch.

```python
class SimpleNN(nn.Module):
```

#### Constructor (`__init__` method)
In the constructor, we initialize two linear layers using `nn.Linear()`. The first layer (named `layer1`) accepts input with 2 features and outputs 2 features. The second layer (`layer2`) takes 2 features as input and returns 1 output feature.

```python
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
```

#### Forward Method
The `forward` method defines the forward pass for this network. We use the sigmoid activation function (`torch.sigmoid()`) for the output of each layer.

```python
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
```

Here's the complete code for the model definition:

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
```

---

### Step 4: Initialize the Model and Define Loss and Optimizer

#### Initialize the Model
We initialize an instance of the `SimpleNN` model. At this point, PyTorch will also initialize the weights and biases for each layer.

```python
model = SimpleNN()
```

#### Define Loss Function
The loss function (`criterion`) calculates how far the model's predictions are from the true values. We use Mean Squared Error Loss (MSE Loss) for this example, which is suitable for regression problems.

```python
criterion = nn.MSELoss()
```

#### Define Optimizer
The optimizer adjusts the weights of the network to minimize the loss. We use Stochastic Gradient Descent (SGD) with a learning rate of 0.1.

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

Here's the complete code for this step:

```python
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

### Step 5: Train the Model

```python
# Training Loop
for epoch in range(10000):
    # Forward Pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward Pass and Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print Loss
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```

### Step 6: Evaluate the Model

```python
with torch.no_grad():
    test_outputs = model(X_test)
    print(f'Test Outputs: {test_outputs}')
```

---

## Conclusion

In this tutorial, we walked through setting up a Python environment, installing PyTorch, and building a simple Neural Network. You should now be able to start building more complex models for various applications.

---

That's the end of the tutorial! I hope you found it helpful.




