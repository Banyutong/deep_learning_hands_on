# 交大密院Deep learning上手手册
Simple Deep Learning Tutorial with PyTorch
Table of Contents
Introduction
System Requirements
Environment Setup
Building a Simple Neural Network
Training and Evaluation
Conclusion
Introduction
This tutorial aims to walk you through the basics of setting up a Deep Learning environment and building a simple Neural Network model using Python and PyTorch.

System Requirements
Operating System: Windows, macOS, or Linux
Python 3.7 or above
Minimum 4GB RAM (8GB recommended)
Environment Setup
Step 1: Install Python
Download Python from the official website.
Follow the installation guide according to your OS.
Step 2: Install Pip (If not installed)
Most Python installations come with pip pre-installed. You can check its installation by running pip --version.
Step 3: Create a Virtual Environment (Optional but Recommended)
Run python3 -m venv myenv to create a virtual environment named myenv.
Activate it:
On Windows: myenv\Scripts\Activate
On macOS and Linux: source myenv/bin/activate
Step 4: Install Required Packages
Install the following packages:

PyTorch: For building and training neural networks.
NumPy: For numerical operations.
bash
Copy code
pip install torch numpy
Building a Simple Neural Network
Step 1: Import Libraries
python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
Step 2: Prepare Data
We'll use synthetic data for this example.

python
Copy code
# Training Data
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Testing Data
X_test = torch.tensor([[0.1, 0.0], [0.0, 0.9], [1.1, 0.0], [0.9, 1.0]], dtype=torch.float32)
Step 3: Define the Model
python
Copy code
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
Training and Evaluation
Step 4: Initialize the Model and Define Loss and Optimizer
python
Copy code
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
Step 5: Train the Model
python
Copy code
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
Step 6: Evaluate the Model
python
Copy code
with torch.no_grad():
    test_outputs = model(X_test)
    print(f'Test Outputs: {test_outputs}')
Conclusion
In this tutorial, we walked through setting up a Python environment, installing PyTorch, and building a simple Neural Network. You should now be able to start building more complex models for various applications.

That's the end of the tutorial! I hope you found it helpful.
