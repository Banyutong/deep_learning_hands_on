# 交大密院Deep learning学习手册
# # UM-SJTU JI Deep learning Hands-on Tutorial
# Session 4 - Long Short-Term Memory (LSTM) Networks on Time Series Data

## Table of Contents

- [Introduction](#introduction)
- [Dataset Preparation](#dataset-preparation)
- [Building the LSTM Network](#building-the-lstm-network)
- [Training and Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)

---

## Introduction

In this fourth session, we'll explore Long Short-Term Memory (LSTM) Networks, a type of recurrent neural network (RNN) well-suited for sequence-based problems. We'll use a simple time series dataset to predict future values based on past observations.

The theoretical part of the LSTM can be found in MIT course Lecture 2. You can refer to the [course official website](http://introtodeeplearning.com/) for more in depth understanding.

---

## Dataset Preparation

### Step 1: Generate Time Series Data

For the sake of simplicity, we'll generate synthetic sine wave data. 

```python
import numpy as np
import torch

# Generate sine wave data
seq_length = 20
num_samples = 1000

time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data = data[:-1].reshape(-1, seq_length)

# Duplicate and stack the sequence
X = np.repeat(data, num_samples, axis=0)
y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(2)  # Add an extra dimension for the single feature
y_tensor = torch.FloatTensor(y)
```

---

## Building the LSTM Network

### Step 3: Define the Model

```python
import torch.nn as nn

class TimeSeries_LSTM(nn.Module):
    def __init__(self):
        super(TimeSeries_LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 50)  # 1 input feature, 50 hidden units
        self.fc = nn.Linear(50, 1)  # 50 input features, 1 output feature

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(1), 50)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(1), 50)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
```

---

## Training and Evaluation

### Step 4: Initialize the Model, Loss, and Optimizer

```python
model = TimeSeries_LSTM()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
```

### Step 5: Train the Model

```python
# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```
### Step 6: Evaluate the Model

In this step, we'll evaluate the model's performance by generating new sine wave data and comparing the predicted values to the actual ones.

#### Generate New Data for Testing

First, we'll generate a new set of sine wave sequences for testing. Let's create 100 new samples.

```python
num_test_samples = 100

# Generate new sine wave data
test_time_steps = np.linspace(0, np.pi, seq_length + 1)
test_data = np.sin(test_time_steps)
test_data = test_data[:-1].reshape(-1, seq_length)

# Duplicate and stack the sequence
test_X = np.repeat(test_data, num_test_samples, axis=0)
test_y = np.sin(test_time_steps[-1] * np.ones((num_test_samples, 1)))

# Convert to PyTorch tensors
test_X_tensor = torch.FloatTensor(test_X).unsqueeze(2)  # Add an extra dimension for the single feature
test_y_tensor = torch.FloatTensor(test_y)
```

#### Predict and Compare

Next, we'll use the trained model to predict the future values of the test sequences and compare them to the actual future values.

```python
# Disable gradient computation
with torch.no_grad():
    predicted = model(test_X_tensor)

    # Calculate the Mean Squared Error
    mse = criterion(predicted, test_y_tensor).item()
    print(f'Mean Squared Error on test data: {mse:.4f}')

# Convert predictions to numpy array
predicted_np = predicted.numpy()

# Visualize a sample test sequence and prediction
import matplotlib.pyplot as plt

plt.plot(test_time_steps[:-1], test_X[0], label='Input Sequence')
plt.scatter([test_time_steps[-1]], [test_y[0]], label='Actual Future Value', c='r')
plt.scatter([test_time_steps[-1]], [predicted_np[0]], label='Predicted Future Value', c='g')
plt.legend()
plt.show()
```

In this visualization, the blue line represents the input sequence, the red point is the actual future value, and the green point is the predicted future value.

---
### Conclusion

In this session, you've learned how to build a simple LSTM network to predict future values of a time series sequence. This synthetic example should give you a foundational understanding of how to work with LSTM networks for more complex sequence-based tasks.

Feel free to modify the architecture or try using real-world time series data to further your understanding.
