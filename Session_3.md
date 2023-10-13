# 交大密院Deep learning学习手册
# # UM-SJTU-JI Deep learning Hands-on Tutorial
# Session 3 - Convolutional Neural Networks (CNNs)

## Table of Contents

- [Introduction](#introduction)
- [Dataset Preparation](#dataset-preparation)
- [Building the CNN](#building-the-cnn)
- [Training and Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)

---

## Introduction

In this third session, we'll explore Convolutional Neural Networks (CNNs), a type of neural network that is particularly powerful for tasks related to image recognition. We'll continue to use the MNIST dataset to compare the performance of CNNs with the previously implemented fully-connected network.

---

## Dataset Preparation

We will use the same dataset loading code from Session 2. If you haven't prepared your dataset yet, refer to the [Dataset Preparation](#dataset-preparation) section of Session 2.

---

## Building the CNN

### Step 3: Define the Model

CNNs typically consist of a series of convolutional layers followed by pooling layers, fully connected layers, and normalization layers. Here's how to define a simple CNN in PyTorch:

```python
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 input channel, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 5) # 32 input channels, 64 output channels, 5x5 kernel
        self.fc1 = nn.Linear(1024, 128)  # Fully connected layer: 1024 input features, 128 output features
        self.fc2 = nn.Linear(128, 10)    # Fully connected layer: 128 input features, 10 output features (for 10 classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 1024)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

This model uses two convolutional layers (`conv1` and `conv2`), followed by two fully connected layers (`fc1` and `fc2`). The activation function we're using is ReLU (Rectified Linear Unit), and we use max pooling for the pooling layers.

---

## Training and Evaluation

### Step 4: Initialize the Model and Define Loss and Optimizer

```python
model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

### Step 5: Train the Model

```python
# Training Loop
for epoch in range(5):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print loss
        if i % 500 == 0:
            print(f'Epoch [{epoch + 1}/5], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item()}')
```

### Step 6: Evaluate the Model

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

---

## Conclusion

In this session, you've learned how to build a simple Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. CNNs are more adept at capturing spatial hierarchies of features in images compared to fully connected networks, making them a standard choice for computer vision tasks.

---

And that concludes Session 3 of our tutorial series! Feel free to modify the architecture or hyperparameters to see how it affects the model's performance.
