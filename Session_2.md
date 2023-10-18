# UM-SJTU-JI Deep learning Hands-on Tutorial
# Session 2: MNIST Classification

## Table of Contents

- [Introduction](#introduction)
- [Dataset Preparation](#dataset-preparation)
- [Building the Neural Network](#building-the-neural-network)
- [Training and Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)

---

## Introduction

In this second session, we'll extend our understanding of neural networks to tackle the MNIST handwritten digit classification problem.

The theoretical part of the simple Neural Network can be found in MIT course Lecture 1. You can refer to [course official website](http://introtodeeplearning.com/) for more in depth understanding.

---

## Dataset Preparation

### Step 1: Install Required Package

Install the torchvision package to easily load the MNIST dataset.

```bash
pip install torchvision
```

### Step 2: Load the Dataset

```python
import torchvision
import torchvision.transforms as transforms

# Data transformation (converting to Tensor and normalizing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Download and load the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
```

---

## Building the Neural Network

### Step 3: Define the Model

```python
class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

In this model, we have three fully connected layers (`fc1`, `fc2`, and `fc3`). We start by flattening the input 28x28 images into a 1D tensor of size 784. The first two layers use the ReLU activation function, while the output layer does not apply any activation function since we'll use it with Cross-Entropy Loss later.

---

## Training and Evaluation

### Step 4: Initialize the Model and Define Loss and Optimizer

```python
model = MNISTNN()
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

In this session, you learned how to build a neural network for classifying handwritten digits from the MNIST dataset. The skills you've gained should be broadly applicable to a variety of classification problems.

---

That concludes the second session of our tutorial! I hope you found it helpful for understanding how to tackle a classification problem using PyTorch and a real-world dataset.
