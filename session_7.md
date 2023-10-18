# UM-SJTU-JI Deep learning Hands-on Tutorial 
# Session 7 - Image Reconstruction using Variational Autoencoder (VAE)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Building the VAE](#building-the-vae)
- [Training the VAE](#training-the-vae)
- [Testing and Visualization](#testing-and-visualization)
- [Conclusion](#conclusion)

---

## Introduction

In this session, we will implement a Variational Autoencoder (VAE) to reconstruct images and visualize the results. VAEs are a type of generative model that learn a latent representation of input data.

The theoretical part of VAE can be found in MIT course Lecture 4. You can refer to [course official website](http://introtodeeplearning.com/) for more in depth understanding.

---

## Prerequisites

Ensure you've installed:

- PyTorch
- Torchvision
- Numpy
- Matplotlib

```bash
pip install torch torchvision numpy matplotlib
```

---

## Dataset and Preprocessing

### Step 1: Load and Preprocess Data

We will use the MNIST dataset for simplicity. 

```python
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
```

---

## Building the VAE

### Step 2: Define the VAE Architecture

```python
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(),
            nn.Linear(400, 40)  # 20 for mean and 20 for log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x.view(-1, 28*28))
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
```

---

## Training the VAE

### Step 3: Train the Model

We'll use a combination of a reconstruction loss (MSE) and the KL divergence to form the VAE loss.

```python
import torch.optim as optim

vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='sum')

def vae_loss(reconstructed_x, x, mu, log_var):
    recon_loss = criterion(reconstructed_x, x.view(-1, 28*28))
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    for batch, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

---

## Testing and Visualization

### Step 4: Test and Visualize Reconstructed Images

```python
import matplotlib.pyplot as plt

# Test the trained VAE on a batch of images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Original Images
fig, axes = plt.subplots(figsize=(12,3), ncols=6, nrows=2)
for i in range(6):
    axes[0,i].imshow(images[i].numpy().squeeze(), cmap='gray')
    axes[0,i].axis('off')

# Reconstructed Images
with torch.no_grad():
    recon_images, _, _ = vae(images)
    recon_images = recon_images.view(-1, 1, 28, 28)
    for i in range(6):
        axes[1,i].imshow(recon_images[i].numpy().squeeze(), cmap='gray')
        axes[1,i].axis('off')

plt.show()
```

The upper row in the plot will display the original images, while the lower row displays the reconstructed images after being encoded and decoded by the VAE.

---

## Conclusion

In this session, you learned to implement a Variational Autoencoder with PyTorch, trained it on the MNIST dataset, and visualized the reconstructed images. VAEs can be utilized for various applications like generating new data that's similar to the training data, and more. Explore further to dive deep into generative modeling!
