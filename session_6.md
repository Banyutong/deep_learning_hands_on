# UM-SJTU-JI Deep learning Hands-on Tutorial 
# Session 6 - Reinforcement Learning on OpenAI Gym Task

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Building the Agent](#building-the-agent)
- [Training the Agent](#training-the-agent)
- [Evaluation and Testing](#evaluation-and-testing)
- [Conclusion](#conclusion)

---

## Introduction

In this fifth session, we'll delve into reinforcement learning using PyTorch and apply it to a simple task in OpenAI Gym. We'll use the classic "CartPole" environment, where the aim is to balance a pole on a moving cart.

The theoretical part of reinforcement learning can be found in MIT course Lecture 6. Please refer to the [course official website](http://introtodeeplearning.com/) for more in depth understanding.

---

## Prerequisites

Ensure you've installed:

- PyTorch
- OpenAI Gym
- Numpy

```bash
pip install torch gym[box2d] numpy
```

---

## Environment Setup

### Step 1: Import Necessary Libraries and Initialize Environment


```python


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')
```

---

## Building the Agent

### Step 2: Define the Model (Q-network)

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_size, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        x = self.out(x)
        return x
```

### Step 3: Set up the Replay Memory

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
```

---

## Training the Agent

### Step 4: Initialize Q-Network and Replay Memory

```python
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
memory = ReplayBuffer(10000)
```

### Step 5: Implement ε-Greedy Strategy and Training Loop

```python
batch_size = 64
gamma = 0.99  # discount factor
epsilon = 1.0

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        
        # Store transition in memory
        memory.add(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward

        # Train Q-network with sampled transitions
        if len(memory.memory) >= batch_size:
            transitions = memory.sample(batch_size)
            batch = np.array(transitions, dtype=object).transpose()
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

            state_batch = torch.FloatTensor(np.stack(state_batch))
            action_batch = torch.LongTensor(np.array(action_batch, dtype=np.int32))
            # action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(np.array(reward_batch, dtype=np.float32))
            # reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(np.stack(next_state_batch))
            not_done_mask = torch.ByteTensor(~np.array(done_batch, dtype=np.uint8))

            current_q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))
            next_max_q_values = q_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (gamma * next_max_q_values * not_done_mask)

            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay ε
    if epsilon > 0.05:
        epsilon *= 0.995
    
    print(f'Episode {episode}, Total Reward: {total_reward}')
```

---

## Evaluation and Testing

### Step 6: Test the Agent
```python
import matplotlib.animation as animation

# Create a figure and axis to display the animation
fig, ax = plt.subplots()

# Test the trained agent for one episode and capture frames
def test_agent(env, trained_agent):
    frames = []
    state = env.reset()[0]
    done = False
    while not done:
        # Render to RGB array and append to frames
        frames.append(env.render())
        
        # Choose action
        with torch.no_grad():
            q_values = trained_agent(torch.FloatTensor(np.array(state)).unsqueeze(0))
            action = torch.argmax(q_values).item()
        
        # Take action
        state, _, done, _, _ = env.step(action)
    
    env.close()
    return frames

# Animate frames

def animate_frames(frames):
    img = plt.imshow(frames[0])  # initialize image with the first frame
    
    def update(frame):
        img.set_array(frame)
        return img,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    plt.axis('off')
    plt.show()

# Apply the functions

trained_agent = q_network  # Replace with your trained Q-network
frames = test_agent(env, trained_agent)
animate_frames(frames)
```
---

## Conclusion

In this session, you've learned how to implement a basic Q-learning agent using PyTorch to solve a simple problem from OpenAI Gym. Reinforcement learning is a broad field with many more concepts to explore, such as policy gradients, deep Q-networks, actor-critic methods, etc. Feel free to explore more and enhance your model!
