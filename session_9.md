# Deep Learning Tutorial with PyTorch: Session 9 - Applying Multi-Head Attention and Transformers for Sentiment Analysis

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Building the Transformer Model](#building-the-transformer-model)
- [Training the Transformer](#training-the-transformer)
- [Evaluation and Testing](#evaluation-and-testing)
- [Conclusion](#conclusion)

---

## Introduction

In this session, we'll deploy Transformers, particularly focusing on the Multi-Head Attention mechanism, for Sentiment Analysis on text data, categorizing it into positive or negative sentiment.

---

## Prerequisites

Ensure to have:

- PyTorch
- Torchtext
- Spacy (for tokenization)

```bash
pip install torch torchtext spacy
python -m spacy download en_core_web_sm
```

---

## Data Preprocessing

### Step 1: Load and Preprocess Data

```python
from torchtext.legacy import data
from torchtext.legacy import datasets

TEXT = data.Field(tokenize='spacy', batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Build the vocabulary and load default word embeddings
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Split data and create iterators
train_data, valid_data = train_data.split()
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=64,
    device=device)
```

---

## Building the Transformer Model

### Step 2: Define the Transformer Model

```python
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, hid_dim, n_layers, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hid_dim),
            num_layers=n_layers
        )
        self.fc = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        #text = [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        #embedded = [batch_size, seq_len, emb_dim]
        transformed = self.transformer(embedded)
        #transformed = [batch_size, seq_len, emb_dim]
        prediction = self.fc(transformed[:, -1, :])
        #prediction = [batch_size, output_dim]
        return prediction
```

---

## Training the Transformer

### Step 3: Training Loop

Define the hyperparameters, optimizer, criterion, and execute the training loop.

```python
# Hyperparameters, optimizer, and criterion
# ...

# Training loop
for epoch in range(num_epochs):
    # Training and validation code
    # ...
```

---

## Evaluation and Testing

### Step 4: Evaluating the Model

After training, evaluate the model using accuracy or any other suitable metric on your validation/test set and utilize the model for predicting sentiment on new sentences.

```python
def predict_sentiment(model, sentence, text_field):
    model.eval()
    tokenized = [tok.text for tok in spacy_en.tokenizer(sentence)]
    indexed = [text_field.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
```

---

## Conclusion

In Session 9, you deployed a Transformer model to perform sentiment analysis on the IMDB dataset. Transformers are versatile and have been extensively applied across numerous NLP applications, showcasing exemplary performance, especially in understanding contextual information in text. Continue exploring further applications and enhancing model architectures to deepen your understanding!
