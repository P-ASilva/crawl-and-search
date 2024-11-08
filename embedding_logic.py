import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Load BERT model and tokenizer
def load_BERT_model():
    model_name = "neuralmind/bert-base-portuguese-cased"  # BERT model for Portuguese
    model = AutoModel.from_pretrained(model_name)
    return model

def load_BERT_tokenizer():
    model_name = "neuralmind/bert-base-portuguese-cased"  # BERT model for Portuguese
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_embedding(texts, batch_size=8):
    embeddings = []
    tokenizer = load_BERT_tokenizer()
    model = load_BERT_model()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over token embeddings to get 2D embedding
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, embedding_dim, reduced_dim):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Linear(embedding_dim, reduced_dim)
        self.decoder = nn.Linear(reduced_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded  