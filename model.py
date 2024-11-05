import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot  as plt
from embedding import load_glove_embeddings, text_to_embedding

import torch


class multi_layered_perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(multi_layered_perceptron, self).__init__()
        # self.embedding_layer = embedding
        print(input_size, hidden_size, num_classes)
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        # x = self.embedding_layer(x)
        out = self._pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def _pool(self, out):
        return out.mean(dim=1)
    
glove_path = 'glove.6B/glove.6B.50d.txt'
embedding_dim = 50
vocab, embeddings_matrix = load_glove_embeddings(glove_path, embedding_dim)
# save vocab
torch.save(vocab, 'embeddings/vocab.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)
df = pd.read_csv('cnn.csv', sep=',')
df['content'] = df['content'].apply(lambda x: x.lower())
torch.save(embeddings_matrix, 'embeddings/matrix.pt')
doc_embeddings = torch.stack([text_to_embedding(text, vocab, embedding_layer) for text in df['content']])

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

reduced_dim = 32 # arbitrary
autoencoder = EmbeddingAutoencoder(embedding_dim, reduced_dim)

# Training loop for the autoencoder on document embeddings
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for embedding in doc_embeddings: 
        optimizer.zero_grad()
        reconstructed, encoded = autoencoder(embedding)
        loss = criterion(reconstructed, embedding)  # reconstruction loss
        loss.backward()
        optimizer.step()

enhanced_doc_embeddings = torch.stack([autoencoder(embedding)[1] for embedding in doc_embeddings])
#model = multi_layered_perceptron(enhanced_doc_embeddings.shape[0], 50, 2)
# mlp_doc_embeddings =  model(enhanced_doc_embeddings) # torch.stack([model(embedding)[1] for embedding in doc_embeddings])
# print(mlp_doc_embeddings.shape)

# store enhanced_doc_embeddings
torch.save(doc_embeddings, 'embeddings/doc_embeddings.pt')
torch.save(enhanced_doc_embeddings, 'embeddings/enhanced_doc_embeddings.pt')
# torch.save(mlp_doc_embeddings, 'embeddings/mlp_doc_embeddings.pt')