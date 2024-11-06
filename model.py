import pandas as pd
import torch
import torch.nn as nn
from embedding import load_glove_embeddings, text_to_embedding

import torch
    
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

torch.save(doc_embeddings, 'embeddings/doc_embeddings.pt')
torch.save(enhanced_doc_embeddings, 'embeddings/enhanced_doc_embeddings.pt')