import torch
import pandas as pd
import torch.nn as nn
from embedding_logic import get_embedding, EmbeddingAutoencoder

# Load and preprocess the dataset
df = pd.read_csv('data/cnn.csv', sep=',')
df['content'] = df['content'].apply(lambda x: x.lower())

# Convert documents and query to BERT embeddings in batches
doc_texts = df['content'].tolist()
doc_embeddings = get_embedding(doc_texts)

torch.save(doc_embeddings, 'doc_embeddings.pt')

# embedding_layer = torch.nn.Embedding.from_pretrained(doc_embeddings, freeze=True)
reduced_dim = 32 # arbitrary
autoencoder = EmbeddingAutoencoder(doc_embeddings.shape[1], reduced_dim)
print("autoencoder created")
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
print(enhanced_doc_embeddings.shape)
torch.save(enhanced_doc_embeddings, 'enhanced_doc_embeddings.pt')

# run cosine similarity on the embeddings
# query = "juros"
# query_embedding = get_embedding([query.lower()])
# print(query_embedding.shape)
# query_embedding = autoencoder(query_embedding)[1].squeeze()  # Get the encoded query embedding, should be (32,)
# # Calculate Euclidean distance instead of cosine similarity
# distances = torch.cdist(enhanced_doc_embeddings, query_embedding.unsqueeze(0), p=2)
# # Get the indices of the top 10 closest documents
# print(distances)
# _, idx = distances.topk(10)

# #similarity = nn.functional.cosine_similarity(enhanced_doc_embeddings, query_embedding.expand_as(enhanced_doc_embeddings), dim=1)
# # print(similarity)
# # idx = similarity.argsort()[-10:][::-1]
# dff = df.loc[idx]
# dff['relevance'] = distances[idx].tolist()
# # Expand query_embedding to match the batch dimension of enhanced_doc_embeddings
# # query_embedding = query_embedding.unsqueeze(0).expand_as(enhanced_doc_embeddings) 
# # similarity = nn.functional.cosine_similarity(enhanced_doc_embeddings, query_embedding, dim=1)
# # # save autoencoder
# # torch.save(autoencoder, 'autoencoder.pt')

# # _, idx = similarity.topk(100)

# dff = df.loc[idx]
# dff['relevance'] = distances[idx].tolist()
# print(dff[['title', 'relevance']])