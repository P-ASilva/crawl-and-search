# load embeddings from embeddings folder containing .pt file
from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np

from embedding import load_glove_embeddings, text_to_embedding


enhanced_doc_embeddings = torch.load('embeddings/enhanced_doc_embeddings.pt')
doc_embeddings = torch.load('embeddings/doc_embeddings.pt')
origin = torch.load('embeddings/matrix.pt')
# load vocab
vocab = torch.load('embeddings/vocab.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(origin, freeze=True)
query = "Lula Juros Brasil"
query_embedding = text_to_embedding(query.lower(), vocab, embedding_layer).unsqueeze(0)

# Convert embeddings to NumPy arrays for plotting
doc_embeddings_np = doc_embeddings.detach().numpy()
enhanced_doc_embeddings_np = enhanced_doc_embeddings.detach().numpy()
query_embedding_np = query_embedding.detach().numpy()

import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Apply PCA to reduce dimensions to 50 before applying t-SNE
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(doc_embeddings_np)

# Apply t-SNE to the PCA-reduced embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, label='Document Embeddings')
plt.scatter(query_embedding_np[:, 0], query_embedding_np[:, 1], c='red', label='Query Embedding')
plt.title("t-SNE Visualization of Enhanced Document Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.show()
