from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from embedding_logic import get_embedding

enhanced_doc_embeddings = torch.load('embeddings/enhanced_doc_embeddings.pt')
doc_embeddings = torch.load('embeddings/doc_embeddings.pt')

query = "Juros Brasil Eleição"
query_embedding = get_embedding([query.lower()])

# Convert embeddings to NumPy arrays for plotting
doc_embeddings_np = doc_embeddings.detach().numpy()
enhanced_doc_embeddings_np = enhanced_doc_embeddings.detach().numpy()
query_embedding_np = query_embedding.detach().numpy()

# Apply PCA to reduce dimensions to 50 before applying t-SNE
pca = PCA(n_components=10)
embeddings_pca = pca.fit_transform(doc_embeddings_np)

# Apply t-SNE to the PCA-reduced embeddings
tsne = TSNE(n_components=2, random_state=42, learning_rate=100, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_pca)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, label='Document Embeddings')
plt.scatter(query_embedding_np[:, 0], query_embedding_np[:, 1], c='red', label='Query Embedding')
plt.title("t-SNE Visualization of Original Document Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()

# save figure to png
plt.savefig('tsne_plot_original.png')

pca = PCA(n_components=10)
embeddings_pca = pca.fit_transform(enhanced_doc_embeddings_np)
# Apply t-SNE to the PCA-reduced embeddings
tsne = TSNE(n_components=2, random_state=42, learning_rate=100, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_pca)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, label='Document Embeddings')
plt.scatter(query_embedding_np[:, 0], query_embedding_np[:, 1], c='red', label='Query Embedding')
plt.title("t-SNE Visualization of Enhanced Document Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()

# save figure to png
plt.savefig('tsne_plot_enhanced.png')

pca = PCA(n_components=10)
embeddings_pca = pca.fit_transform(enhanced_doc_embeddings_np)
# plot 3d graph to visualize and genertae 3d embeddings from enhanced embeddings
tsne = TSNE(n_components=3, random_state=42, learning_rate=100, perplexity=30)
embeddings_3d = tsne.fit_transform(embeddings_pca)
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.scatter3D(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='blue', alpha=0.6, label='Document Embeddings')
ax.scatter3D(query_embedding_np[:, 0], query_embedding_np[:, 1], query_embedding_np[:, 2], c='red', label='Query Embedding')
plt.title("3D Visualization of Enhanced Document Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()

# save figure to png
plt.savefig('tsne_plot_enhanced_3d.png')
pca = PCA(n_components=10)
embeddings_pca = pca.fit_transform(doc_embeddings_np)
# plot 3d graph to visualize and genertae 3d embeddings from original embeddings
tsne = TSNE(n_components=3, random_state=42, learning_rate=100, perplexity=30)
embeddings_3d = tsne.fit_transform(doc_embeddings_np)
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.scatter3D(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='blue', alpha=0.6, label='Document Embeddings')
ax.scatter3D(query_embedding_np[:, 0], query_embedding_np[:, 1], query_embedding_np[:, 2], c='red', label='Query Embedding')
plt.title("3D Visualization of Original Document Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()

# save figure to png
plt.savefig('tsne_plot_original_3d.png')
