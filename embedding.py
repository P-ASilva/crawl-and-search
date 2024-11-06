import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dowload import get_sentence_embedding
import torch
import numpy as np

# Load GloVe embeddings and build vocabulary
def load_glove_embeddings(filepath, embedding_dim=50):
    vocab = {}
    embeddings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = values[1:]
            if len(vector) == embedding_dim:  # Ensure correct dimension
                vocab[word] = len(embeddings)
                embeddings.append(np.asarray(vector, dtype='float32'))
    embeddings = np.array(embeddings)
    return vocab, torch.tensor(embeddings, dtype=torch.float32)

# Convert text to mean embedding
def text_to_embedding(text, vocab, embedding_layer):
    tokens = text.lower().split()
    indices = [vocab.get(token, 0) for token in tokens]  # 0 for unknown tokens
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    word_embeddings = embedding_layer(indices_tensor)
    mean_embedding = word_embeddings.mean(dim=0)
    return mean_embedding

glove_path = 'glove.6B/glove.6B.50d.txt'  # Update with the correct GloVe file path
embedding_dim = 50
vocab, embeddings_matrix = load_glove_embeddings(glove_path, embedding_dim)
embedding_layer = torch.nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)

q = "Lula"
# read cnn.csv with indexes 
df = pd.read_csv('cnn.csv', sep=',')
df['content'] = df['content'].apply(lambda x: x.lower())

# Convert documents and query to embeddings
doc_embeddings = torch.stack([text_to_embedding(text, vocab, embedding_layer) for text in df['content']])
query_embedding = text_to_embedding(q, vocab, embedding_layer).unsqueeze(0)

# Calculate cosine similarity between query and documents
similarity_scores = cosine_similarity(query_embedding.detach().numpy(), doc_embeddings.detach().numpy()).flatten()

# Get top 10 most relevant results
top_indices = similarity_scores.argsort()[-10:][::-1]
top_df = df.iloc[top_indices]
top_df['relevance'] = similarity_scores[top_indices]

# Format results
json_dict = {
    title: {
        "subtitle": subtitle,
        "content": content[:500*4],  # Shortened content
        "relevance": relevance
    } for title, subtitle, content, relevance in zip(top_df["title"], top_df['subtitle'], top_df['content'], top_df['relevance'])
}

# Sort and filter by relevance threshold
sorted_results = sorted(json_dict.items(), key=lambda x: x[1]['relevance'], reverse=True)
ordered_results = [result[1] for result in sorted_results if result[1]['relevance'] >= 0.1]
json_dict = {"results": ordered_results}

# print(json_dict)
