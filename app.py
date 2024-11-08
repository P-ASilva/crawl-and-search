from flask import Flask
from flask import request
import torch.nn as nn
import pandas as pd
from flask import jsonify
import torch
from embedding_logic import get_embedding, EmbeddingAutoencoder

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/query")
def query():
    q = request.args.get('query')
    # read cnn.csv with indexes 
    df = pd.read_csv('data/cnn.csv', sep=',')

    # load embeddings
    enhanced_doc_embeddings = torch.load('embeddings/enhanced_doc_embeddings.pt')
    print(enhanced_doc_embeddings.shape)
    autoencoder = torch.load('autoencoder.pt')
    query_embedding = get_embedding([q.lower()])
    print(query_embedding.shape)
    query_embedding = autoencoder(query_embedding)[1].squeeze()  # Get the encoded query embedding, should be (32,)
    # Calculate Euclidean distance instead of cosine similarity
    distances = torch.cdist(enhanced_doc_embeddings, query_embedding.unsqueeze(0), p=2)
    print(distances.shape)
    # Get the indices of the top 10 closest documents
    _, idx = distances.topk(10)
    dff = df.loc[idx]
    dff['relevance'] = distances[idx].tolist()

    json_dict = {title: {"subtitle":subtitle, "content": content[:500*4], "relevance": relevance} for title, subtitle, content, relevance in zip(dff["title"], dff['subtitle'], dff['content'], dff['relevance'])}
    sorted_results = sorted(json_dict.items(), key=lambda x: x[1]['relevance'], reverse=True)
    ordered_results = [result[1] for result in sorted_results if result[1]['relevance'] >= 0.1]
    json_dict = {"results": ordered_results}

    return jsonify(json_dict)
