from flask import Flask
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from bs4 import BeautifulSoup
import requests
import pandas as pd
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/query")
def query():
    q = request.args.get('query')
    # read cnn.csv with indexes 
    df = pd.read_csv('cnn.csv', sep=',')
    vectorizer = TfidfVectorizer()
    df['content'] = df['content'].apply(lambda x: x.lower().strip())

    X = vectorizer.fit_transform(df['content'])

    Q = vectorizer.transform([str(q).lower().strip()])
    R = X @ Q.T
    R = R.toarray().flatten()

    idx = R.argsort()[-10:]
    
    dff = df.loc[idx]
    dff['relevance'] = R.argsort()[-10:].tolist()
    print(dff['relevance'])

    json_dict = {title: {"subtitle":subtitle, "content": content[:500*4], "relevance": relevance} for title, subtitle, content, relevance in zip(dff["title"], dff['subtitle'], dff['content'], dff['relevance'])}
    sorted_results = sorted(json_dict.items(), key=lambda x: x[1]['relevance'], reverse=True)
    ordered_results = [result[1] for result in sorted_results if result[1]['relevance'] != 0]
    json_dict = {"results": ordered_results}
    # print(json_dict)
    return jsonify(json_dict)
