from flask import Flask
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from bs4 import BeautifulSoup
import requests
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/request")
def request_test():
    base_api_endpoint = "https://www.cnnbrasil.com.br/economia/macroeconomia/pagina/"
    # create a df with columns 'title', 'subtitle', 'content'
    df = pd.DataFrame(columns = ['title', 'subtitle', 'content'])
    # loop
    c = 0
    for pag in range(1, 100):
        api_endpoint = f"{base_api_endpoint}{pag}/"
        response = requests.get(api_endpoint)

        if response.status_code == 200:
            # Process the response data
            html_data = response.text
            soup = BeautifulSoup(html_data, 'html.parser')
            fine_soup = soup.find_all('li', class_ = 'home__list__item')
            for obj in fine_soup:
                # extract link
                link = obj.find('a')['href']
                follow_up = requests.get(link)
                follow_soup = BeautifulSoup(follow_up.text, 'html.parser')
                # isolate tag <article>
                article = follow_soup.find('article')
                # extract the text from the header class 'single-header'
                header = article.find('header', class_ = 'single-header')
                # find title with tag <h1> in header
                title = header.find('h1').text
                print(title)
                # find subtitle with tag <p> in header with class 'single-header__excerpt'
                subtitle = header.find('p', class_ = 'single-header__excerpt').text
                # extract all with tag <p> in div with class 'single-content'
                content = article.find('div', class_ = 'single-content').find_all('p')
                # join all <p> in a single string removing the tag
                content = ' '.join([c.text for c in content])
                # add a new row to the df with the extracted data
                df.loc[c] = {'title': title, 'subtitle': subtitle, 'content': content}
                c += 1
        # save df to csv
        df.to_csv('cnn.csv')
    
    return f"{fine_soup}"

# make a route named query, that takes as a now url param a string named q 

@app.route("/query")
def query():
    q = request.args.get('query')

    # df = pd.read_csv("cnn.csv")
    # vectorizer = TfidfVectorizer()
    # df['Plot'] = df['content'].apply(lambda x: x.upper())

    # X = vectorizer.fit_transform(df['Plot'])

    # Q = vectorizer.transform(['Zombie apocalypse'])
    # R = X @ Q.T
    # R = R.toarray().flatten()
    # idx = R.argsort()[-3]
    # print(df.iloc[idx]['title'])

    return f"Query: {q}"