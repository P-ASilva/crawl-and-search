import os
from bs4 import BeautifulSoup
import pandas as pd
import requests


base_api_endpoint = "https://www.cnnbrasil.com.br/economia/macroeconomia/pagina/"
# read cnn.csv if it exists
if os.path.exists('cnn.csv'):
    df = pd.read_csv('cnn.csv')
else:
    df = pd.DataFrame(columns=['title', 'subtitle', 'content'])
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
            # check if title already exists in the dataframe
            if title in df['title'].values:
                continue
            # find subtitle with tag <p> in header with class 'single-header__excerpt'
            subtitle = header.find('p', class_ = 'single-header__excerpt').text
            # extract all with tag <p> in div with class 'single-content'
            content = article.find('div', class_ = 'single-content').find_all('p')
            # join all <p> in a single string removing the tag
            content = ' '.join([c.text for c in content])
            # add a new row to the df with the extracted data
            df.loc[c] = {'title': title, 'subtitle': subtitle, 'content': content, 'link':link}
            c += 1
    # save df to csv
    df.to_csv('cnn.csv')