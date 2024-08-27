import os
import re
from flask import Flask
from bs4 import BeautifulSoup
from datetime import datetime
import requests

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/request")
def request_test():
    api_endpoint = "https://www.cnnbrasil.com.br/economia/macroeconomia/"
    response = requests.get(api_endpoint)

    if response.status_code == 200:
        # Process the response data
        html_data = response.text
        soup = BeautifulSoup(html_data, 'html.parser')

        fine_soup = soup.find_all('li', class_ = 'home__list__item')
        # print(fine_soup)
        for li in fine_soup:
            print(li.find('a').get('href'))
            datetime_str = li.find('span').get_text()
            print(datetime_str)
            # filter out the links whose last date are past the latest analysed date
            latest_date = os.getenv('LATEST_SCRAPE_DATE')
            # YY/MM/DD HH:MM
            print("latest_date")
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d %H:%M")
            # use re lib to filter date from datetime
            print("current_date")
            date_regex = r'\d{2}[\/\-]\d{2}[\/\-]\d{4}'
            date = re.findall(date_regex, datetime_str)[0]
            print(date)
            time_regex = r'\d{2}:\d{2}'
            time = re.findall(time_regex, datetime_str)[0]
            print(time)
            print('time?')
            dateandtime = date + " " + time

            date = datetime.strptime(dateandtime, "%d/%m/%Y %H:%M")

            if date >= latest_date:
                # use re lib to filter time from datetime
                print("Date is newer than latest date")
            else:
                print("Date is older than latest date")

            # pegar info do link e adicionar como uma linha do csv

    else:
        print(f"Error: {response.status_code}")
        return "HELP"

    return f"{fine_soup}"
