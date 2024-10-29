import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def get_links(url):
    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    absolute_links = [urljoin(url, link['href']) for link in links]
    return absolute_links

def crawl_web(start_url, max_urls=10):
    urls = []
    visited = set()

    # Start crawling from the initial URL
    urls_to_visit = [start_url]
    count = 0
    while count < max_urls:
        current_url = urls_to_visit.pop(0)
        if current_url in visited:
            continue
        print(f'Crawling: {current_url}')
        count += 1
        visited.add(current_url)
        new_urls = get_links(current_url)
        #print("Got links!")
        
        for url in new_urls:
            #print(url)
            if url.endswith('/'):
                url = url[:-1]
                
            if (url not in visited) and\
                ('#' not in url) and\
                (not url.startswith('mailto:')):
                
                urls_to_visit.append(url)
        
        time.sleep(0.1)  # Be polite and avoid overloading the server

    return visited

# Example usage
start_url = 'http://www.nasa.gov'
visited = crawl_web(start_url, max_urls=10)
print(visited)
