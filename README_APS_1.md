### CNN Web Scraping and TF-IDF Search Process

#### Introduction
This project aims to provide a functional API template to search on a document database using their contents to provide a relevancy score for each document. The `app.py` script utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to search through the data scraped using the `webscraper_cnn.py` script, which is currently set to scrape CNN's website, where it'll be extracting new's titles, subtitles and content for the purpose of generating a unique database for document classification.
*The application is built using Flask and the number of news can be changed within the scraper's code.*

#### Project Structure

- **`app.py`**: The main Python script containing a minimalistic Flask application
- **`webscraper_cnn`**: contains a simple web scraper made with the CNN website in mind, specifically in the macroeconomy section.

#### Main Libraries Used

- **Flask**: A micro web framework used for building the web server.
- **pandas**: A data manipulation library, used here for handling and processing the CSV file.
- **scikit-learn (sklearn)**: A machine learning library that provides the `TfidfVectorizer` for converting text into numerical representations for analysis.

#### Web Scraping Process

1. **Sending HTTP Request**: The `requests` library is used to send a GET request to the CNN website.
2. **Parsing HTML**: The HTML content of the webpage is parsed using `BeautifulSoup`, which helps in extracting the relevant text content, such as the title, subtitle, and article body.
3. **Data Storage**: The extracted content is stored in a structured format (CSV file). This allows for easy retrieval and processing in later stages.

#### TF-IDF Search Process

1. **Loading Data**: The scraped data stored in `cnn.csv` is loaded into a pandas DataFrame.
2. **Text Preprocessing**: All text content in the DataFrame is converted to lowercase to ensure uniformity.
3. **Vectorization**:
   - The `TfidfVectorizer` from `scikit-learn` is used to convert the text data into a matrix of TF-IDF features.
   - The TF-IDF approach quantifies the importance of each word in a document relative to a corpus of documents.
4. **Query Processing**:
   - The search query entered by the user is also vectorized using the same TF-IDF model.
   - This allows for comparison between the query and the content of each document in the corpus.
5. **Similarity Calculation**:
   - The similarity between the vectorized query and each document in the dataset is calculated as follows:
   
   ```python
   X = vectorizer.fit_transform(df['content'])
   q = q.lower()
   Q = vectorizer.transform([q])
   R = X @ Q.T
   R = R.toarray().flatten()
   ```
   - `X = vectorizer.fit_transform(df['content'])`: In order to use the TF-IDF search process, we must first convert the dataset that we will search on to an abstract form, which functions as both a vocabulary and a frequency indicator for the words within each document in the full 'content' dataframe. (also described in item 3)
   - `Q = vectorizer.transform([q])`: First in the search process, we transform our query into a sparse matrix, shaped in a way to match shapes with the X matrix, representing the distribution of words in the query, based in the words present in the complete 'content' dataset. (also described in item 4)
   - `R = X @ Q.T`: This line of code performs a matrix multiplication between the transformed data X and the transformed query Q, resulting in a relevance number that reflects the similarity between the query and the documents present in X.

   - Documents are ranked based on their similarity score to the query.
6. **Result Filtering and Display**:
   - The top 10 most relevant documents are selected based on their similarity scores.
   - The results are then formatted and returned as a JSON object, containing the title, subtitle, content snippet and relevance score for each document. If this application is deployed, proper credit and citations will be added with hyperlinks.

#### Understanding TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that is widely used in information retrieval and text mining to assess the importance of a word in a document relative to a collection of documents (also known as a corpus).

1. **Term Frequency (TF)**:
- **Definition**: TF measures how frequently a term appears in a document.
- **Abstraction**:
  
  ```math
  \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
  ```

- **Purpose**: A higher term frequency indicates that the word is significant within that document.

2. **Inverse Document Frequency (IDF)**:
- **Definition**: IDF measures how important a term is across the entire corpus.
- **Abstraction**:
  
  ```math
  \text{IDF}(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right)
  ```

- **Purpose**: A high IDF value indicates that the term is rare in the corpus, and therefore, more significant.

3. **TF-IDF Score**:
- **Definition**: The TF-IDF score is the product of TF and IDF.
- **Abstraction**:
  
  ```math
  \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
  ```

- **Purpose**: This score helps highlight words that are both frequent in a document and rare across the corpus, making them more relevant for identifying the document's unique content.

#### API Endpoints

- **`/`**: A simple endpoint that returns a "Hello, World!" message to ensure the server is running.
- **`/query`**: The primary endpoint that accepts a query string as a parameter and returns the search results. 

  **Example Usage**:
  ```bash
  http://localhost:4040/query?query=your_search_term
  ```

#### Installation and Setup

1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Web Scraper and wait for it to finish scanning the site**
   ```bash
   python webscraper_cnn.py
   ```
   #### change the length of the for loop inside the file depending on the time and space available to store data.
4. **Run the Application**:
   ```bash
   python app.py
   ```

   The Flask server will start and run locally on port 4040.

4. **Access the API**: 
   You can send search queries to the `/query` endpoint to retrieve the relevant articles based on the content scraped from CNN.

#### Why Search the Most Recent News on CNN?
By isolating recent news, we can perform searches focused on the journalistic relevance of people, institutions, and areas of interest. The utility of this project, therefore, lies in identifying popular topics and detecting changes in media coverage across various subjects, while being focused in an adaptable data sample.

#### Test/Control Cases:
In order to test the search process, a few control cases where explored and may allow for inferences based on the data sample obtained in the scraping process.

1. **http://10.103.0.28:4040/query?query=banco%20central**: 
    Test with 10 relevant results (Highly relevant). 
    When providing the query "Banco Central," we get more than 10 results in return, as it is a very popular in the data we collected. This can also be observed in the relevance score of the query.
2. **http://10.103.0.28:4040/query?query=cyrus%20de%20la%20rubia**:
    Test with fewer than 10 results (Less relevant).
    When providing the query "Cyrus de la Rubia," we get fewer than 10 news articles in return, which is expected since this name is not mentioned frequently in the current data sample (2024.2) and it's journalistic relevance could be linked to mentions from others.
3. **http://10.103.0.28:4040/query?query=desastre%20climatico%20muitos%20mortos**:
    Test with non-obvious results (Specific).
    When providing the query "climate disaster many dead," it is noticeable that the news article that appears does not directly refer to causes of a disaster, nor to related costs at the state level. The result refers to national changes in consumption habits, which could indicate that the journalistic relevance of the disaster itself is not very high in this iteration of the dataframe.

**Note:** Any and all inferences based on the data are subject to the variability of the data used and the restrictions/themes of the news network from which the data were sourced, these are examples of information that may or not be accurate to the truth of the matter.

**Disclaimer** ChatGpt was used to generate this readme template as well as to provide aid in formatting the text.

#### Conclusion
This application combines web scraping and text processing techniques to provide a searchable database of CNN articles. It leverages the power of TF-IDF for efficient and relevant search results, making it a robust tool for querying large text datasets and a good start for data collection on news articles.
