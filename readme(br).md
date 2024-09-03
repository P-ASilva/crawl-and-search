Here's the Portuguese (Brazil) version of your README:

---

### README: Processo de Web Scraping e Busca TF-IDF no Site da CNN

#### Introdução
Este documento fornece uma visão geral da funcionalidade implementada no script `app.py`, que é projetado para realizar uma busca nas 2700 notícias mais recentes no site da CNN, utilizando o algoritmo TF-IDF (Term Frequency-Inverse Document Frequency) para pesquisar nos dados extraídos usando o script `webscraper_cnn.py`. A aplicação foi construída usando Flask e o número de notícias pode ser alterado no código do scraper, mas para esta iteração decidi não fazer adaptações para economizar tempo.

#### Estrutura do Projeto

- **`app.py`**: O script principal em Python que contém uma aplicação Flask minimalista.
- **`webscraper_cnn`**: Contém um web scraper simples feito com o site da CNN em mente, especificamente na seção de macroeconomia.

#### Principais Bibliotecas Utilizadas

- **Flask**: Um micro framework web usado para construir o servidor web.
- **pandas**: Uma biblioteca de manipulação de dados, usada aqui para lidar e processar o arquivo CSV.
- **scikit-learn (sklearn)**: Uma biblioteca de aprendizado de máquina que fornece o `TfidfVectorizer` para converter texto em representações numéricas para análise.

#### Processo de Web Scraping

1. **Enviando Requisição HTTP**: A biblioteca `requests` é usada para enviar uma requisição GET ao site da CNN.
2. **Parsing do HTML**: O conteúdo HTML da página é analisado usando `BeautifulSoup`, que ajuda a extrair o conteúdo de texto relevante, como o título, subtítulo e corpo do artigo.
3. **Armazenamento de Dados**: O conteúdo extraído é armazenado em um formato estruturado (arquivo CSV). Isso permite fácil recuperação e processamento em etapas posteriores.

#### Processo de Busca TF-IDF

1. **Carregando os Dados**: Os dados extraídos e armazenados em `cnn.csv` são carregados em um DataFrame do pandas.
2. **Pré-processamento de Texto**: Todo o conteúdo de texto no DataFrame é convertido para minúsculas para garantir a uniformidade.
3. **Vetorização**:
   - O `TfidfVectorizer` da `scikit-learn` é usado para converter os dados de texto em uma matriz de características TF-IDF.
   - A abordagem TF-IDF quantifica a importância de cada palavra em um documento em relação a um corpus de documentos.
4. **Processamento da Consulta**:
   - A consulta de pesquisa inserida pelo usuário também é vetorizada usando o mesmo modelo TF-IDF.
   - Isso permite a comparação entre a consulta e o conteúdo de cada documento no corpus.
5. **Cálculo de Similaridade**:
   - A similaridade entre a consulta vetorizada e cada documento no dataset é calculada da seguinte forma:
   
    ```python
    X = vectorizer.fit_transform(df['content'])
    q = q.lower()
    Q = vectorizer.transform([q])
    R = X @ Q.T
    R = R.toarray().flatten()
   ```
   - `X = vectorizer.fit_transform(df['content'])`: Para usar o processo de busca TF-IDF, devemos primeiro converter o dataset que iremos buscar em uma forma abstrata, que funciona como um vocabulário e um indicador de frequência das palavras em cada documento no DataFrame completo 'content'. (também descrito no item 3)
   - `Q = vectorizer.transform([q])`: Primeiro no processo de busca, transformamos nossa query em uma matriz esparsa, modelada de forma ser compátivel em tamanho com a matriz X, representando a distribuição de palavras na query, baseada nas palavras presentes no dataset completo 'content'. (também descrito no item 4)
   - `R = X @ Q.T`: Esta linha de código realiza uma multiplicação de matrizes entre os dados transformados X e a query transformada Q, resultando em um número de relevância que reflete a similaridade entre a query e os documentos representados em X.

   - Os documentos são classificados com base em sua pontuação de similaridade com a consulta.

6. **Filtragem e Exibição dos Resultados**:
   - Os 10 documentos mais relevantes são selecionados com base em suas pontuações de similaridade.
   - Os resultados são então formatados e retornados como um objeto JSON, contendo o título, subtítulo, trecho do conteúdo e a pontuação de relevância de cada documento. Se esta aplicação for implantada, créditos e citações adequados serão adicionados com hiperlinks.

#### O que é TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** é uma medida estatística usada para avaliar a importância de uma palavra em um documento em relação a um conjunto de documentos (ou corpus). É comumente usada em recuperação de informação, mineração de texto e motores de busca.

1. **Frequência de Termos (TF)**:

**TF (Term Frequency)** mede a frequência com que um termo aparece em um documento específico. Quanto maior a frequência do termo dentro do documento, maior será o valor de TF. Para levar em consideração o comprimento do documento, o TF é normalizado.

2. **Frequência Inversa de Documentos (IDF)**:

**IDF (Inverse Document Frequency)** avalia a importância de um termo em todo o corpus. Termos que aparecem em muitos documentos recebem uma pontuação IDF menor, indicando que são comuns e menos informativos. Por outro lado, termos que aparecem em poucos documentos têm uma pontuação IDF mais alta, sugerindo que são mais únicos e significativos.

3. **Pontuação TF-IDF**:

A pontuação TF-IDF é o produto de TF e IDF. Esta pontuação destaca termos que são frequentes em um documento específico, mas raros em todo o corpus, tornando-os mais relevantes para a identificação do conteúdo único do documento.

#### Endpoints da API

- **`/`**: Um endpoint simples que retorna uma mensagem "Hello, World!" para garantir que o servidor está em execução.
- **`/query`**: O endpoint principal que aceita uma string de consulta como parâmetro e retorna os resultados da busca.

  **Exemplo de Uso**:
  ```bash
  http://localhost:5000/query?query=termo_de_busca
  ```

  A resposta é um objeto JSON contendo os artigos mais relevantes juntamente com seus metadados associados.

#### Instalação e Configuração

1. **Clonar o Repositório**: 
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Instalar Dependências**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Executar o Web Scraper e esperar que ele termine de escanear o site**
   ```bash
   python webscraper_cnn.py
   ```
   #### Altere o comprimento do loop for dentro do arquivo dependendo do tempo e espaço disponível para armazenar os dados.
4. **Executar a Aplicação**:
   ```bash
   python app.py
   ```

   O servidor Flask começará a rodar localmente na porta 4040.

4. **Acessar a API**: 
   Você pode enviar consultas de busca ao endpoint `/query` para recuperar os artigos relevantes com base no conteúdo extraído da CNN.

#### Por que buscar as notícias mais recentes da CNN?
Ao isolarmos notícias recentes, podemos fazer buscas orientadas a relevância jornalistica de pessoas, instituições e temas dentro das notícias armazenadas no banco de dados. A utilidade deste projeto se encontra na identificação de tópicos populares e mudanças na cobertura de mídia de diversos temas, enquanto mantém uma base de dados adaptável, com sua aplicação real sendo dependente do escopo da análise, periodo de coleta dos dados e quantidade coletada.

#### Casos de teste/controle:
Para testar o processo de busca, alguns casos de controle foram criados que permitem algumas inferências baseadas nos dados obtidos. 
1. **http://10.103.0.28:4040/query?query=banco%20central**: 
    Teste com 10 resultados relevantes (Muito relevante). 
    Ao fornecer a query "Banco Central" temos o retorno de 10 resultados, visto que é um tópico bem popular nessa versão do banco de dados (09/2024) isso também pode ser observado na pontuação de relevância da query.
2. **http://10.103.0.28:4040/query?query=cyrus%20de%20la%20rubia**
    Teste com menos de 10 resultados (Pouco relevante).
    Ao fornecer a query "cyrus de la rubia" temos o retorno de menos de 10 notícias, o que é esperado visto que seu nome é não é mencionado com tanta frequência nesta coletânea de notícias.
3. **http://10.103.0.28:4040/query?query=desastre%20climatico%20muitos%20mortos**:
    Teste com resultado não-óbvio (específico).
    Ao fornecer a query "desastre climatico muitos mortos" percebe-se que a notícia que aparece não se refere diretamente ao ocorrido, nem a custos correlatos no nível estatal. O resultado se refere a mudanças nacionais em hábitos de consumo, mostrando como a relevância jornalistica desses termos não é muito alta neste banco de dados.

**OBS:** Toda e qualquer inferência com base nos dados está sugeita a variabilidade dos dados utilizados e restrições/temas da rede de notícias de onde foram retirados os dados.

#### Conclusão
Esta aplicação combina técnicas de web scraping e processamento de texto para fornecer um banco de dados pesquisável de artigos da CNN. Ela aproveita o poder do TF-IDF para resultados de busca eficientes e relevantes, tornando-se uma ferramenta robusta para consultas em grandes conjuntos de dados de texto e uma boa opção para coleta de dados de notícias recentes.