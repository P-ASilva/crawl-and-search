# Document Embedding Query System

## Dataset

The dataset used for the embedding-based query system is the same used in the first version of this project, set to scrape 100 pages of news articles in order to increase sample size and generate better data clusters. Each document contain a title, content and origin data. This dataset has a variable and specific nature, depending on historic moments and current relevance of terms, which generates the need for pre-trained embeddings to fill in the gaps in semantic coverage. This allows for accurate document querying based on semantic relevance, weighted by the embeddings generated from the documents themselves, making it suitable for natural language processing (NLP) tasks involving text similarity and information retrieval.

## Embedding Generation Process

To generate embeddings, we start with pre-trained GloVe embeddings as a base representation. Them we use these embeddings to generated document specific embeddings, which are them reduced by an autoencoder layer that reduces the dimensionality to 32 units, followed by a ReLU-activated hidden layer. The autoencoder effectively fine-tunes the embeddings to emphasize dataset-specific features while maintaining the general semantic information. The final output of the autoencoder serves as the enhanced, task-specific document embedding. 

## Training Process

The training process for the denoising autoencoder aims to reconstruct the embeddings as closely as possible to the original documents embeddings, with the goal of capturing both general and domain-specific semantics. We utilize Mean Squared Error (MSE) as the loss function to measure the difference between the original embeddings and their reconstructed counterparts. This loss function encourages the model to minimize the reconstruction error, effectively preserving essential features while discarding noise. By minimizing MSE, we create embeddings that capture meaningful patterns specific to the dataset, enhancing relevance in document retrieval. The loss function is defined as follows:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{x}_i - x_i)^2
$$

where \( \hat{x}_i \) represents the reconstructed embedding and \( x_i \) is the original MLP-transformed embedding.