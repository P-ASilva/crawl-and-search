# Document Embedding Query System

## Dataset Description

The dataset used in this project consists of a collection of news articles derived from a variety of media sources, primarily focused on world events and influential public figures. Each document in the dataset includes textual content along with metadata such as the publication date and source. The dataset was selected to test how well pre-trained word embeddings, fine-tuned through neural network transformations, can represent content in a query retrieval system. This allows for accurate document querying based on semantic relevance, making it suitable for natural language processing (NLP) tasks involving text similarity and information retrieval.

## Embedding Generation Process

To generate embeddings, we start with pre-trained GloVe embeddings as a base representation. These embeddings are first transformed by a multi-layer perceptron (MLP) model, designed with an input layer of 50 units (matching the dimensionality of the GloVe vectors), a hidden layer of 128 units with ReLU activation, and an output layer producing embeddings that capture semantic transformations. This intermediate output is then further processed by a denoising autoencoder, with an encoder layer that reduces the dimensionality to 32 units, followed by a ReLU-activated hidden layer. The autoencoder effectively fine-tunes the embeddings to emphasize dataset-specific features while maintaining the general semantic information. The final output of the autoencoder serves as the enhanced, task-specific document embedding. 

<!-- The figure below shows the architecture of the embedding generation process: -->

<!-- ![Neural Network Architecture](path/to/your/image.png) -->

## Training Process

The training process for the denoising autoencoder aims to reconstruct the embeddings as closely as possible to the original MLP-transformed embeddings, with the goal of capturing both general and domain-specific semantics. We utilize Mean Squared Error (MSE) as the loss function to measure the difference between the original embeddings and their reconstructed counterparts. This loss function is suitable for our problem, as it encourages the model to minimize the reconstruction error, effectively preserving essential features while discarding noise. By minimizing MSE, we create embeddings that capture meaningful patterns specific to the dataset, enhancing relevance in document retrieval. The loss function is defined as follows:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{x}_i - x_i)^2
$$

where \( \hat{x}_i \) represents the reconstructed embedding and \( x_i \) is the original MLP-transformed embedding.

--- 