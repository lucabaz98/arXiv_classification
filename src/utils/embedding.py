
import pickle
import numpy as np
from keras.layers import Embedding
from keras.initializers import Constant

# Reading the GloVe words embedding (pre-trained model)
with open('../../data/external/glove-embedding.pkl', 'rb') as file:
    glove_embedding = pickle.load(file)


# Function which creates the embedding matrix (using GloVe embedding pre-trained model)
def buildEmbeddingMatrix(embedding_dim, vocabulary):

    vocabulary_size = len(vocabulary)

    embedding_matrix = np.zeros((vocabulary_size + 2, embedding_dim))

    for i, word in enumerate(vocabulary):

        word_embedding_repr = glove_embedding.get(word)

        if word_embedding_repr is not None:
            embedding_matrix[i] = word_embedding_repr
        else:
            embedding_matrix[i] = np.zeros(embedding_dim)

    return embedding_matrix


# Function which creates the embedding matrix (using Word2Vec built vocabulary)
def buildingEmbeddingMatrixWord2Vec(embedding_dim, vocabulary, vocabulary_embedding):

    vocabulary_size = len(vocabulary)

    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))

    for i, word in enumerate(vocabulary):

        embedding_vector = vocabulary_embedding[word]
        embedding_matrix[i] = embedding_vector

    return embedding_matrix


# Function which creates the embedding layer for the neural network
def createEmbeddingLayer(embedding_matrix, regularizer):

    return Embedding(

        embedding_matrix.shape[0], embedding_matrix.shape[1],
        embeddings_initializer = Constant(embedding_matrix), embeddings_regularizer = regularizer,
        trainable = False
    )