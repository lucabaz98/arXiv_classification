
import tensorflow
from keras.layers.preprocessing.text_vectorization import TextVectorization
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# Function which creates the text vectorizer object with Keras
def createTextVectorizer(vocabulary_size, words_per_sentence, train_set):

    # TextVectorization Keras object
    text_vectorizer = TextVectorization(
        max_tokens = vocabulary_size, standardize = None, split = "whitespace", ngrams = None,
        output_mode = "int", output_sequence_length = words_per_sentence, vocabulary = None
    )

    # Create the vocabulary fitted with the train data
    text_vectorizer.adapt(train_set)

    return text_vectorizer


# Function which creates the text vectorizer from Word2Vec model
def createTextVectorizerWord2Vec(train_set, vocabulary_size, embedding_dim):

    data = [ row.split() for row in train_set.to_numpy() ]

    word2vec_model = Word2Vec(
        sentences = data, vector_size = embedding_dim, min_count = 3,
        workers = 4, max_final_vocab = vocabulary_size, sorted_vocab = 1
    )

    text_vectorizer = Tokenizer()
    text_vectorizer.word_index = word2vec_model.wv.key_to_index

    return { 'text_vectorizer': text_vectorizer, 'vocabulary_embedding': word2vec_model.wv }


# Function which performs the vectorization on the textual data
def textVectorization(data, text_vectorizer):
    return text_vectorizer(data)


# Functino which performst the vectorization on textual data with Word2Vec
def textVectorizationWord2Vec(data, text_vectorizer, words_per_sentence):
    return tensorflow.convert_to_tensor(
        pad_sequences(text_vectorizer.texts_to_sequences(data), maxlen = words_per_sentence, padding = 'post')
    )