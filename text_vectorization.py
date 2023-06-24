
from keras.layers.preprocessing.text_vectorization import TextVectorization

# Function which creates the text vectorizer object
def createTextVectorizer(vocabulary_size, words_per_sentence, train_set):

    # TextVectorization Keras object
    text_vectorizer = TextVectorization(
        max_tokens = vocabulary_size, standardize = None, split = "whitespace", ngrams = None,
        output_mode = "int", output_sequence_length = words_per_sentence, vocabulary = None
    )

    # Create the vocabulary fitted with the train data
    text_vectorizer.adapt(train_set)

    return text_vectorizer

# Function which performs the vectorization on the textual data
def textVectorization(data, text_vectorizer):
    return text_vectorizer(data)