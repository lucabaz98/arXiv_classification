
import nltk
import random
import numpy as np
import pickle
from nltk.corpus import wordnet

# Reading the GloVe words embedding (pre-trained model)
with open('../glove-embedding.pkl', 'rb') as file:
    glove_embedding = pickle.load(file)


# Function which gets the n most similar words for each token given in input, based on Glove Embedding
def getSimilarWords(tokenized_text, n):

    similar_words_total = []

    # For each word/token of the tokenized text given in input
    for word in tokenized_text:

        try:

            # Get the word embedding
            word_embedding = glove_embedding[word]

            # Compute distances between the word and the other words in the Glove Embedding
            distances = np.dot(list(glove_embedding.values()), word_embedding)

            # Get the n most similar words 
            most_similar_indices = np.argsort(-distances)[:n]
            similar_words = [ list(glove_embedding.keys())[i] for i in most_similar_indices ]

            similar_words_total = similar_words_total + similar_words
        
        # Excpetion if there are not similar words
        except KeyError:

            similar_words_total = similar_words_total + []

    return similar_words_total


# Function which inserts n words within the text given in input based on similarity 
def randomInsert(text, n):

    index = 0

    # Text tokenization
    tokenized_text = nltk.word_tokenize(text)
    augmented_tokenized_text = tokenized_text.copy()

    if(len(tokenized_text) > 5):

        # Random sample from initial text
        tokenized_text_sample = random.sample(tokenized_text, k = 5)

    else:
        tokenized_text_sample = tokenized_text

    # Get similar words
    similar_words = getSimilarWords(tokenized_text_sample, 5)

    if(len(similar_words) > n):
        similar_words = random.sample(similar_words, k = n)
    else:
        n = len(similar_words)

    while(index < n):
        
        # Random position insert
        new_word_index = random.randint(0, len(augmented_tokenized_text) - 1)
        augmented_tokenized_text.insert(new_word_index, similar_words[index])

        index = index + 1

    return " ".join(augmented_tokenized_text)
