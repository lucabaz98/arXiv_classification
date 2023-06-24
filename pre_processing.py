
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function which does the pre-processing on textual data
def dataPreProcessing(arxiv_data):
    
    processed_data = arxiv_data.copy()

    for index, row in processed_data.iterrows():

        # Convert to lowercase
        row['text'] = row['text'].lower()

        # Remove special characters and symbols
        row['text'] = re.sub(r'[^a-zA-Z\s]', '', row['text'])

        # Text tokenization
        tokens = word_tokenize(row['text'])

        # Stopwords removal
        stop_words = set(stopwords.words('english'))
        tokens = [ word for word in tokens if word not in stop_words ]

        # Words lemmatization 
        lemmatizer = WordNetLemmatizer()
        tokens = [ lemmatizer.lemmatize(word) for word in tokens ]

        # Join tokens array in a single string
        tokens = ' '.join(tokens)

        # Processed text
        processed_data.loc[index,'text'] = tokens

    return processed_data