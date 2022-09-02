import numpy as np
import pandas as pd

# Librerias de Natural Language Toolkit para el trato de palabras
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

train = pd.read_csv("train.csv")

print("--- Vista inicial de los datos ---")
print(train.head())

print("--- Conteo de valores NA ---")
print(train.isnull().sum())

# Tokenizacion de un mensaje en palabras
def get_corpus(texts):
    corpus = []
    for text in texts:
        corpus.extend(word_tokenize(text))
    return corpus

# Conjunto de "stop words", i.e. palabras comunes como articulos y preposiciones.
# Estas palabras no aportan mucho al mensaje o nucleo de un texto. 
def get_stop_words():
    return set(stopwords.words('english'))
    
# Removimiento de stop words de un mensaje tokenizado por palabras
def remove_stop_words(text):
    result = ""
    stop_words = get_stop_words()
    for word in get_corpus([text]):
        if word not in stop_words:
            result += word + " "
    return result.strip()

# Conjunto de signos de puntuación.
# Estos no aportan al mensaje principal de un texto. 
def get_punctuation():
    return string.punctuation

# Removimiento de signos de puntuacion
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


