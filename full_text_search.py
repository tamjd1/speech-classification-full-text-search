from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import VERB
import numpy as np
import nltk
import os
import sys
f = open(os.devnull, 'w')
sys.stderr = f

nltk.download('wordnet')

with open("data/corpus.txt", encoding="utf-8") as f:
    CORPUS = [line.strip() for line in f.readlines() if line.startswith("\"")]

analyzer = TfidfVectorizer().build_analyzer()
lemmatizer = WordNetLemmatizer()


def lemmatize(document):
    return (lemmatizer.lemmatize(word, pos=VERB) for word in analyzer(document) if word not in ENGLISH_STOP_WORDS)


def search(search_terms, corpus=CORPUS):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=lemmatize)
    features = vectorizer.fit_transform([search_terms] + corpus)
    scores = (features[0, :] * features[1:, :].T).A[0]

    found_documents = []

    match_count = 0
    for score in scores:
        if score:
            match_count += 1

    if match_count:
        sorted_scores = np.argsort(scores)
        for i in range(match_count):
            match = sorted_scores[-1-i]
            answer = corpus[match]
            found_documents.append(answer)

    return found_documents
