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

# download the wordnet dataset provided by nltk
# to be used for lemmatizing and tokenizing the corpus
nltk.download('wordnet')

# open the corpus text file containing documents
# and parse each document so they can be used
# to calculate TF/IDF scores
with open("data/corpus.txt", encoding="utf-8") as f:
    CORPUS = [line.strip() for line in f.readlines() if line.startswith("\"")]

analyzer = TfidfVectorizer().build_analyzer()
lemmatizer = WordNetLemmatizer()


def lemmatize(document):
    """
    This function generates an array of lemmatized words contained in the document
    If the words exist in the ENGLISH_STOP_WORDS dictionary provided by sklearn
        they are ignored and not added to the lemmatized list
    The lemmatization occurs on the verbs, so each verb is converted to its root word
        in order to get all the relevant documents for a given verb
    :param document: document string to lemmatize
    :return: list of lemmatized words
    """
    return (lemmatizer.lemmatize(word, pos=VERB) for word in analyzer(document) if word not in ENGLISH_STOP_WORDS)


def search(search_terms, corpus=CORPUS):
    """
    This function generates an object containing TF/IDF scores on the corpus
        then uses the scores to return the most relevant documents in the corpus
        based on the search term(s) provided
    :param search_terms: string containing one or more words to search
    :param corpus: list of documents in which to search for search terms
                    (default: documents parsed from ./data/corpus.txt)
    :return: list of documents, sorted by relevancy
    """
    # instantiate TF/IDF vectorizer, provided by sklearn
    vectorizer = TfidfVectorizer(min_df=1, analyzer=lemmatize)
    # get the features and TF/IDF scores from the corpus and the search terms
    features = vectorizer.fit_transform([search_terms] + corpus)
    scores = (features[0, :] * features[1:, :].T).A[0]

    found_documents = []

    # determine the number of documents for which the score was greater than zero
    #   that is, there was a match
    match_count = 0
    for score in scores:
        if score:
            match_count += 1

    # sort the scores by highest to lowest and get the document index
    # so the documents can be retrieved from the corpus
    sorted_scores = np.argsort(scores)
    for i in range(match_count):
        # retrieve each document that matched the search term(s)
        # by most relevant to least
        match = sorted_scores[-1-i]
        document = corpus[match]
        found_documents.append(document)

    # return documents list sorted by relevancy
    return found_documents
