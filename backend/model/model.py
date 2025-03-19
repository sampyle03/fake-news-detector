import os
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt


def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """
    cleaned_text = re.sub(r"[^\w\s']+", " ", text) # https://stackoverflow.com/questions/31191986/br-tag-screwing-up-my-data-from-scraping-using-beautiful-soup-and-python
    tokens = word_tokenize(cleaned_text) # tokenizes text
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    for t in tokens:
        if t not in set(stopwords.words('english')): # ignores stopwords
            yield lemmatizer.lemmatize(t.lower())
            # yield (ps.stem(t.lower())) # can uncomment to use stemming

def tokenize_ngrams(common_ngrams, cleaned_tokens):
    for i in range(len(cleaned_tokens)-1):
        bigram = (cleaned_tokens[i], cleaned_tokens[i+1])
        try:
            trigram = (cleaned_tokens[i], cleaned_tokens[i+1], cleaned_tokens[i+2])
            try:
                quadgram = (cleaned_tokens[i], cleaned_tokens[i+1], cleaned_tokens[i+2], cleaned_tokens[i+3])
                if quadgram in common_ngrams: # checks if four words from tokens being checked are one of commonQuadgrams
                    yield quadgram[0] + " " + quadgram[1] + " " + quadgram[2] + " " + quadgram[3]
            except IndexError:
                pass
            if trigram in common_ngrams: # checks if three words from tokens being checked are one of commonTrigrams
                yield trigram[0] + " " + trigram[1] + " " + trigram[2]
        except IndexError:
            pass
        if bigram in common_ngrams: # checks if two words from tokens being checked are one of commonBigrams
            yield bigram[0] + " " + bigram[1]


def classify_statement(data_path, statement):
    """
    Function: Classifies a statement as fake or real
    Parameters: None
    Returns: None
    """

    tf_idf_statements = pd.read_pickle(os.path.join(current_dir, data_path))
    truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['label']
    common_ngrams = pd.read_pickle(os.path.join(current_dir, "../data/pickle/common_ngrams.pkl"))
    words_in_corpus = pd.read_pickle(os.path.join(current_dir, "../data/pickle/words_in_corpus.pkl"))

    statement = list(clean_text(statement))
    statement += list(tokenize_ngrams(common_ngrams, statement))
    statement_vector = [0 for _ in range(len(words_in_corpus))]

    unique_words_in_statement = set()
    for word in statement:
        try:
            statement_vector[words_in_corpus[word]] += 1
            if word not in unique_words_in_statement:
                unique_words_in_statement.add(word)
        except KeyError:
            pass
    
    term_idfs = pd.read_pickle(os.path.join(current_dir, "../data/pickle/term_idfs.pkl"))
    for word in unique_words_in_statement:
        tf = statement_vector[words_in_corpus[word]] / len(statement) # get each terms tf value
        idf = term_idfs[words_in_corpus[word]]
        statement_vector[words_in_corpus[word]] = tf * idf

    similarities = cosine_similarity(csr_matrix(statement_vector).reshape(1, -1), tf_idf_statements)[0]
    k = 10
    most_similar = [(i, similarities[i]) for i in range(0, k)]
    for i in range(len(similarities)):
        insert_at = None
        for j in range(len(most_similar)-1, -1, -1):
            if similarities[i] <= most_similar[j][1]:
                break
            else:
                insert_at = j
        if insert_at is not None:
            most_similar.insert(insert_at, (i, similarities[i]))
            most_similar.pop()
    
    knn_truth_ratings = {}
    for item in most_similar:
        if truth_ratings[item[0]] not in knn_truth_ratings:
            knn_truth_ratings[truth_ratings[item[0]]] = 1
        else:
            knn_truth_ratings[truth_ratings[item[0]]] += 1
    print("Truth Ratings")
    print(knn_truth_ratings)
    
# Main
current_dir = os.path.dirname(__file__)
while True:
    statement = input("Enter a statement to classify: ")
    classify_statement("../data/pickle/tf_idf_statements.pkl", statement)