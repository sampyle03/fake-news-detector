import os
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from nltk import ne_chunk as nltk_ne_chunk
from nltk import pos_tag as nltk_pos_tag
from nltk import ngrams as nltk_ngrams
from nltk.tree import Tree as nltk_tree
from nltk import FreqDist


def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """

    tokens = word_tokenize(text) # Tokenizes text
    ner_chunks = nltk_ne_chunk(nltk_pos_tag(tokens)) # Gets the named entities of the tokens
    tokens = nltk_pos_tag(tokens) # Gets the parts of speech tags of the tokens

    tags = []
    for chunk in ner_chunks:
        if isinstance(chunk, nltk_tree): # Check if chunk is a named entity
            entity_type = chunk.label()
            entity_words = [word for (word, pos) in chunk]
            entity_phrase = ' '.join(entity_words)  # Merge into "Annies List"
            tags.append((entity_phrase, entity_type)) # Set word's tag to its entity type
        else: 
            word, pos = chunk
            tags.append((word, pos)) # Set word's tag to its POS tag

    if tokens and tokens[0][0].lower() == "says":
        tokens = tokens[1:]
        tags = tags[1:]
    
    excluded_words = {'', ' '} # Exclude empty strings and spaces
    for t in tags:
        # remove punctation marks from start and end of token
        token = re.sub(r"^[^a-zA-Z0-9]+", "", t[0])
        token = re.sub(r"[^a-zA-Z0-9]+$", "", token).lower()
        #remove spaces before words
        token = re.sub(r"^\s+", "", token)
        if token in excluded_words:
            continue
        #if t[0] not in stopwords and is not a fully built up of punctuation marks
        if token not in set(stopwords.words('english')) and re.match(r'^[^\w\s]+$', token) is None:
            yield (token.lower(), t[1])



def detect_ngrams(tokens): # detects ngrams in tokens and returns which those most commonly found
    # ngrams = bigrams + trigrams + quadgrams
    ngrams = list(nltk_ngrams(tokens, 2)) + list(nltk_ngrams(tokens, 3)) + list(nltk_ngrams(tokens, 4))
    ngram_count = FreqDist(ngrams)
    for token, count in ngram_count.items():
        if len(token) == 2 and count > 14: # bigram considered "commonly found" if it appears more than 14 times
            yield token
        elif len(token) == 3 and count > 6:
            yield token
        elif len(token) == 4 and count > 4:
            yield token

def tokenize_ngrams(common_ngrams, cleaned_tokens):
    for i in range(len(cleaned_tokens)-1):
        bigram = (cleaned_tokens[i], cleaned_tokens[i+1])
        try:
            trigram = (cleaned_tokens[i], cleaned_tokens[i+1], cleaned_tokens[i+2])
            try:
                quadgram = (cleaned_tokens[i], cleaned_tokens[i+1], cleaned_tokens[i+2], cleaned_tokens[i+3])
                if quadgram in common_ngrams: # checks if four words from tokens being checked are one of commonQuadgrams
                    yield ((quadgram[0][0] + " " + quadgram[1][0] + " " + quadgram[2][0] + " " + quadgram[3][0]), (quadgram[0][1], quadgram[1][1], quadgram[2][1], quadgram[3][1]))
            except IndexError:
                pass
            if trigram in common_ngrams: # checks if three words from tokens being checked are one of commonTrigrams
                yield ((trigram[0][0] + " " + trigram[1][0] + " " + trigram[2][0]), (trigram[0][1], trigram[1][1], trigram[2][1]))
        except IndexError:
            pass
        if bigram in common_ngrams: # checks if two words from tokens being checked are one of commonBigrams
            yield ((bigram[0][0] + " " + bigram[1][0]), (bigram[0][1], bigram[1][1]))

def remove_ngram_tokens(tokens, ngram_tokens):
    """
    Function: Removes the singular tokens found in ngrams from the list of tokens
    Parameters: tokens - the list of tokens to be cleaned
    Returns: cleaned tokens
    """
    for ngram_to_check in ngram_tokens:
        ngram_to_check = ngram_to_check[0].split(" ")
        ngram_start_index = -1
        next_index = 0
        matches = 0
        for i in range(len(ngram_to_check)):
            for j in range(next_index, len(tokens) - len(ngram_to_check)):
                if tokens[j][0] == ngram_to_check[i]: # if one token from ngram is found in tokens
                    if ngram_start_index == -1:
                        ngram_start_index = j
                    matches += 1
                    next_index = j + 1
                    break # move onto next word in ngram
                else:
                    matches = 0
                    ngram_start_index = -1
            if matches == len(ngram_to_check): # if all tokens from ngram are found in tokens
                # delete each token found in ngram from tokens
                for i in range(ngram_start_index, ngram_start_index + len(ngram_to_check)):
                    tokens[i] = None
                break
        tokens = [token for token in tokens if token is not None]
    return tokens

def knn_plot_results(grid_search, x_label, y_label, title):
    """
    visualises hyperparameter tuning results (new function)
    """
    results = grid_search.cv_results_
    k_values = grid_search.param_grid['n_neighbors']
    avg_scores = results['mean_test_score']

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_scores, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def knnc_train_model():
    """
    Function: Trains and saves the knn model with optimal k.
    Parameters: None
    Returns: None
    """
    # load training data
    tf_idf_statements = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements.pkl")) # csr_matrix - (statement_no, word_no)   tf-idf value
    truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['label'] # pandas series

    # hyperparameter tuning
    # test k=5 to 320 in intervals of 8
    param_grid = {'n_neighbors': list(range(1, 521, 20))}
    knn = KNeighborsClassifier(metric='cosine')
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(tf_idf_statements, truth_ratings)

    # visualise results
    knn_plot_results(grid_search, "k", "accuracy", "KNN Classification Accuracy")

    # save best model
    best_knn = grid_search.best_estimator_
    pd.to_pickle(best_knn, os.path.join(current_dir, "../data/pickle/best_knnc_model.pkl"))

class RoundedKNeigborsRegressor(KNeighborsRegressor):
    def predict(self, X):
        y_pred = super().predict(X)
        return np.round(y_pred).astype(float)

def knnr_train_model():
    """
    Function: Trains and saves the knn model with optimal k.
    Parameters: None
    Returns: None
    """
    # load training data
    tf_idf_statements = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements.pkl"))
    truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['ordinal_label']

    # hyperparameter tuning
    # test k=1 to 150 in intervals of 5
    param_grid = {'n_neighbors': list(range(1, 151, 5))}
    knn = RoundedKNeigborsRegressor(metric='cosine')
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(tf_idf_statements, truth_ratings)
    
    # visualise results
    knn_plot_results(grid_search, "k", "neg_mean_squared_error", "KNN Regression Mean Squared Error")

    # save best model
    best_knn = grid_search.best_estimator_# visualise results
    pd.to_pickle(best_knn, os.path.join(current_dir, "../data/pickle/best_knnr_model.pkl"))

def classify_statement(data_path, statement):
    """
    Function: Classifies a statement as fake or real
    Parameters: None
    Returns: None
    """

    #tf_idf_statements = pd.read_pickle(os.path.join(current_dir, data_path))
    #truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['label']
    common_ngrams = pd.read_pickle(os.path.join(current_dir, "../data/pickle/common_ngrams.pkl"))
    words_in_corpus = pd.read_pickle(os.path.join(current_dir, "../data/pickle/words_in_corpus.pkl"))
    term_idfs = pd.read_pickle(os.path.join(current_dir, "../data/pickle/term_idfs.pkl"))

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
    
    for word in unique_words_in_statement:
        tf = statement_vector[words_in_corpus[word]] / len(statement) # get each terms tf value
        idf = term_idfs[words_in_corpus[word]]
        statement_vector[words_in_corpus[word]] = tf * idf

    best_knnc = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnc_model.pkl"))
    best_knnr = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnr_model.pkl"))

    x_new = csr_matrix(statement_vector).reshape(1, -1)
    knnc_classification = best_knnc.predict(x_new)[0]
    knnr_classification = best_knnr.predict(x_new)[0]
    print(f"knnc says statement is classified as: {knnc_classification}")
    print(f"knnr says statement is classified as: {knnr_classification}")

# Main
current_dir = os.path.dirname(__file__)
#knnc_train_model() # run once then comment out
#knnr_train_model() # run once then comment out
while True:
    statement = input("Enter a statement to classify: ")
    classify_statement("../data/pickle/tf_idf_statements.pkl", statement)