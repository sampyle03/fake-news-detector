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


def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """
    cleaned_text = re.sub(r"[^\w\s']+", " ", text) # https://stackoverflow.com/questions/31191986/br-tag-screwing-up-my-data-from-scraping-using-beautiful-soup-and-python
    tokens = word_tokenize(cleaned_text) # tokenizes text
    lemmatizer = WordNetLemmatizer()
    #ps = PorterStemmer()
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
    trains and saves the knn model with optimal k.
    run this once before classifying statements.
    """
    # load training data
    tf_idf_statements = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements.pkl"))
    truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['label']

    # hyperparameter tuning
    # test k=1 to 130 in intervals of 5
    param_grid = {'n_neighbors': list(range(5, 321, 8))} 
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
    trains knn regression model and evaluates accuracy
    """
    # load training data
    tf_idf_statements = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements.pkl"))
    truth_ratings = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_data.pkl"))['ordinal_label']

    # hyperparameter tuning
    # test k=1 to 130 in intervals of 5
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
knnc_train_model()
knnr_train_model()
while True:
    statement = input("Enter a statement to classify: ")
    classify_statement("../data/pickle/tf_idf_statements.pkl", statement)