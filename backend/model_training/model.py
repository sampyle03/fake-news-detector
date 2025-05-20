#model.py
# this is model training and evaluation for k-NN and k-NN regression - BERT training and evaluation is in bert_classifier.py
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
from sklearn.metrics import accuracy_score, make_scorer, f1_score, classification_report
from nltk import ne_chunk as nltk_ne_chunk
from nltk import pos_tag as nltk_pos_tag
from nltk import ngrams as nltk_ngrams
from nltk.tree import Tree as nltk_tree
from nltk import FreqDist
from sklearn.model_selection import StratifiedKFold



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

        # lemmatize token
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token)

        #if t[0] not in stopwords and is not a fully built up of punctuation marks
        if token not in set(stopwords.words('english')) and re.match(r'^[^\w\s]+$', token) is None:
            yield (token.lower(), t[1])



def detect_ngrams(tokens): # detects ngrams in tokens and returns which those most commonly found
    """
    Function: detects ngrams in tokens and returns which those most commonly found
    Parameters: tokens - the list of tokens to detect ngrams in
    Yields: ngram (tuple) - one of the most commonly found ngrams
    """
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
    """
    Function: tokenizes the ngrams in the cleaned tokens
    Parameters: common_ngrams - the list of common ngrams to check for
                cleaned_tokens - the list of cleaned tokens to check for ngrams in
    Yields: ngram (tuple) - the ngram found in the cleaned tokens
    """
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
    Returns: cleaned_tokens (list) - the list of tokens with singular tokens found in ngrams removed and ngrams added
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
    Function: displays results of hyperparameter tuning for k-NN classifier/regressor (dependent on parameters)
    Parameters: grid_search - the grid search object
                x_label - the label for the x-axis
                y_label - the label for the y-axis
                title - the title of the plot
    Returns: None
    """
    results = grid_search.cv_results_ # get results of grid search
    k_values = grid_search.param_grid['n_neighbors'] # get k values
    avg_scores = results['mean_test_score'] # get mean test scores

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_scores, marker='o') 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def knnc_train_model():
    """
    Function: Trains the k-NN classifier, finding optimal k value and visualising hyperpaameter (k) training and saves it to a pickle file
    Parameters: None
    Returns: None
    """
    current_dir = os.path.dirname(__file__)

    # load TF-IDF matrix for train set
    X_train = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements_train.pkl"))

    # load ordinal labels and convert to 1 if label=4 or more, otherwisee 0
    y_ordinals = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_train.pkl"))["ordinal_label"]
    y_train = (y_ordinals >= 4).astype(int)

    # sample weights 
    sample_weights = y_train.map(lambda y: 3 if y == 1 else 1)

    # k-NN classifier and grid search
    knnc = KNeighborsClassifier(metric='cosine', weights='distance')
    param_grid = {'n_neighbors': list(range(1, 20, 1))}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(knnc, param_grid, cv=cv, scoring="f1_weighted")
    grid_search.fit(X_train, y_train)

    knn_plot_results(grid_search,x_label="k",y_label="mean F1 (weighted)",title="K-NN Classification F1 vs k")

    # get the best model
    best_knnc = grid_search.best_estimator_
    print("Best k-NN Classifier k value:", grid_search.best_params_['n_neighbors'])

    # save model
    pd.to_pickle(best_knnc, os.path.join(current_dir, "../data/pickle/best_knnc_model.pkl"))

class RoundedKNeighborsRegressor(KNeighborsRegressor): # inherits from KNeighborsRegressor
    def predict(self, X): # overrides predict method to round the predictions to the nearest integer
        y_pred = super().predict(X)
        return np.round(y_pred).astype(float)
    
def regressor_f1(y_true, y_pred): # custom scoring function for k-NN regressor (not used for final product)
    y_pred_bin = (np.round(y_pred) >= 4).astype(int)
    y_true_bin = (y_true >= 4).astype(int)
    return f1_score(y_true_bin, y_pred_bin)

def knnr_train_model():
    """
    Function: Trains the k-NN regressor, finding optimal k value and visualising hyperpaameter (k) training and saves it to a pickle file
    Parameters: None
    Returns: None
    """
    X_train = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tf_idf_statements_train.pkl"))

    # load ordinal labels (0–5)
    y_train = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_train.pkl"))['ordinal_label']

    # scorer = make_scorer(regressor_f1)
    param_grid = {'n_neighbors': list(range(1, 151, 5))}
    knn_reg = RoundedKNeighborsRegressor(metric='cosine')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(knn_reg, param_grid, cv=cv, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)

    knn_plot_results(grid_search, x_label="k", y_label="neg_mean_absolute_error", title="KNN Regression neg_mean_absolute_error vs k")

    best_knnr = grid_search.best_estimator_
    pd.to_pickle(best_knnr,os.path.join(current_dir,"../data/pickle/best_knnr_model.pkl"))

    print("Best k-NN regressor k value:", grid_search.best_params_['n_neighbors'])

def classify_statement(statement):
    """
    Function: Classifies a statement as fake or real using boththe k-NN classifier and the k-NN regressor.
    Parameters: statement (str) - the statement to classify
    Returns: None
    """

    current_dir = os.path.dirname(__file__)

    vectorizer = pd.read_pickle(os.path.join(current_dir, "../data/pickle/tfidf_vectorizer.pkl"))

    # transform statement into vector
    x_new = vectorizer.transform([statement])
    print("statement_vector shape:", x_new.shape)

    # load models
    best_knnc = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnc_model.pkl"))
    best_knnr = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnr_model.pkl"))

    # make predictions
    knnc_classification = best_knnc.predict(x_new)[0]
    reg_raw  = best_knnr.predict(x_new)[0]

    # round & threshold the regressor
    knnr_int  = int(np.round(reg_raw)) # 0–5 integer
    print(f"knnc says statement is classified as: {'true' if knnc_classification == 1 else 'false'}")
    print(f"knnr says statement is classified as: {'true' if knnr_int >= 4 else 'false'} - Confidence: {'low' if knnr_int == 3 else 'medium' if knnr_int == 2 else 'high' if (knnr_int == 5 or knnr_int == 0) else 'low/medium'}")

def evaluate(model, tfidf_path, ordinal_path, is_regressor=False):
    """
    Function: Evaluates the model on the test set and then prints the corresponding confusion matrix, classification report, accuracy, and F1 score.
    Parameters: model - the model to evaluate
                tfidf_path - the path to the TF-IDF matrix file
                ordinal_path - the path to the ordinal labels file
                is_regressor - whether the model is a regressor or not (defaulted to False)
    Returns: None
    """
    X = pd.read_pickle(tfidf_path)
    ordinals = pd.read_pickle(ordinal_path)['ordinal_label']
    y_true = (ordinals >= 4).astype(int)

    y_pred = model.predict(X)
    if is_regressor:
        y_pred = np.round(y_pred).astype(int)
        y_pred = (y_pred >= 4).astype(int)

    print("Confusion Matrix:")
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)) # https://www.geeksforgeeks.org/pandas-crosstab-function-in-python/
    print("Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["False", "True"]))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))





# Main - uncomment to run
current_dir = os.path.dirname(__file__)
# knnc_train_model() # run once then comment out
# knnr_train_model() # run once then comment out

# best_knnc = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnc_model.pkl"))
# best_knnr = pd.read_pickle(os.path.join(current_dir, "../data/pickle/best_knnr_model.pkl"))


# print("Validation (Classifier):")
# evaluate(best_knnc,os.path.join(current_dir, "../data/pickle/tf_idf_statements_valid.pkl"),
#          os.path.join(current_dir, "../data/pickle/semi_processed_valid.pkl"), is_regressor=False)

# print("Validation (Regressor):")
# evaluate(best_knnr, os.path.join(current_dir, "../data/pickle/tf_idf_statements_valid.pkl"),
#          os.path.join(current_dir, "../data/pickle/semi_processed_valid.pkl"), is_regressor=True)

# print("Test (Classifier):")
# evaluate(best_knnc, os.path.join(current_dir, "../data/pickle/tf_idf_statements_test.pkl"),
#          os.path.join(current_dir, "../data/pickle/semi_processed_test.pkl"), is_regressor=False)

# print("Test (Regressor):")
# evaluate(best_knnr, os.path.join(current_dir, "../data/pickle/tf_idf_statements_test.pkl"),
#          os.path.join(current_dir, "../data/pickle/semi_processed_test.pkl"), is_regressor=True)

# while True:
#     statement = input("Enter a statement to classify: ")
#     classify_statement(statement)