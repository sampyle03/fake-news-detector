#data_processor.py
import pandas as pd
import os
import sys
import csv
from nltk.corpus import stopwords
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer as ps
from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams
from nltk import FreqDist
from scipy.sparse import csr_matrix # Data type for storing which words are in a statement
import pickle as pkl
from math import log10 as math_log
from nltk import pos_tag as nltk_pos_tag
from nltk import ne_chunk as nltk_ne_chunk
from nltk.tree import Tree as nltk_tree
from sklearn.feature_extraction.text import TfidfVectorizer

# import numpy as np
# import nltk
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """

    tokens = nltk_pos_tag(word_tokenize(text)) # Tokenizes text and gets the parts of speech tags of the tokens
    ner_chunks = nltk_ne_chunk(tokens) # Gets the named entities of the tokens

    tags = []
    for chunk in ner_chunks:
        if isinstance(chunk, nltk_tree): # if chunk is a named entity
            entity_type = chunk.label() # get type of named entity
            entity_words = [word for (word, pos) in chunk] # get words in the named entity
            entity_phrase = ' '.join(entity_words)  # join words of entiity into single phrase
            tags.append((entity_phrase, entity_type)) # Set word's tag to its entity type
        else: # if chunk is a regular token, then set its tag to its POS tag
            word, pos = chunk
            tags.append((word, pos))

    if tokens and tokens[0][0].lower() == "says" and tokens[1][0].lower() == "that": # if first two tokens are "says that", remove them
        tokens = tokens[2:]
        tags = tags[2:]
    elif tokens and tokens[0][0].lower() == "says": # if first token is "says", remove it
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
        token = LEMMATIZER.lemmatize(token)

        #if t[0] not in stopwords and is not a fully built up of punctuation marks
        if token not in STOPWORDS and re.match(r'^[^\w\s]+$', token) is None:
            yield (token.lower(), t[1])



def detect_ngrams(tokens):
    """
    Function: detects ngrams in tokens and returns which those most commonly found
    Parameters: tokens - the list of tokens to detect ngrams in
    Yields: ngram (tuple) - one of the most commonly found ngrams
    """
    # ngrams = bigrams + trigrams + quadgrams
    ngrams = list(nltk_ngrams(tokens, 2)) + list(nltk_ngrams(tokens, 3)) + list(nltk_ngrams(tokens, 4))
    ngram_count = FreqDist(ngrams) # counts the frequency of each ngram
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

def custom_tokenizer(text):
    """
    Function: Tokenizes text using custom tokenizer features
    Parameters: text - the text to be tokenized
    Returns: tokens - the list of tokens
    """
    tokens = list(clean_text(text))
    with open(os.path.join(current_dir, "../data/pickle/common_ngrams.pkl"), "rb") as f:
        common_ngrams = pkl.load(f)
    tokens += list(tokenize_ngrams(common_ngrams, tokens))
    return [f"{word}_{tag}" for (word, tag) in tokens]

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

def load_data(data_path, file_name = 'semi_processed_data.pkl'):
    """
    Function: Loads all valid data entried and saves it to a pkl file found it the data folder; only to be used once
    Parameters: None
    Returns: words_in_corpus (list) - the list of words in the corpus
    """

    os.system('cls')
    print(f"Loading {file_name} data...")
    # Load data from tsv file
    #chosen_columns = ["id", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation","barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"] - this considers counts of speaker's ratings
    chosen_columns = ["id", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "context"]
    loaded_data = pd.DataFrame(columns=chosen_columns)
    with open(data_path, encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"') # https://www.programiz.com/python-programming/reading-csv-files?utm_source=chatgpt.com
        possible_ratings = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        for row in rd:

            # Exclude rows with incomplete data
            if (len(row) == 14):

                # Exclude rows with incorrect data types
                if ((isinstance(row[0], str)) and (isinstance(row[1], str)) and (row[1] in possible_ratings)
                and isinstance(row[2], str) and isinstance(row[3], str) and isinstance(row[4], str)
                and isinstance(row[5], str) and isinstance(row[6], str) and isinstance(row[7], str)
                and isinstance(row[13], str)):
                    
                    #adds row to pandas table
                    loaded_data = pd.concat([loaded_data, pd.DataFrame([[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[13]]], columns=chosen_columns)], ignore_index=True)
                    loaded_data.loc[loaded_data['label'] == 'true', 'ordinal_label'] = 5
                    loaded_data.loc[loaded_data['label'] == 'mostly-true', 'ordinal_label'] = 4
                    loaded_data.loc[loaded_data['label'] == 'half-true', 'ordinal_label'] = 3
                    loaded_data.loc[loaded_data['label'] == 'barely-true', 'ordinal_label'] = 2
                    loaded_data.loc[loaded_data['label'] == 'false', 'ordinal_label'] = 1
                    loaded_data.loc[loaded_data['label'] == 'pants-fire', 'ordinal_label'] = 0


    loaded_data.to_pickle(os.path.join(current_dir,('../data/pickle/' + str(file_name))))
    input("- Data Loaded\n- Press Enter")


def tokenize_data(data_path, train_valid_test = 'train'):
    """
    Function: Loads data from pkl file and tokenizes statements, saving it as a seperate pkl file
    Parameters: data_path (str) - the path to the pkl file to be loaded
    Returns: None """
    # Load data from pkl file
    os.system('cls')
    print("Loading pkl file...")
    unprocessed_data = pd.read_pickle(os.path.join(current_dir, data_path))

    # Process data

    os.system('cls')
    print("Tokenizing statements...")
    print("This will take a couple of hours to run...")
    unprocessed_data['statement'] = unprocessed_data['statement'].apply(lambda x: list(clean_text(x)))

    # Detect ngrams
    common_ngrams = list(detect_ngrams(unprocessed_data['statement'].sum()))
    # Add tokenized ngrams to data
    unprocessed_data['ngrams'] = unprocessed_data['statement'].apply(lambda x: list(tokenize_ngrams(common_ngrams, x)))
    # do remove_ngram_tokens
    unprocessed_data['statement'] = unprocessed_data.apply(lambda x: remove_ngram_tokens(x['statement'], x['ngrams']), axis=1)

    file_name = f'tokenized_statements_{train_valid_test}'
    unprocessed_data.to_pickle(os.path.join(current_dir, f'../data/pickle/{file_name}.pkl'))
    input("- Statements Tokenized\n- Press Enter")
    return common_ngrams

def concatenate_statements_ngrams(data_path, train_valid_test = 'train'):
    """
    Function: Loads data from a pkl file and concatenates statements with ngrams
    Parameters: data_path - the path to the pkl file to be loaded
    Returns: None
    """

    os.system('cls')
    print("Loading tokenized statements...")

    tokenized_statements = pd.read_pickle(os.path.join(current_dir, data_path))


    # Concatenate statements with ngrams
    tokenized_statements['statement'] = tokenized_statements['statement'] + tokenized_statements['ngrams']
    tokenized_statements.drop(columns=['ngrams'], inplace=True)

    file_name = f'statements_ngrams_{train_valid_test}'
    tokenized_statements.to_pickle(os.path.join(current_dir, f'../data/pickle/{file_name}.pkl'))
    input("- Statements Concatenated\n- Press Enter")

"""
def process_statements(data_path, train_valid_test = 'train'):
    \"""
    Function: Loads data from pkl file, processes it and saves it to another pkl file
    Parameters: data_path - the path to the pkl file to be loaded
    Returns: None
    \"""

    os.system('cls')
    print("Reading tokenized statements...")

    tokenized_statements = pd.read_pickle(os.path.join(current_dir, data_path))
    
    # keep track of words used in corpus
    os.system('cls')
    print("Generating word list...")
    print("Just a moment...")

    words_in_corpus = {}
    statements_list = []
    for statement in tokenized_statements['statement']:
        statements_list.append([0 for _ in range(len(words_in_corpus))])
        for word in statement:
            if word not in words_in_corpus:
                words_in_corpus[word] = len(words_in_corpus)
                for old_statements in statements_list[:-1]:
                    old_statements.append(0)
                statements_list[len(statements_list)-1].append(1)
            else:
                statements_list[len(statements_list)-1][words_in_corpus[word]] += 1

    file_name = f'words_in_corpus_{train_valid_test}'
    with open(os.path.join(current_dir, f'../data/pickle/{file_name}.pkl'), 'wb') as file:
        pkl.dump(words_in_corpus, file)
    
    statements_matrix = csr_matrix(statements_list) # (statement, word)   num_of_appearances



    # Calculate the inverse document frequency of each word
    os.system('cls')
    print("Calculating document frequency of statements...")

    coo_matrix = statements_matrix.tocoo()
    num_of_docs = statements_matrix.shape[0]
    docs_containing_terms = {}
    statement_no = 0
    words_in_statement = set()
    num_words_in_statements = {}
    count = 0
    for i in range(len(coo_matrix.col)):
        if i != statement_no:
            num_words_in_statements[statement_no] = count
            statement_no += 1
            words_in_statement = set()
            count = 0
        if coo_matrix.col[i] not in words_in_statement:
            words_in_statement.add(coo_matrix.col[i])
            if coo_matrix.col[i] in docs_containing_terms:
                docs_containing_terms[coo_matrix.col[i]] += 1
            else:
                docs_containing_terms[coo_matrix.col[i]] = 1
        count += 1

    
    term_idfs = {}
    for word_no, appearances in docs_containing_terms.items():
        term_idfs[word_no] = math_log(num_of_docs/appearances)
    
    file_name = f'term_idfs_{train_valid_test}'
    with open(os.path.join(current_dir, f'../data/pickle/{file_name}.pkl'), 'wb') as file:
        pkl.dump(term_idfs, file)
    
    
    # Calculate the term frequency-inverse document frequency vectors of each statement
    os.system('cls')
    print("Calculating tf-idf vectors of each statement...")
    print("Just a moment...")

    tf_idf_statements = []
    statement_tf_idf = [0 for _ in range(len(words_in_corpus))]
    current_doc = None

    for i in range(coo_matrix.nnz):
        doc_no = coo_matrix.row[i]
        term_no = coo_matrix.col[i]
        data = coo_matrix.data[i]
        
        if doc_no != current_doc:
            if current_doc is not None:
                tf_idf_statements.append(statement_tf_idf)
            current_doc = doc_no
            statement_tf_idf = [0 for _ in range(len(words_in_corpus))]
        
        word_tf = data / num_words_in_statements[doc_no]
        statement_tf_idf[term_no] = word_tf * term_idfs[term_no]
    
    tf_idf_statements.append(statement_tf_idf)
    tf_idf_statements = csr_matrix(tf_idf_statements) # (statement_no, word_no)   tf-idf value

    # Save the tf-idf vectors to a pkl file
    os.system('cls')
    print("Saving tf-idf vectors...")
    print("Just a moment...")
    file_name = f'tf_idf_statements_{train_valid_test}'
    with open(os.path.join(current_dir, f'../data/pickle/{file_name}.pkl'), 'wb') as file:
        pkl.dump(tf_idf_statements, file)
    input("- tf-idf Vectors Saved\n- Press Enter")
"""

def build_tfidf():
    """
    Function: Builds the tf-idf vectorizer and vectors for the training, validation and test data, saving them to pkl files
    Parameters: None
    Returns: None
    """
    # load semi-processed datasets
    train = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_train.pkl"))
    valid = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_valid.pkl"))
    test  = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_test.pkl"))

    # concatenate all datasets to create one vectorizer for all of them
    all_texts = pd.concat([train['statement'], valid['statement'], test['statement']], ignore_index=True)

    # create tf-idf vectorizer
    vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer, lowercase = False, preprocessor  = None, token_pattern = None)
    vectorizer.fit(all_texts.tolist())

    #convert the vectorizer to a pickle file for later use in other scripts
    pd.to_pickle(vectorizer,os.path.join(current_dir, "../data/pickle/tfidf_vectorizer.pkl"))

    # transform datasets to tf-idf matrices then save as a pickle file for their corresponding dataset
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        X = vectorizer.transform(df['statement'].tolist())
        pd.to_pickle(X, os.path.join(current_dir, f"../data/pickle/tf_idf_statements_{name}.pkl"))
        print(f"Saved tf-idf matrix for {name}: shape {X.shape}")




# Main
current_dir = os.path.dirname(__file__)
train_data_path = os.path.join(current_dir, "../data/train.tsv") #LIAR dataset
valid_data_path = os.path.join(current_dir, "../data/valid.tsv") #LIAR dataset
test_data_path = os.path.join(current_dir, "../data/test.tsv") #LIAR dataset

"""------------------------------------- TRAINING DATA (Uncomment below to run) -------------------------------------"""

# # load training data
# load_data(train_data_path, 'semi_processed_train.pkl') # load training data

# # tokenize train statements and find common ngrams
# common_ngrams = tokenize_data('../data/pickle/semi_processed_train.pkl', "train")
# with open(os.path.join(current_dir, '../data/pickle/common_ngrams.pkl'), 'wb') as file:
#     pkl.dump(common_ngrams, file)

# # concatenate statements with ngrams
# concatenate_statements_ngrams('../data/pickle/tokenized_statements.pkl', "train")

# # process train data
# # process_statements('../data/pickle/statements_ngrams.pkl', "train") # now replaced by build_tfidf





# """------------------------------------- VALIDATION DATA -------------------------------------"""

# # load validation data
# load_data(valid_data_path, 'semi_processed_valid.pkl')

# # tokenize validation statements, common ngrams already found in training data
# tokenize_data('../data/pickle/semi_processed_valid.pkl', "valid")

# # concatenate statements with ngrams
# concatenate_statements_ngrams('../data/pickle/tokenized_statements_valid.pkl', "valid")

# # process validation data
# # process_statements('../data/pickle/statements_ngrams_valid.pkl', "valid") # now replaced by build_tfidf








# """------------------------------------- TEST DATA -------------------------------------"""
# # load test data
# load_data(test_data_path, 'semi_processed_test.pkl')

# # tokenize test statements, common ngrams already found in training data
# tokenize_data('../data/pickle/semi_processed_test.pkl', "test")

# # concatenate statements with ngrams
# concatenate_statements_ngrams('../data/pickle/tokenized_statements_test.pkl', "test")

# # process test data
# # process_statements('../data/pickle/statements_ngrams_test.pkl', "test") # now replaced by build_tfidf






# """------------------------------------- BUILD TF-IDF -------------------------------------"""

# build_tfidf()