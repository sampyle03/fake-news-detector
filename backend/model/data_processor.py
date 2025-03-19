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

def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """
    cleaned_text = re.sub(r"[^\w\s']+", " ", text) # https://stackoverflow.com/questions/31191986/br-tag-screwing-up-my-data-from-scraping-using-beautiful-soup-and-python
    tokens = word_tokenize(cleaned_text) # tokenizes text
    if tokens[0].lower() == "says":
        tokens = tokens[1:]
    lemmatizer = WordNetLemmatizer()
    for t in tokens:
        if t not in set(stopwords.words('english')):
            yield lemmatizer.lemmatize(t.lower())
            # yield (ps.stem(t.lower())) # can uncomment to use stemming

def detect_ngrams(tokens): # detects ngrams in tokens and returns which those most commonly found
    # ngrams = bigrams + trigrams + quadgrams
    ngrams = list(nltk_ngrams(tokens, 2)) + list(nltk_ngrams(tokens, 3)) + list(nltk_ngrams(tokens, 4))
    ngram_count = FreqDist(ngrams)
    for token, count in ngram_count.items():
        if len(token) == 2 and count > 17: # bigram considered "commonly found" if it appears more than 14 times
            yield token
        elif len(token) == 3 and count > 7:
            yield token
        elif len(token) == 4 and count > 5:
            yield token

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

def load_data(data_path):
    """
    Function: Loads all valid data entried and saves it to a pkl file found it the data folder; only to be used once
    Parameters: None
    Returns: words
    """

    os.system('cls')
    print("Loading data...")
    # Load data from tsv file
    chosen_columns = ["id", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation","barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]
    loaded_data = pd.DataFrame(columns=chosen_columns)
    with open(data_path, encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        possible_ratings = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        for row in rd:

            # Exclude rows with incomplete data
            if (len(row) == 14):

                # Exclude rows with incorrect data types (NOT WORKING, row[8] - row[12] isinstance str)
                if ((isinstance(row[0], str)) and (isinstance(row[1], str)) and (row[1] in possible_ratings)
                and isinstance(row[2], str) and isinstance(row[3], str) and isinstance(row[4], str)
                and isinstance(row[5], str) and isinstance(row[6], str) and isinstance(row[7], str)
                and row[8].isnumeric() and row[9].isnumeric() and row[10].isnumeric()
                and row[11].isnumeric() and row[12].isnumeric() and isinstance(row[13], str)):
                    
                    #adds row to pandas table
                    loaded_data = pd.concat([loaded_data, pd.DataFrame([row], columns=chosen_columns)], ignore_index=True)

    loaded_data.to_pickle(os.path.join(current_dir,'../data/pickle/semi_processed_data.pkl'))
    input("- Data Loaded\n- Press Enter")

def tokenize_data(data_path):
    """
    Function: Loads data from pkl file and tokenizes statements, saving it as a seperate pkl file
    Parameters: data_path - the path to the pkl file to be loaded
    Returns: None """
    # Load data from pkl file
    os.system('cls')
    print("Loading pkl file...")
    unprocessed_data = pd.read_pickle(os.path.join(current_dir, data_path))

    # Process data

    os.system('cls')
    print("Tokenizing statements...")
    print("This will take a minute or two...")
    unprocessed_data['statement'] = unprocessed_data['statement'].apply(lambda x: list(clean_text(x)))

    # Detect ngrams
    common_ngrams = list(detect_ngrams(unprocessed_data['statement'].sum()))
    # Add tokenized ngrams to data
    unprocessed_data['ngrams'] = unprocessed_data['statement'].apply(lambda x: list(tokenize_ngrams(common_ngrams, x)))

    unprocessed_data.to_pickle(os.path.join(current_dir,'../data/pickle/tokenized_statements.pkl'))
    input("- Statements Tokenized\n- Press Enter")
    return common_ngrams

def concatenate_statements_ngrams(data_path):
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

    tokenized_statements.to_pickle(os.path.join(current_dir, '../data/pickle/statements_ngrams.pkl'))
    input("- Statements Concatenated\n- Press Enter")

def process_statements(data_path):
    """
    Function: Loads data from pkl file, processes it and saves it to another pkl file
    Parameters: data_path - the path to the pkl file to be loaded
    Returns: None
    """

    os.system('cls')
    print("Reading tokenized statements...")

    tokenized_statements = pd.read_pickle(os.path.join(current_dir, data_path))
    
    # keep track of words used in corpus
    os.system('cls')
    print("Generating word list...")
    print("Just a moment...")

    words_in_corpus = {}
    statements_list = []
    unique_in_corpus = 0
    for statement in tokenized_statements['statement']:
        statements_list.append([0 for _ in range(len(words_in_corpus))])
        for word in statement:
            if word not in words_in_corpus:
                words_in_corpus[word] = unique_in_corpus
                unique_in_corpus += 1
                for old_statements in statements_list[:-1]:
                    old_statements.append(0)
                statements_list[len(statements_list)-1].append(1)
            else:
                statements_list[len(statements_list)-1][words_in_corpus[word]] += 1

    with open(os.path.join(current_dir, '../data/pickle/words_in_corpus.pkl'), 'wb') as file:
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
    
    with open(os.path.join(current_dir, '../data/pickle/term_idfs.pkl'), 'wb') as file:
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
    tf_idf_statements = csr_matrix(tf_idf_statements)

    # Save the tf-idf vectors to a pkl file
    os.system('cls')
    print("Saving tf-idf vectors...")
    print("Just a moment...")
    with open(os.path.join(current_dir, '../data/pickle/tf_idf_statements.pkl'), 'wb') as file:
        pkl.dump(tf_idf_statements, file)
    input("- tf-idf Vectors Saved\n- Press Enter")




# Main
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "../data/train.tsv") #LIAR dataset


#load_data(data_path)

common_ngrams = tokenize_data('../data/pickle/semi_processed_data.pkl') # tokenize statements and find common ngrams
with open(os.path.join(current_dir, '../data/pickle/common_ngrams.pkl'), 'wb') as file:
    pkl.dump(common_ngrams, file)

concatenate_statements_ngrams('../data/pickle/tokenized_statements.pkl')

process_statements('../data/pickle/statements_ngrams.pkl')