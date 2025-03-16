#data_processor.py
import pandas as pd
import os
import csv
from nltk.corpus import stopwords
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix # Data type for storing which words are in a statement

def clean_text(text):
    """
    Function: Cleans given text of stopwords and punctuation, and then lemmatizes the text
    Parameters: text (str)
    Returns: cleaned tokens
    """
    cleaned_text = re.sub(r"[^\w\s']+", " ", text) # https://stackoverflow.com/questions/31191986/br-tag-screwing-up-my-data-from-scraping-using-beautiful-soup-and-python
    tokens = word_tokenize(cleaned_text) # tokenizes text
    lemmatizer = WordNetLemmatizer()
    for t in tokens:
        if t not in set(stopwords.words('english')):
            yield lemmatizer.lemmatize(t.lower())
            # yield (ps.stem(t) # can uncomment to use stemming

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

    loaded_data.to_pickle(os.path.join(current_dir,'../data/semi_processed_data.pkl'))
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

    unprocessed_data.to_pickle(os.path.join(current_dir,'../data/tokenized_statements.pkl'))
    input("- Statements Tokenized\n- Press Enter")

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
    
    statements_matrix = csr_matrix(statements_list) # (statement, word)   num_of_appearances
    statements_matrix.to_pickle(os.path.join(current_dir, '../data/processed_statements.pkl'))
    input("- Statements Processed\n- Press Enter")

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "../data/train.tsv") #LIAR dataset
#load_data(data_path)
#tokenize_data('../data/semi_processed_data.pkl')
process_statements('../data/tokenized_statements.pkl')