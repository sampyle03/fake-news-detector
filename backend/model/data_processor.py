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

    # Load data from tsv file
    chosen_columns = ["id", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation","barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]
    loaded_data = pd.DataFrame(columns=chosen_columns)
    with open(data_path, encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        possible_ratings = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        words = []
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

    loaded_data.to_pickle(os.path.join(current_dir,'../data/data.pkl'))
    input("- Data Loaded\n- Press Enter")

def process_data():
    """
    Function: Loads data from pkl file, processes it and saves it to another pkl file
    Parameters: None
    Returns: None
    """

    features = pd.DataFrame()

    # Load data from pkl file
    print("Loading pkl file...")
    unprocessed_data = pd.read_pickle(os.path.join(current_dir,'../data/data.pkl'))

    # Process data

    words = {}
    os.system('cls')
    print("Tokenizing statements...")
    print("This may take a minute...")
    unprocessed_data['statement'] = unprocessed_data['statement'].apply(lambda x: list(clean_text(x)))

    # keep track of words used in statements
    os.system('cls')
    print("Generating word list...")
    count = 0
    for statement in unprocessed_data['statement']:
        for word in statement:
            if word not in words:
                words[count] = word
                count += 1
    
    os.system('cls')
    print(unprocessed_data['statement'].head())
    print("Done")
    




current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "../data/train.tsv") #LIAR dataset
#load_data(data_path)
process_data()