# bert_classifier.py
import os
import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf

def tokenize(texts):
    """
    Function: tokenizes the texts using the BERT tokenizer.
    Parameters: texts (list)
    Returns: tokens (dict)
    """
    texts = list(texts)
    return tokenizer.batch_encode_plus(texts, max_length=128, padding='max_length', truncation=True, return_tensors="tf")


current_dir = os.path.dirname(os.path.abspath(__file__))
semi_processed_data = pd.read_pickle(os.path.join(current_dir, '../data/pickle/semi_processed_data.pkl'))

# Load the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenize(semi_processed_data['statement'])

labels = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true': 5}
semi_processed_data['ordinal_label'] = semi_processed_data['label'].map(labels)


dataset = tf.data.Dataset.from_tensor_slices(( # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    tokens,
    semi_processed_data['ordinal_label'].values 
))
BATCH_SIZE = 16
dataset = dataset.shuffle(buffer_size=len(semi_processed_data)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)