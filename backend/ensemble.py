import os
from tweet_scraper import scrape_tweet
import pandas as pd
import tensorflow as tf
from transformers import create_optimizer, AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import numpy as np
from model_training.data_processor import custom_toknizer
from model_training.model import RoundedKNeighborsRegressor
from sklearn.metrics import f1_score
import logging
import warnings
from transformers import logging as hf_logging
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import shap
import torch
import re

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

def clean_post(text):
    """
    Function: cleans text by removing a link and attempting to remove whitespaces
    Parameters: text (str)
    Returns: cleaned text (str) - text with no links and no unnecesary spaces that may be caused by web scraping
    """
    text = re.sub(r'https?://\S+', '', text) # remove links
    text = text.replace('\r', ' ').replace('\n', ' ') # remove line breaks
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = re.sub(r'\s+([.,!?:;)\]”’])', r'\1', text) # remove space before punctuation
    text = re.sub(r'([(\[“‘])\s+', r'\1', text) # remove space after punctuation
    return text.strip()


class Ensemble:
    def __init__(self, current_dir, w1, w2, w3, t=0.5):

        self.bert_tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_dir, "models/bert_classifier"))
        self.bert_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(current_dir, "models/bert_classifier"))

        self.knn_vectorizer = pd.read_pickle(os.path.join(current_dir, "data/pickle/tfidf_vectorizer.pkl"))
        self.knnc = pd.read_pickle(os.path.join(current_dir, "data/pickle/best_knnc_model.pkl"))
        self.knnr = pd.read_pickle(os.path.join(current_dir, "data/pickle/best_knnr_model.pkl"))

        self.bert_threshold = 0.4
        self.knnr_threshold = 2.1

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.total_threshold = t

        bert_pipeline = pipeline("text-classification", model=self.bert_model, tokenizer=self.bert_tokenizer, return_all_scores=True)
        self.shap_explainer = shap.Explainer(bert_pipeline) # https://stackoverflow.com/questions/73568255/what-is-the-correct-way-to-obtain-explanations-for-predictions-using-shap

    def predict(self, text, return_shap=False):
        text = [clean_post(text)]

        # BERT prediction
        bert_vector = self.bert_tokenizer(text, padding = True, truncation = True, max_length = 256, return_tensors = "tf")
        bert_out = self.bert_model.predict(bert_vector, verbose = 0)
        logits = bert_out.logits
        bert_probs = tf.nn.softmax(logits, axis=1).numpy()[0] # convert softmax of logits to probabilities to get the probability of each class
        bert_prob_1 = bert_probs[1] # probability of class 1 (true)

        # k-NN classifier probabilities
        knn_vector = self.knn_vectorizer.transform(text)
        knnc_probs = self.knnc.predict_proba(knn_vector)[0]
        knnc_prob_1 = knnc_probs[1]

        # k-NN regressor score mapped to [0,1]
        knnr_raw = self.knnr.predict(knn_vector)[0]
        knnr_prob_1 = np.clip(knnr_raw / 5.0, 0, 1)

        # weighted ensemble score
        combined_score = (
            self.w1 * bert_prob_1 +
            self.w2 * knnc_prob_1 +
            self.w3 * knnr_prob_1
        ) / (self.w1 + self.w2 + self.w3)  # normalise weights

        # apply thresholds
        final_prediction = int(combined_score >= self.total_threshold)

        if return_shap: # if return_shap is True, return the SHAP values for explanation of BERT model classification
            return (final_prediction, (bert_prob_1, knnc_prob_1, knnr_prob_1), combined_score, self.shap_explainer(text, max_evals=50))
        else:
            return (final_prediction, (bert_prob_1, knnc_prob_1, knnr_prob_1), combined_score)


    def evaluate(self, current_dir, ordinal_path):
        """
        Function: Loops over the test set, calls predict() for each statement
        Parameters: current_dir (str): path to the current directory, ordinal_path (str): path to the ordinal labels pkl file
        Returns: y_
        """
        
        # Load the semi-processed test dataset and ordinal labels
        test_df  = pd.read_pickle(os.path.join(current_dir, "data/pickle/semi_processed_test.pkl"))
        ordinals = (pd.read_pickle(ordinal_path)["ordinal_label"].map({0: np.int64(0), 1: np.int64(0),2: np.int64(0), 3: np.int64(0),4: np.int64(1), 5: np.int64(1)}))

        y_true = []
        combined_scores = []

        # predict each statement in the test dataset
        for i in range(len(test_df)):
            statement = test_df["statement"][i]
            pred_label, _, score = self.predict(statement)

            # append the predicted label and the ordinal label to the lists
            y_true.append(ordinals[i])
            combined_scores.append(score)

        # convert y_true and combined_scores to numpy arrays
        y_true         = np.array(y_true, dtype=int)
        combined_scores = np.array(combined_scores, dtype=float)

        return y_true, combined_scores





current_dir = os.path.dirname(os.path.abspath(__file__))

# print("Best regressor threshold:", best_thresh, "→ F1:", best_f1)

# w1s = [1, 0.8, 0.6, 0.4, 0.2, 1e-10]
# w2s_lists = [[1e-10], [0.2, 1e-10], [0.4, 0.2, 1e-10], [0.6, 0.4, 0.2, 1e-10], [0.8, 0.6, 0.4, 0.2, 1e-10], [1, 0.8, 0.6, 0.4, 0.2, 1e-10]]


# best_overall = (-1, None)  # (best f1+acc, (w1,w2,w3,t))
# ordinal_path = os.path.join(current_dir, "data/pickle/semi_processed_test.pkl")

# for i, w1 in enumerate(w1s):
#     for w2 in w2s_lists[i]:
#         w3 = 1 - (w1 + w2)
#         w3 = 1e-10 if w3 == 0 else w3

#         ens = Ensemble(current_dir, w1, w2, w3)
#         # now evaluate returns raw truths and combined scores
#         y_true, scores = ens.evaluate(current_dir, ordinal_path)

#         # sweep threshold from 0.1 to 0.9 in 0.01 steps
#         best_local = (-1, 0, 0)  # (f1+acc, best_t, best_acc)
#         for t in np.linspace(0.1, 0.9, 81):
#             y_pred = (scores >= t).astype(int)
#             f1  = f1_score(y_true, y_pred)
#             acc = accuracy_score(y_true, y_pred)
#             if f1 + acc > best_local[0]:
#                 best_local = (f1 + acc, t, acc)

#         total, best_t, best_acc = best_local
#         print(f"w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f} → best t={best_t:.2f}, F1+Acc={total:.3f} (Acc={best_acc:.3f})")

#         if total > best_overall[0]:
#             best_overall = (total, (w1, w2, w3, best_t))

# print("Overall best:", best_overall[1])

best_ensemble = Ensemble(current_dir, 0.6, 0.2, 0.2, 0.45)
# print(best_ensemble.predict("Says that our small staff of 51 is still fewer than we had a decade ago, yet our caseload -- like that of other courts -- has grown."))
#print(scrape_tweet("https://x.com/Channel4News/status/1924548654653702233"))

def predict(tweet_url):
    tweet_text = scrape_tweet(tweet_url) # scrape the tweet text
    if not tweet_text: # if there is no text in the tweet, return None
        return None
    prediction_data = best_ensemble.predict(tweet_text, return_shap=True) # use the best ensemble to predict the tweet text
    return prediction_data
