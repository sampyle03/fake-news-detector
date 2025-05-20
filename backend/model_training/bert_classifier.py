# bert_classifier.py
import os
import pandas as pd
import tensorflow as tf
from transformers import create_optimizer, AutoTokenizer, TFAutoModelForSequenceClassification, BertConfig
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random



BATCH_SIZE = 16
SEED = 42

def clean_text(text):
    """
    Function: cleans the text by removing the word "says" at the beginning of each text
    Parameters: text (str)
    Returns: cleaned text (str)
    """
    if text.startswith("Says that"):
        text = text[9:] # remove the first 9 characters
    elif text.startswith("Says"):
        text = text[5:] # remove the first 5 characters
    return text

def tokenize(texts, tokenizer):
    """
    Function: tokenizes the texts using the BERT tokenizer.
    Parameters: texts (list)
    Returns: tokens (dict)
    """
    texts = list(texts)
    texts = [clean_text(text) for text in texts]

    return tokenizer.batch_encode_plus(texts, # list of texts
                                    padding='max_length', 
                                    max_length= 256, # max length of the text
                                    add_special_tokens = True, 
                                    return_attention_mask = True, 
                                    truncation=True, 
                                    return_tensors="tf")

def train_bert_classifier():
    """
    Function: trains a BERT classifier on the LIAR training dataset and saves the model and tokenizer.
    Parameters: None
    Returns: None
    """

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


    os.system('cls')
    print("Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # load pickles of semi-processed datasets
    train_df = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_train.pkl"))
    val_df   = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_valid.pkl"))

    # prepare statements and their corresponding labels (mapped to binary)
    map_labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1} # map the labels to 0 (false) and 1 (true); pants-fire -> false, false -> false, barely-true -> false, half-true -> false, mostly-true -> true, true -> true
    train_texts  = train_df["statement"]
    train_labels = train_df["ordinal_label"].map(map_labels)
    val_texts    = val_df["statement"]
    val_labels   = val_df["ordinal_label"].map(map_labels)


    # load the data
    os.system('cls')
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_tokens = tokenize(train_texts, tokenizer)
    val_tokens = tokenize(val_texts, tokenizer)


    # split data into train and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_tokens),
        train_labels.values
    )).shuffle(len(train_labels), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_tokens),
        val_labels.values
    )).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)




    # Prepare configuration of model
    os.system('cls')
    print("Loading model...")
    config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4, num_labels=2) # https://huggingface.co/docs/transformers/en/main_classes/configuration

    # prepare the model with configuration
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config) # https://stackoverflow.com/questions/62327803/having-6-labels-instead-of-2-in-hugging-face-bertforsequenceclassification if using 6 labels
    for layer in model.bert.encoder.layer:
        layer.trainable = True # unfreeze all the layers of the model


    num_of_epochs = 2 # Epochs used for training the model
    # Optimizer for the model
    total_train_steps = (len(train_labels) // BATCH_SIZE) * 3 # https://huggingface.co/docs/transformers/en/tasks/question_answering
    optimizer, schedule = create_optimizer(init_lr=3e-5, num_warmup_steps=int(0.1 * total_train_steps), num_train_steps=total_train_steps, weight_decay_rate=0.02)

    # prepare the loss function and metrics - usually used for multi-class classification
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss function for multi-class classification
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")] # accuracy metric for multi-class classification

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    weights = compute_class_weight("balanced", classes= np.unique(train_labels), y=train_labels.values)
    class_weights = enumerate(weights)
    class_weights = dict(class_weights)

    # make model training stop early if val_loss doesn'r improve within 1 epoch - only useful for a large number of epochs rather than here with 2 epochs.
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
    os.system('cls')

    # train the model
    print("Training model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=num_of_epochs, callbacks=[callback], class_weight=class_weights)
    print("Model trained!")
    input("Press Enter to continue...")

    # save model and tokenizer
    input("Press Enter to save the model (otherise, press Ctrl+C to exit):")
    print("Saving model...")
    model.save_pretrained(os.path.join(current_dir, "../models/bert_classifier"))
    tokenizer.save_pretrained(os.path.join(current_dir, "../models/bert_classifier"))

    # save model weights
    weights_path = os.path.join(current_dir, "../models/bert_classifier_weights")
    model.save_weights(weights_path) 

    # output the class distribution of the labels
    print("----------------------- Class Distrubution -----------------------")
    print("Class distribution of the training dataset:")
    print(train_labels.value_counts(normalize=True))
    print("\n\nClass distribution of the validation dataset:")
    print(val_labels.value_counts(normalize=True))

def test_saved_model():
    """
    Function: loads saved BERT model and tokenizer and tests on semi_processed_test dataset.
    Parameters: None
    Returns: None
    """

    os.system('cls')
    print("Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    map_labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1} # map the labels to 0 (false) and 1 (true); pants-fire -> false, false -> false, barely-true -> false, half-true -> false, mostly-true -> true, true -> true

    # load pickles of semi-processed datasets
    test_df = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_test.pkl"))
    test_texts  = test_df["statement"]
    test_labels = test_df["ordinal_label"].map(map_labels)

    # load the tokenizer and model
    os.system('cls')
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_dir, "../models/bert_classifier"))
    model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(current_dir, "../models/bert_classifier"))

    # ttokenize the test data
    test_tokens = tokenize(test_texts, tokenizer)

    # create the test dataset 
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_tokens),
        test_labels.values
    )).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


    total_train_steps = (len(test_labels) // BATCH_SIZE)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=int(0.1 * total_train_steps), num_train_steps=total_train_steps, weight_decay_rate=0.01)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss function for multi-class classification
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    #evaluate on the test dataset
    os.system('cls')
    print("Evaluating model...")
    results = model.evaluate(test_dataset)
    
    # calculate probabilities and predictions
    logits = model.predict(test_dataset).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    threshold = 0.41
    y_pred = (probs[:, 1] >= threshold).astype(int)
    y_true = np.array(list(test_dataset.unbatch().map(lambda x, y: y)))

    # calculate metrics
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, probs[:, 1])

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["False", "True"]))

    print("Model Results:")
    print(f"Loss: {results[0]}")
    print(f"Accuracy: {results[1]}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
    print("Model evaluated!")



# train_bert_classifier()
# input("Press Enter to test the saved model (otherise, press Ctrl+C to exit):")
# test_saved_model()