BATCH_SIZE = 16
SEED = 42

def clean_text(text):
    """
    Function: cleans the text by removing the word "says" at the beginning of each text
    Parameters: text (str)
    Returns: cleaned text (str)
    """
    if text.startswith("Says"):
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
    import os
    import pandas as pd
    import tensorflow as tf
    from transformers import create_optimizer, BertConfig, AutoTokenizer, TFAutoModelForSequenceClassification
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


    os.system('cls')
    print("Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_df = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_train.pkl"))
    val_df   = pd.read_pickle(os.path.join(current_dir, "../data/pickle/semi_processed_valid.pkl"))
    train_texts  = train_df["statement"]
    train_labels = train_df["ordinal_label"]
    val_texts    = val_df["statement"]
    val_labels   = val_df["ordinal_label"]


    # Load the data
    os.system('cls')
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_tokens = tokenize(train_texts, tokenizer)
    val_tokens = tokenize(val_texts, tokenizer)


    # Split data into train and validation
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
    config = BertConfig.from_pretrained( # https://huggingface.co/docs/transformers/en/main_classes/configuration
        "bert-base-uncased",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        num_labels=6, # number of labels -> pants-fire, false, barely-true, half-true, mostly-true, true
    )

    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6) # https://stackoverflow.com/questions/62327803/having-6-labels-instead-of-2-in-hugging-face-bertforsequenceclassification
    for layer in model.bert.encoder.layer:
        layer.trainable = True # unfreeze the layers of the model


    num_of_epochs = 12 # Epochs used for training the model
    # Optimizer for the model
    total_train_steps = (len(train_labels) // BATCH_SIZE) * num_of_epochs # https://huggingface.co/docs/transformers/en/tasks/question_answering
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,  
        num_warmup_steps=int(0.1 * total_train_steps),
        num_train_steps=total_train_steps,
        weight_decay_rate=0.01
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss function for multi-class classification
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")] # accuracy metric for multi-class classification

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    weights = compute_class_weight("balanced", classes= np.unique(train_labels), y=train_labels.values)
    class_weights = enumerate(weights)
    class_weights = dict(class_weights)

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    os.system('cls')
    print("Training model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=num_of_epochs, callbacks=[callback], class_weight=class_weights)
    print("Model trained!")
    input("Press Enter to continue...")

    # Save model and tokenizer
    os.system('cls')
    input("Press Enter to save the model (otherise, press Ctrl+C to exit):")
    print("Saving model...")
    model.save_pretrained(os.path.join(current_dir, "../models/bert_classifier"))
    tokenizer.save_pretrained(os.path.join(current_dir, "../models/bert_classifier"))

    # Save model weights
    weights_path = os.path.join(current_dir, "../models/bert_classifier_weights")
    model.save_weights(weights_path) 

    # Output the class distribution of the labels
    print("----------------------- Class Distrubution -----------------------")
    print("Class distribution of the training dataset:")
    print(train_labels.value_counts(normalize=True))
    print("\n\nClass distribution of the validation dataset:")
    print(val_labels.value_counts(normalize=True))
    

train_bert_classifier()
#continue_training_from_epoch(8, 12)