import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess_imdb(vocab_size=5000, max_len=100):
    data = tfds.load("imdb_reviews", split="train", as_supervised=True)
    texts = []
    for text, _ in tfds.as_numpy(data):
        texts.append(text.decode("utf-8"))

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            input_seq = seq[:i]
            target_seq = seq[i]
            input_seq = [0]*(max_len - len(input_seq)) + input_seq[-max_len:]
            X.append(input_seq)
            y.append(target_seq)

    return tf.constant(X), tf.constant(y), vocab_size
