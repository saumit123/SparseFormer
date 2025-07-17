import tensorflow as tf
from sparse_transformer import SparseTransformer
from utils import preprocess_imdb

def generate_text(seed_text, model, tokenizer, max_len=100, n_words=20):
    output = tokenizer.texts_to_sequences([seed_text])[0]
    for _ in range(n_words):
        input_seq = output[-max_len:]
        input_seq = [0] * (max_len - len(input_seq)) + input_seq
        input_tensor = tf.constant([input_seq])
        pred = model(input_tensor)
        next_word_id = tf.argmax(pred[0, -1]).numpy()
        output.append(next_word_id)

    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    return " ".join([inv_vocab.get(i, "?") for i in output])
