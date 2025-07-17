import streamlit as st
import tensorflow as tf
from sparse_transformer import SparseTransformer
from utils import preprocess_imdb
from inference import generate_text

st.title("🔍 Sparse Transformer Demo")
st.markdown("Enter a seed sentence and generate text using a simplified Sparse Transformer.")

# Load tokenizer and model
X, y, vocab_size = preprocess_imdb()
max_len = X.shape[1]

model = SparseTransformer(vocab_size, max_len, num_layers=2, d_model=128, num_heads=4)
model.load_weights("sparse_transformer_imdb.h5")

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(["dummy"])  # Required to initialize

seed_text = st.text_input("Enter seed text:", "The movie was")

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_text(seed_text, model, tokenizer)
        st.success("Generated Text:")
        st.write(output)
