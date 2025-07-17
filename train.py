from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from sparse_transformer import SparseTransformer
from utils import preprocess_imdb

# Load and preprocess
X, y, vocab_size = preprocess_imdb()
max_len = X.shape[1]

# Build model
model = SparseTransformer(vocab_size, max_len, num_layers=2, d_model=128, num_heads=4)
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Train
model.fit(X, y, batch_size=32, epochs=3)
model.save_weights("sparse_transformer_imdb.h5")
