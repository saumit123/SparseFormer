import tensorflow as tf
from tensorflow.keras import layers

class SparseSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, window_size=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, heads, seq, depth)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        # Sparse windowed attention
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(self.depth))
        local_mask = self._create_local_mask(seq_len)
        scores = tf.where(local_mask, scores, tf.fill(tf.shape(scores), -1e9))

        attn_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attn_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(concat)

    def _create_local_mask(self, seq_len):
        # mask[i, j] = True if |i-j| <= window_size
        indices = tf.range(seq_len)
        i = tf.expand_dims(indices, axis=0)
        j = tf.expand_dims(indices, axis=1)
        mask = tf.abs(i - j) <= self.window_size
        mask = tf.expand_dims(mask, axis=0)  # (1, seq_len, seq_len)
        return tf.expand_dims(mask, axis=1)  # (1, 1, seq_len, seq_len)
    


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.att = SparseSelfAttention(d_model, num_heads)
        self.norm1 = layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.att(x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)


class SparseTransformer(tf.keras.Model):
    def __init__(self, vocab_size, max_len, num_layers=2, d_model=128, num_heads=4, ff_dim=256):
        super().__init__()
        self.embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = layers.Embedding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_layers)]
        self.out = layers.Dense(vocab_size)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos = tf.range(0, seq_len)
        x = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.out(x)
