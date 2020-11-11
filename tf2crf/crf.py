import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        chain_initializer: the initialize method for transitions, default orthogonal.

    Input shape:
        nD tensor with shape `(batch_size, sentence length, num_classes)`.

    Output shape:
        nD tensor with shape: `(batch_size, sentence length, num_classes)`.

    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """

    def __init__(self, chain_initializer="orthogonal", **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.transitions = None
        self.supports_masking = True
        self.mask = None
        self.accuracy_fn = tf.keras.metrics.Accuracy()

    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({
            "chain_initializer": "orthogonal"
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        units = input_shape[-1]
        self.transitions = self.add_weight(
            name="transitions",
            shape=[units, units],
            initializer=self.chain_initializer,
        )

    def call(self, inputs, mask=None, training=None):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)

        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        return viterbi_sequence, inputs, sequence_lengths, self.transitions

