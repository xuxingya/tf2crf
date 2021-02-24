import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        units: Positive integer, dimensionality of the output space. If it is None, the last dim of input must be num_classes
        chain_initializer: the initialize method for transitions, default orthogonal.
        regularizer: Regularizer for crf transitions, can be 'l1', 'l2' or other tensorflow regularizers.

    Input shape:
        nD tensor with shape `(batch_size, sentence length, features)` or `(batch_size, sentence length, num_classes)`.

    Output shape:
        in training:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`
            inputs: the input tensor of the CRF layer with shape `(batch_size, sentence length, num_classes)`
            sequence_lengths: true sequence length of inputs with shape `(batch_size)`
            self.transitions: the internal transition parameters of CRF with shape `(num_classes, num_classes)`
        in predicting:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`

    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """

    def __init__(self, units=None, chain_initializer="orthogonal", regularizer=None, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.regularizer = regularizer
        self.transitions = None
        self.supports_masking = True
        self.mask = None
        self.accuracy_fn = tf.keras.metrics.Accuracy()
        self.units = units
        if units is not None:
            self.dense = tf.keras.layers.Dense(units)

    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({
            "chain_initializer": "orthogonal"
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.units:
            units = self.units
        else:
            units = input_shape[-1]
        self.transitions = self.add_weight(
            name="transitions",
            shape=[units, units],
            initializer=self.chain_initializer,
            regularizer=self.regularizer
        )

    def call(self, inputs, mask=None, training=False):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
        if self.units:
            inputs = self.dense(inputs)
        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        return viterbi_sequence, inputs, sequence_lengths, self.transitions

