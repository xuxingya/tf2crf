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
        boundary_initializer: the initialize method for boundary, default zero
        use_boundary: whether to use boundary, default true
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

    def __init__(self, units=None, chain_initializer="orthogonal", boundary_initializer='zeros', use_boundary=True, regularizer=None, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.boundary_initializer = tf.keras.initializers.get(boundary_initializer)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.transitions = None
        self.supports_masking = True
        self.mask = None
        self.units = units
        self.use_boundary = use_boundary
        self.accuracy_fn = tf.keras.metrics.Accuracy()

    def get_config(self):
        config = {
            "chain_initializer": tf.keras.initializers.serialize(
                self.chain_initializer),
            "regularizer": self.regularizer,
            "use_boundary": self.use_boundary,
            "boundary_initializer": tf.keras.initializers.serialize(
                self.boundary_initializer
            ),
            "units": self.units,
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
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

        if self.use_boundary:
            self.left_boundary = self.add_weight(
                shape=(units,),
                name="left_boundary",
                initializer=self.boundary_initializer,
            )
            self.right_boundary = self.add_weight(
                shape=(units,),
                name="right_boundary",
                initializer=self.boundary_initializer,
            )
        if self.units is not None:
            self.dense = tf.keras.layers.Dense(units, dtype=self.dtype)
        else:
            self.dense = lambda x: tf.cast(x, dtype=self.dtype)

    def call(self, inputs, mask=None, training=False):
        potentials = self.dense(inputs)
        if mask is not None:
            if tf.keras.backend.ndim(mask) != 2:
                raise ValueError("Input mask to CRF must have dim 2 if not None")

        if mask is not None:
            left_boundary_mask = self._compute_mask_left_boundary(mask)
            first_mask = left_boundary_mask[:, 0]
            if first_mask is not None and tf.executing_eagerly():
                no_left_padding = tf.math.reduce_all(first_mask)
                left_padding = not no_left_padding
                if left_padding:
                    raise NotImplementedError(
                        "Currently, CRF layer do not support left padding"
                    )
        # appending boundary probability info
        if self.use_boundary:
            potentials = self.add_boundary_energy(
                potentials, mask, self.left_boundary, self.right_boundary
            )
        sequence_length = self._get_sequence_length(inputs, mask)
        viterbi_sequence, _ = tfa.text.crf_decode(
            potentials, self.transitions, sequence_length
        )
        return viterbi_sequence, potentials, sequence_length, self.transitions

    def _get_sequence_length(self, input_, mask):
        """Currently underline CRF fucntion (provided by
        tensorflow_addons.text.crf) do not support bi-direction masking (left
        padding / right padding), it support right padding by tell it the
        sequence length.
        this function is compute the sequence length from input and
        mask.
        """
        if mask is not None:
            sequence_length = self.mask_to_sequence_length(mask)
        else:
            # make a mask tensor from input, then used to generate sequence_length
            input_energy_shape = tf.shape(input_)
            raw_input_shape = tf.slice(input_energy_shape, [0], [2])
            alt_mask = tf.ones(raw_input_shape)

            sequence_length = self.mask_to_sequence_length(alt_mask)

        return sequence_length

    def mask_to_sequence_length(self, mask):
        """compute sequence length from mask."""
        sequence_length = tf.reduce_sum(tf.cast(mask, tf.int64), 1)
        return sequence_length

    @staticmethod
    def _compute_mask_right_boundary(mask):
        """input mask: 0011100, output right_boundary: 0000100."""
        # shift mask to left by 1: 0011100 => 0111000
        offset = 1
        left_shifted_mask = tf.concat(
            [mask[:, offset:], tf.zeros_like(mask[:, :offset])], axis=1
        )

        # NOTE: below code is different from keras_contrib
        # Original code in keras_contrib:
        # end_mask = K.cast(
        #   K.greater(self.shift_left(mask), mask),
        #   K.floatx()
        # )
        # has a bug, confirmed
        # by the original keras_contrib maintainer
        # Luiz Felix (github: lzfelix),

        # 0011100 > 0111000 => 0000100
        right_boundary = tf.math.greater(
            tf.cast(mask, tf.int32), tf.cast(left_shifted_mask, tf.int32)
        )

        return right_boundary

    @staticmethod
    def _compute_mask_left_boundary(mask):
        """input mask: 0011100, output left_boundary: 0010000."""
        # shift mask to right by 1: 0011100 => 0001110
        offset = 1
        right_shifted_mask = tf.concat(
            [tf.zeros_like(mask[:, :offset]), mask[:, :-offset]], axis=1
        )

        # 0011100 > 0001110 => 0010000
        left_boundary = tf.math.greater(
            tf.cast(mask, tf.int32), tf.cast(right_shifted_mask, tf.int32)
        )

        return left_boundary

    def add_boundary_energy(self, potentials, mask, start, end):
        def expand_scalar_to_3d(x):
            # expand tensor from shape (x, ) to (1, 1, x)
            return tf.reshape(x, (1, 1, -1))

        start = tf.cast(expand_scalar_to_3d(start), potentials.dtype)
        end = tf.cast(expand_scalar_to_3d(end), potentials.dtype)
        if mask is None:
            potentials = tf.concat(
                [potentials[:, :1, :] + start, potentials[:, 1:, :]], axis=1
            )
            potentials = tf.concat(
                [potentials[:, :-1, :], potentials[:, -1:, :] + end], axis=1
            )
        else:
            mask = tf.keras.backend.expand_dims(tf.cast(mask, start.dtype), axis=-1)
            start_mask = tf.cast(self._compute_mask_left_boundary(mask), start.dtype)

            end_mask = tf.cast(self._compute_mask_right_boundary(mask), end.dtype)
            potentials = potentials + start_mask * start
            potentials = potentials + end_mask * end
        return potentials

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:2]
        return output_shape

    def compute_mask(self, input_, mask=None):
        """keep mask shape [batch_size, max_seq_len]"""
        return mask