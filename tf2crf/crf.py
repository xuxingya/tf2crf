import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        num_labels (int): the number of labels to tag each temporal input.

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

    def __init__(self, sparse_target=True, **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        self.sparse_target = sparse_target
        self.sequence_lengths = None
        self.mask = None

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "transitions": K.eval(self.transitions),
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        output_dim = input_shape[-1]
        assert len(input_shape) == 3
        self.transitions = self.add_weight(
            name="transitions",
            shape=[output_dim, output_dim],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            self.sequence_lengths = K.sum(K.cast(mask, 'int64'), axis=-1)
            self.mask = mask
        else:
            self.sequence_lengths = K.sum(K.ones_like(inputs[:,:,0], dtype='int64'), axis=-1)
        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, self.sequence_lengths
        )
        output = K.one_hot(viterbi_sequence, inputs.shape[-1])
        return K.in_train_phase(inputs, output)

    def loss(self, y_true, y_pred):
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)
        log_likelihood, self.transitions = tfa.text.crf_log_likelihood(
            y_pred,
            y_true,
            self.sequence_lengths,
            transition_params=self.transitions,
        )
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.out_dim, )

    def accuracy(self, y_true, y_pred):
        mask = self.mask
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)

        y_pred, _ = tfa.text.crf_decode(
            y_pred, self.transitions, self.sequence_lengths
        )
        y_true = K.cast(y_true, y_pred.dtype)
        # 逐标签取最大来粗略评测训练效果
        is_equal = K.equal(y_true, y_pred)
        is_equal = K.cast(is_equal, K.floatx())
        if mask is None:
            return K.mean(is_equal)
        else:
            mask = K.cast(mask, K.floatx())
            return K.sum(is_equal * mask) / K.sum(mask)