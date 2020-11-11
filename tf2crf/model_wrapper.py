import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood
from .other_losses import compute_dsc_loss


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model, use_dsc=False):
        super().__init__()
        self.base_model = base_model
        self.accuracy_fn = tf.keras.metrics.Accuracy(name='accuracy')

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss)

    def accuracy(self, y_true, y_pred):
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        sample_weights = tf.sequence_mask(sequence_length, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_sequence, sample_weights)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss = self.compute_loss(
                x, y, sample_weight, training=True
            )
        gradients = tape.gradient(crf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))

        return {"crf_loss": crf_loss, 'accuracy': self.accuracy_fn.result()}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, sample_weight)
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"crf_loss_val": crf_loss, 'val_accuracy': self.accuracy_fn.result()}


class ModelWithCRFLossDSCLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model, alpha=0.6):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.accuracy_fn = tf.keras.metrics.Accuracy(name='accuracy')

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        ds_loss = compute_dsc_loss(potentials, y, self.alpha)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight[0]
            ds_loss = ds_loss * sample_weight[1]
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss), tf.reduce_mean(ds_loss)

    def accuracy(self, y_true, y_pred):
        viterbi_sequence, potentials, sequence_length, chain_kernel = y_pred
        mask = tf.sequence_mask(sequence_length, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_sequence, mask)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + ds_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))

        return {"crf_loss": crf_loss, "dsc_loss": ds_loss, 'accuracy': self.accuracy_fn.result()}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(x, y, sample_weight)
        self.accuracy_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"crf_loss_val": crf_loss, "dsc_loss_val": ds_loss, 'val_accuracy': self.accuracy_fn.result()}
