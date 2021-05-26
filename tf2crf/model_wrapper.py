from typing import Union

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
    """
    Wrapper around the base model for custom training logic.
    Args:
        base_model: The model including the CRF layer
        sparse_target: if the y label is sparse or one-hot, default True
        metric: the metric for training, default 'accuracy'. Warning: Currently tensorflow metrics like AUC need the output and y_true to be one-hot to cauculate, they are not supported.
    """

    def __init__(self, base_model, sparse_target=True, metric: Union[str, object] = 'accuracy'):
        super().__init__()
        self.base_model = base_model
        self.sparse_target = sparse_target
        self.metric = metric
        if isinstance(metric, str):
            if metric == 'accuracy':
                self.metrics_fn = tf.keras.metrics.Accuracy(name='accuracy')
            else:
                raise ValueError('unknown metric name')
        else:
            self.metrics_fn = self.metric
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def call(self, inputs, training=False):
        if training:
            return self.base_model(inputs)
        else:
            return self.base_model(inputs)[0]

    def compute_loss(self, x, y, training=False):
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(x, training=training)
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
            loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss": self.loss_tracker.result(), self.metrics_fn.name: self.metrics_fn.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metrics_fn]

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
        loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss_val": self.loss_tracker.result(), f'val_{self.metrics_fn.name}': self.metrics_fn.result()}


class ModelWithCRFLossDSCLoss(tf.keras.Model):
    """
        Wrapper around the base model for custom training logic. And DSC loss to help improve the performance of NER task.
        Args:
            base_model: The model including the CRF layer
            sparse_target: if the y label is sparse or one-hot, default True
            metric: the metric for training, default 'accuracy'. Warning: Currently tensorflow metrics like AUC need the output and y_true to be one-hot to cauculate, they are not supported.
            alpha: parameter for DSC loss
        """

    def __init__(self, base_model, sparse_target=True, metric: Union[str, object] = 'accuracy', alpha=0.6):
        super().__init__()
        self.base_model = base_model
        self.sparse_target = sparse_target
        self.metric = metric
        if isinstance(metric, str):
            if metric == 'accuracy':
                self.metrics_fn = tf.keras.metrics.Accuracy(name='accuracy')
            else:
                raise ValueError('unknown metric name')
        else:
            self.metrics_fn = self.metric
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.alpha = alpha

    def call(self, inputs, training=False):
        if training:
            return self.base_model(inputs)
        else:
            return self.base_model(inputs)[0]

    def compute_loss(self, x, y, sample_weight, training=False):
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(x, training=training)
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        ds_loss = compute_dsc_loss(potentials, y, self.alpha)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight[0]
            ds_loss = ds_loss * sample_weight[1]
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss), tf.reduce_mean(ds_loss)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)

        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(x, y, sample_weight, training=True)
            loss = crf_loss + ds_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss": self.loss_tracker.result(), self.metrics_fn.name: self.metrics_fn.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metrics_fn]

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(x, y, sample_weight, training=True)
        loss = crf_loss + ds_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss_val": self.loss_tracker.result(), f'val_{self.metrics_fn.name}': self.metrics_fn.result()}
