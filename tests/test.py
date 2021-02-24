import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model

from tf2crf import CRF, ModelWithCRFLoss, ModelWithCRFLossDSCLoss


def shape_list(x):
    """
    copied from transformers.modeling_tf_utils
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class TFTester(unittest.TestCase):

    def test_crf_no_kernel(self):
        x = np.array([np.random.uniform(0, 1, (3, 9))] * 100)
        crf = CRF()
        result = crf(x)
        assert shape_list(result[0]) == [100, 3]
        assert shape_list(result[1]) == [100, 3, 9]
        assert shape_list(result[2]) == [100]
        assert shape_list(result[3]) == [9, 9]

    def test_crf_kernel(self):
        x = np.array([np.random.uniform(0, 1, (3, 128))] * 100)
        crf = CRF(units=9)
        result = crf(x)
        assert shape_list(result[0]) == [100, 3]
        assert shape_list(result[1]) == [100, 3, 9]
        assert shape_list(result[2]) == [100]
        assert shape_list(result[3]) == [9, 9]

    def test_model_training(self):
        x = np.array([[5, 2, 3] * 3] * 100)
        y = np.array([[1, 2, 3] * 3] * 100)
        model = get_model(9)
        model.fit(x=x, y=y, epochs=5, batch_size=4, validation_split=0.1, verbose=0)

    def test_model_inference(self):
        model = get_model(9)
        model.predict([[5, 2, 3] * 3])

    def test_save_model(self):
        model = get_model(9)
        model.predict([[5, 2, 3] * 3])
        model.save('tests/1')
        model.save_weights('tests/model.h5')

    def test_mixed_precison(self):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        x = np.array([[5, 2, 3] * 3] * 100)
        y = np.array([[1, 2, 3] * 3] * 100)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        model = get_model(9)
        model.fit(x=x, y=y, epochs=5, batch_size=4, validation_split=0.1, verbose=0)

    def test_model_dsc(self):
        x = np.array([[5, 2, 3] * 3] * 100)
        y = np.array([[1, 2, 3] * 3] * 100)
        base_model = get_base_model(9)
        model = ModelWithCRFLossDSCLoss(base_model)
        model.compile(optimizer='adam')
        model.fit(x=x, y=y, epochs=5, batch_size=4, validation_split=0.1, verbose=0)

    def test_mixed_precison_dsc(self):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        x = np.array([[5, 2, 3] * 3] * 100)
        y = np.array([[1, 2, 3] * 3] * 100)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        base_model = get_base_model(9)
        model = ModelWithCRFLossDSCLoss(base_model)
        model.compile(optimizer='adam')
        model.fit(x=x, y=y, epochs=5, batch_size=4, validation_split=0.1, verbose=0)


def get_model(units: int):
    base_model = get_base_model(units)
    model = ModelWithCRFLoss(base_model)
    model.compile(optimizer='adam')
    return model


def get_base_model(units: int):
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(10, 20, trainable=True, mask_zero=True)(inputs)
    crf = CRF(units=units, dtype='float32', name='crf')
    output = crf(output)
    base_model = Model(inputs=inputs, outputs=output)
    return base_model


if __name__ == "__main__":
    unittest.main()
