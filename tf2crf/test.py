from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tf2crf import CRF
import numpy as np

tf.random.set_seed(200)


def test_model():
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
    output = Bidirectional(GRU(64, return_sequences=True))(output)
    output = Dense(9, activation=None)(output)
    crf = CRF(dtype='float32', name='crf')
    output = crf(output)
    model = Model(inputs, output)
    model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])
    return model


def train():
    model = test_model()
    x = np.array([[5, 2, 3] * 3] * 100)
    y = np.array([[1, 2, 3] * 3] * 100)
    model.fit(x=x, y=y, epochs=10, batch_size=4, validation_split=0.1)
    tf.keras.models.save_model(model, 'test/1')
    # model.save_weights('model.h5')


def load_model():
    model = test_model()
    model.load_weights('test/1/variables/variables')
    # model.load_weights('model.h5')
    model.summary()


if __name__ == '__main__':
    train()
    load_model()
