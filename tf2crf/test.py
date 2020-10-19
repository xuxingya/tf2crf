from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tf2crf import CRF

tf.random.set_seed(200)


def test():
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(100, 40, trainable=True, mask_zero=False)(inputs)
    output = Bidirectional(GRU(64, return_sequences=True))(output)
    output = Dense(9, activation=None)(output)
    crf = CRF(dtype='float32')
    output = crf(output)
    model = Model(inputs, output)
    model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])

    x = [[5, 2, 3] * 3] * 10
    y = [[1, 2, 3] * 3] * 10

    model.fit(x=x, y=y, epochs=10, batch_size=4)
    model.save('model')


if __name__ == '__main__':
    test()
