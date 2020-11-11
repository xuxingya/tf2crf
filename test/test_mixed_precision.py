from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
from tf2crf import CRF, ModelWithCRFLoss
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def test():
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
    output = Bidirectional(GRU(64, return_sequences=True))(output)
    output = Dense(9, activation=None)(output)
    crf = CRF(dtype='float32')
    output = crf(output)
    base_model = Model(inputs, output)
    model = ModelWithCRFLoss(base_model)
    model.compile(optimizer='adam')

    x = np.array([[5, 2, 3] * 3] * 100)
    y = np.array([[1, 2, 3] * 3] * 100)

    model.fit(x=x, y=y, epochs=10, batch_size=4, validation_split=0.1)
    model.save('model')


if __name__ == '__main__':
    test()
