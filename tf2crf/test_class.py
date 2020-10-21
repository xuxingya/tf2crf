from tensorflow.keras.layers import Dense, Dropout
from transformers import TFBertModel
import tensorflow as tf
from tf2crf import CRF

tf.random.set_seed(200)


class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name='intent')
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dropout = Dropout(0.1)
        self.dense = Dense(9, activation='relu')
        self.crf = CRF(9)

    def call(self, inputs, **kwargs):
        sequence_out, pooled_output = self.bert(inputs, **kwargs)
        sequence_out = self.dropout(sequence_out)
        sequence_out = self.dense(sequence_out)
        output = self.crf(sequence_out)
        return output


def test():
    model = TestModel()
    model.compile(loss=model.crf.loss, optimizer='adam', metrics=[model.crf.accuracy])

    x = [[5, 2, 3] * 3] * 10
    y = [[1, 2, 3] * 3] * 10

    model.fit(x=x, y=y, epochs=20, batch_size=8)


if __name__ == '__main__':
    test()
