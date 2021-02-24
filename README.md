# tf2crf
* a simple CRF layer for tensorflow 2 keras
* support keras masking

## Install
```python
$ pip install tf2crf
```

## Features
* easy to use CRF layer with tensorflow
* support mixed precision training
* support the ModelWithCRFLossDSCLoss with DSC loss, which increases f1 score with unbalanced data (refer the paper [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf))
## Attention
* Add internal kernel like CRF in keras_contrib, so now there is no need to stack a Dense layer before the CRF layer.
* I have changed the previous way that putting loss function and accuracy function in the CRF layer. Instead I choose to use ModelWappers (refered to jaspersjsun), which is more clean and flexible.
## Tips
tensorflow >= 2.1.0
Recommmend use the latest tensorflow-addons which is compatiable with your tf version.

## Example
```python
import tensorflow as tf
from tf2CRF import CRF
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
from tf2crf import CRF, ModelWithCRFLoss

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
crf = CRF(units=9, type='float32')
output = crf(output)
base_model = Model(inputs, output)
model = ModelWithCRFLoss(base_model, sparse_target=True)
model.compile(optimizer='adam')

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
model.save('tests/1')

```

