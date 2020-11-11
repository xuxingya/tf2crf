# tf2crf
* a simple CRF layer for tensorflow 2 keras
* support keras masking

## Install
```python
$ pip install tf2crf
```

## Features
* easy to use CRF layer with tensorflow
* now support the ModelWithCRFLossDSCLoss with DSC loss, which increases f1 score with unbalanced data (refer the paper [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf))
## Attention
* A dense layer is needed before the CRF layer to convert inputs to shape (batch_size, timesteps, num_classes).
The 'num_class' is how many tags or catogories the model predicts.
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
output = Dense(9, activation=None)(output)
crf = CRF(dtype='float32')
output = crf(output)
base_model = Model(inputs, output)
model = ModelWithCRFLoss(base_model)
model.compile(optimizer='adam')

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
tf.keras.models.save_model(model, 'test/1')

```

## Supoort for tensorflow mixed precision training
Currently these is a bug in tensorflow-addons.text.crf, which causes a dtype error when using miex precision. To correctly use mixed precison, you need to modify the line 488 of tensorflow_addons/text/crf.py to:
```python
crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params, dtype=inputs.dtype)
```
## Example
```python
import tensorflow as tf
from tf2CRF import CRF
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
from tf2crf import CRF, ModelWithCRFLoss
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
output = Dense(9, activation=None)(output)
crf = CRF(dtype='float32')
output = crf(output)
base_model = Model(inputs, output)
model = ModelWithCRFLoss(base_model)
model.compile(optimizer='adam')

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
tf.keras.models.save_model(model, 'test/1')
```
## How to save the model
Currently, Loading the model directly is not supported. So you need to load the model weights instead.
For example:
```python
tf.keras.models.save_model(model, '1')
model.load_weights('1/variables/variables')
```
or
```
model.save_weights('model.h5')
model.load_weights('model.h5')
```