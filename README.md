# tf2crf
* a simple CRF layer for tensorflow 2 keras
* support keras masking

## Install
```python
$ pip install tf2crf
```
## Tips
tensorflow >= 2.1.0
Recommmend use the latest tensorflow-addons which is compatiable with your tf version.
## Example
```python
from tf2CRF import CRF
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
output = Dense(9, activation=None)(output)
crf = CRF(dtype='float32')
output = crf(output)
model = Model(inputs, output)
model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
model.save('model')

```

## Supoort for tensorflow mixed precision training
Currently these is a bug in tensorflow-addons.text.crf, which causes a dtype error when using miex precision. This bug has been fixed in master branch, but is not released. so if you want to use mixed precision training. You need to **pip install tfa-nighly** instead.
## Example
```python
from tf2CRF import CRF
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
output = Dense(9, activation=None)(output)
crf = CRF(dtype='float32')
output = crf(output)
model = Model(inputs, output)
model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
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