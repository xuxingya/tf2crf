# tf2crf
* a simple CRF layer for tensorflow 2 keras
* support keras masking

## Install
```python
$ pip install tf2crf
```
## Tips
It has been tested under tensorflow 2.1.0 and tensorflow-nightly.
## Example
```python
from tf2CRF import CRF
from tensorflow.keras import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(len(vocab), dim, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
output = Dense(len(class_num), activation=None)(output)
crf = CRF()
output = crf(output)
model = Model(inputs, output)
model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])

```