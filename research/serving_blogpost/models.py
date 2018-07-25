from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers


def GRU_stack(num_input_feats):
  model = Sequential()
  model.add(layers.GRU(32,
                       dropout=0.1,
                       recurrent_dropout=0.5,
                       return_sequences=True,
                       input_shape=(None, num_input_feats)))
  model.add(layers.GRU(64, activation='relu',
                       dropout=0.1,
                       recurrent_dropout=0.5))
  model.add(layers.Dense(1))
  model.compile(optimizer=optimizers.RMSprop(), loss='mae')
  return model


