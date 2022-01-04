from    tensorflow.keras.models               import  Sequential
from    tensorflow.keras.layers               import  Dense, SeparableConv2D, Conv2D, Flatten, Conv3D
from    tensorflow.keras.layers               import  Dropout, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU


def prep_model_2D_C(nhot, optimizer, regress=True, loss='mae', nmesh=128):
  print("\n\nPreparing model.\n\n")

  _input = (nmesh, nmesh, 1)

  model  = Sequential()
  model.add(Flatten(input_shape=_input))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1,  activation='sigmoid'))

  if regress:                                                                                                                                                                                                    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  else:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  return  model
