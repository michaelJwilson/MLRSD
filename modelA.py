from    tensorflow.keras.models               import  Sequential
from    tensorflow.keras.layers               import  Dense, SeparableConv2D, Conv2D, Flatten, Conv3D
from    tensorflow.keras.layers               import  Dropout, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU


def prep_model_2D_A(nhot, nmesh=128):
  ##  Loosely based on:  https://arxiv.org/pdf/1807.08732.pdf                                                                                                                                                      
  model = Sequential()

  ##  layer output = relu(dot(W, input) + b);  E.g.  W (input_dimension, 16).
  model.add(Conv2D(16, kernel_size=8, strides=2, padding='valid', activation='relu', input_shape=(nmesh, nmesh, 1)))
  model.add(Dropout(0.1))
  model.add(Conv2D(32, kernel_size=4, strides=2, padding='valid', activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(nmesh, kernel_size=4, strides=2, padding='valid', activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, kernel_size=4, strides=2, padding='valid', activation='relu'))
  ##  model.add(Dropout(0.1))
  model.add(Conv2D(256, kernel_size=4, strides=2, padding='valid', activation='relu', input_shape=(nmesh, nmesh, 1)))  ##  16 output units.                                                                      
  ##  model.add(Dropout(0.1))                                                                                                                                                                                    
  model.add(Flatten())

  ##  nhot scores sum to one.                                                                                                                                                                                    
  model.add(Dense(1024, activation='relu'))
  model.add(Dense( 256, activation='relu'))

  model.add(Dense(nhot, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  return  model
