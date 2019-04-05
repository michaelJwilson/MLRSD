import  matplotlib;                matplotlib.use('Agg')
import  copy

from    keras.datasets     import  mnist
from    keras.utils        import  to_categorical
from    keras.models       import  Sequential
from    keras.layers       import  Dense, Conv2D, Flatten, Conv3D
from    keras.models       import  model_from_json
from    nbodykit.lab       import  LinearMesh, cosmology, BigFileMesh

import  matplotlib.pyplot  as      plt
import  pylab              as      pl
import  numpy              as      np


train   = True

print('\n\nWelcome.\n\n')

X       = np.zeros((10, 128, 128, 128))

for i, seed in enumerate(np.arange(42, 52, 1)):
  mesh       = BigFileMesh('fields/field_%d' % seed, 'Field')
  X[i,:,:,:] = mesh.preview()

##  128 x 128 x 128 pixels, with 60K items in train and 10K in validation.
X_train  = X.reshape(10, 128, 128, 128, 1)
X_test   = copy.copy(X_train)

_y_train = np.random.randint(10, size=10)
_y_test  = np.random.randint(10, size=10)

##  one-hot encode target column.
y_train  = to_categorical(_y_train)
y_test   = to_categorical(_y_test)

if train:
  model  = Sequential()

  model.add(Conv3D(64, kernel_size=3, activation='relu', input_shape=(128, 128, 128, 1)))
  model.add(Conv3D(32, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
                                                                                                                 
  model_json = model.to_json()

  with open('model.json', 'w') as json_file:
    json_file.write(model_json)

  model.save_weights('model.h5')

else:
  ofile  = open('model.json', 'r')
  ojson  = ofile.read()

  ofile.close()

  model  = model_from_json(ojson)
  model.load_weights('model.h5')
 
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = model.predict(X_test)
score       = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:',     score[0])
print('Test accuracy:', score[1])

print('\n\nDone.\n\n')
