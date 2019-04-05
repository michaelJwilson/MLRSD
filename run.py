import  matplotlib;                matplotlib.use('Agg')
import  os
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


nseeds  = 10 
train   = True

print('\n\nWelcome.\n\n')

cosmos  = np.loadtxt('cosmo.txt').tolist()
labels  = []

for x in cosmos:
  for i in np.arange(nseeds):
    labels.append(x)

nruns   = len(labels)
labels  = np.array(labels)

nruns   = 100
X       = np.zeros((nruns, 128, 128, 128))

for iid in np.arange(nruns):
  print('Loading %d' % iid)

  root       = os.environ['CSCRATCH']
  mesh       = BigFileMesh(root + '/MLRSD/fields/field_%d' % iid, 'Field')
  X[i,:,:,:] = mesh.preview()

##  digitize in f.
bins     = np.arange(0.4, 0.6, 1.e-3)

##  128 x 128 x 128 pixels, with 60K items in train and 10K in validation.
X_train  = X.reshape(nruns, 128, 128, 128, 1)
X_test   = copy.copy(X_train)

_y_train = np.digitize(labels[:,2][:nruns], bins)
_y_test  = np.digitize(labels[:,2][:nruns], bins)

##  One-hot encode target column.
y_train  = to_categorical(_y_train)
y_test   = to_categorical(_y_test)

nhot     = len(y_test[0,:])

print(nhot)

if train:
  model  = Sequential()

  model.add(Conv3D(64, kernel_size=8, activation='relu', input_shape=(128, 128, 128, 1)))
  model.add(Conv3D(32, kernel_size=8, activation='relu'))
  model.add(Flatten())
  model.add(Dense(nhot, activation='softmax'))

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
