import  matplotlib
import  os
import  json
import  copy

from    keras.datasets     import  mnist
from    keras.utils        import  to_categorical
from    keras.models       import  Sequential
from    keras.layers       import  Dense, Conv2D, Flatten, Conv3D, Dropout
from    keras.models       import  model_from_json
from    nbodykit.lab       import  LinearMesh, cosmology, BigFileMesh, BigFileCatalog

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

nruns   = 400
nvalid  =  75

X       = np.zeros((nruns, 128, 128, 128))

LOS     = [0,0,1]

for iid in np.arange(nruns):
  print('Loading %d' % iid)

  fpath             = '/global/cscratch1/sd/mjwilson/MLRSD/fastpm/fpm-%d-1.0000' % iid
  cat               = BigFileCatalog(fpath, dataset='1/', header='Header')

  rsd_factor        = cat.attrs['RSDFactor']
  cat['zPosition']  = cat['Position'] + rsd_factor * cat['Velocity'] * LOS

  mesh              = cat.to_mesh(Nmesh=128, BoxSize=1000., position='zPosition')
  field             = mesh.to_field(mode='complex')

  X[i,:,:,:]        = mesh.preview()

##  digitize in f.
fmin     = labels[:,2].min()
fmax     = labels[:,2].max()

bins     = np.linspace(fmin, fmax, 20)

##  128 x 128 x 128 pixels, with 60K items in train and 10K in validation.
X        = X.reshape(nruns, 128, 128, 128, 1)
X_train  = X[nvalid:, :, :, :, :]
X_test   = X[:nvalid, :, :, :, :]

_y_train = np.digitize(labels[:,2][nvalid:nruns], bins)
_y_test  = np.digitize(labels[:,2][:nvalid], bins)

##  One-hot encode target column.
y_train  = to_categorical(_y_train)
nhot     = len(y_train[0,:])

y_test   = to_categorical(_y_test, num_classes=nhot)

print(nhot)

if train:
  model  = Sequential()

  model.add(Conv3D(64, kernel_size=3, activation='relu', input_shape=(128, 128, 128, 1)))
  model.add(Dropout(0.3))
  model.add(Conv3D(32, kernel_size=3, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(nhot, activation='softmax'))  ##  nhot scores sum to one. 

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  history    = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=28)
  
  with open('history.json', 'w') as fp:
    json.dump(history.history, fp, sort_keys=True, indent=4)

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

##  Plot.
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('\n\nDone.\n\n')
