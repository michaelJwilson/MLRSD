import  matplotlib
import  pickle
import  os
import  json
import  copy
import  tensorflow         as      tf

from    tensorflow.keras.utils     import  to_categorical
from    tensorflow.keras.models    import  Sequential
from    tensorflow.keras.layers    import  Dense, Conv2D, Flatten, Conv3D, Dropout
from    tensorflow.keras.models    import  model_from_json
from    nbodykit.lab               import  LinearMesh, cosmology, BigFileMesh, BigFileCatalog

import  matplotlib.pyplot  as      plt
import  pylab              as      pl
import  numpy              as      np


##  module load python/3.6-anaconda-4.4 
##  source /usr/common/contrib/bccp/conda-activate.sh 3.6
##  module load tensorflow/intel-1.11.0-py36
##  export PYTHONPATH=$PYTHONPATH/usr/common/software/tensorflow/intel-tensorflow/1.11.0-py36/lib/python3.6/site-packages/
##  conda install tensorflow -c intel

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

##  Split in batches of 50.
nsplit  =  50
nvalid  =  20
ntimes  =  np.floor(nruns / nsplit)

LOS     = [0,0,1]

##  digitize in f.                                                                                                                                                               
fmin     = .25 ** 0.545
fmax     = .35 ** 0.545

nhot     = 20
bins     = np.linspace(fmin, fmax, 20)

model    = Sequential()
model.add(Conv3D(64, kernel_size=3, activation='relu', input_shape=(128, 128, 128, 1)))
model.add(Dropout(0.1))
model.add(Conv3D(64, kernel_size=3, activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())

##  nhot scores sum to one.
model.add(Dense(nhot, activation='softmax'))                                                                                                                                  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for ii, split in enumerate(np.arange(ntimes)): 
  zero   = ii * nsplit  
  
  ##  128 x 128 x 128 pixels, with 60K items in train and 10K in validation. 
  X      = np.zeros((nsplit, 128, 128, 128, 1))

  for iid in np.arange(nsplit):
    mid  = zero + iid

    print('Loading %d' % mid)

    fpath           = '/global/cscratch1/sd/mjwilson/MLRSD/fastpm/fpm-%d-1.0000' % mid

    mesh            = BigFileMesh(fpath, dataset='1/Field', mode='real', header='Header')
    X[iid,:,:,:, :] = mesh.preview()
   
  X_train         = X[nvalid:, :, :, :, :]
  X_test          = X[:nvalid, :, :, :, :]
  
  _labels         = labels[zero : zero + nsplit]

  _y_train        = np.digitize(_labels[:,2][nvalid:], bins)
  _y_test         = np.digitize(_labels[:,2][:nvalid], bins)

  ##  One-hot encode target column.
  y_train         = to_categorical(_y_train, num_classes=nhot)
  y_test          = to_categorical(_y_test,  num_classes=nhot)

  history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=16)
  history = history.history

  pickle.dump(history, open('history/history_%d.p' % zero, 'wb'))

  model_json = model.to_json()

  with open('model/model_%d.json' % zero, 'w') as json_file:
    json_file.write(model_json)

  ##  model.save_weights('model.h5')

'''
else:
  history = pickle.load(open('history.p', 'rb'))

  ##
  ofile  = open('model.json', 'r')
  ojson  = ofile.read()

  ofile.close()

  ## 
  model  = model_from_json(ojson)
  model.load_weights('model.h5')
 
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

predictions = model.predict(X_test)
score       = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:',     score[0])
print('Test accuracy:', score[1])

##  Plot.
for stat in ['loss', 'acc']:
  tstat  = history[stat]
  vstat  = history['val_' + stat]

  epochs = range(1, 1 + len(tstat))

  plt.plot(epochs, tstat, 'bo', label='Training')
  plt.plot(epochs, vstat, 'b',  label='Validation')
  plt.xlabel('EPOCHS')
  plt.ylabel(stat.upper())
  plt.legend()
  plt.show()

print('\n\nDone.\n\n')
