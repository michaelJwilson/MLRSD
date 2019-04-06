import  matplotlib
import  pickle
import  os
import  json
import  copy
import  tensorflow         as      tf

from    tensorflow.keras.utils               import  to_categorical
from    tensorflow.keras.models              import  Sequential
from    tensorflow.keras.layers              import  Dense, Conv2D, Flatten, Conv3D, Dropout, MaxPooling2D, BatchNormalization
from    tensorflow.keras.models              import  model_from_json
from    tensorflow.keras.preprocessing.image import ImageDataGenerator
from    nbodykit.lab                         import  LinearMesh, cosmology, BigFileMesh, BigFileCatalog

import  matplotlib.pyplot  as      plt
import  pylab              as      pl
import  numpy              as      np


def prep_model2DA(nhot):
  model = Sequential()

  ##  output = relu(dot(W, input) + b);  W (input_dimension, 16).
  ##  https://arxiv.org/pdf/1807.08732.pdf
  model.add(Conv2D(16, kernel_size=8, strides=2, padding='valid', activation='relu', input_shape=(128, 128, 1)))
  model.add(Dropout(0.1))
  model.add(Conv2D(32, kernel_size=4, strides=2, padding='valid', activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(128, kernel_size=4, strides=2, padding='valid', activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, kernel_size=4, strides=2, padding='valid', activation='relu'))
  ##  model.add(Dropout(0.1))
  model.add(Conv2D(256, kernel_size=4, strides=2, padding='valid', activation='relu', input_shape=(128, 128, 1)))  ##  16 output units. 
  ##  model.add(Dropout(0.1))

  model.add(Flatten())

  ##  nhot scores sum to one.                                                                                                                                                  
  model.add(Dense(1024, activation='relu'))
  model.add(Dense( 256, activation='relu'))

  model.add(Dense(nhot, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  return  model

def prep_model2DB(nhot):
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1))) 
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu')) 
  model.add(Dropout(0.4))  
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(nhot, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  return  model

def pprocess(X):
  axis          = np.random.randint(2)
  npix          = np.random.randint(128)
  sign          = -1. ** np.random.randint(2)

  return  sign * np.roll(X, npix, axis=axis)
  
##  module load python/3.6-anaconda-4.4                                                                                                         
##  source /usr/common/contrib/bccp/conda-activate.sh 3.6                                                                                    
##  module load tensorflow/intel-1.11.0-py36                                                                                              
##  export PYTHONPATH=$PYTHONPATH/usr/common/software/tensorflow/intel-tensorflow/1.11.0-py36/lib/python3.6/site-packages/                    
##  conda install tensorflow -c intel 

nseeds  = 10
nslice  = 10
train   = True

print('\n\nWelcome.\n\n')

cosmos  = np.loadtxt('cosmo.txt').tolist()
labels  = []

for x in cosmos:
 for i in np.arange(nseeds):
  for j in np.arange(nslice):
   labels.append(x)

nruns   = len(cosmos) * nseeds
labels  = np.array(labels)

##  Split in explicit batches.
nsplit  =  450
nvalid  =  150
ntimes  =  np.floor(nruns / nsplit)

LOS     = [0,0,1]

##  digitize in f.                                                                                                                                                               
fmin     = .25 ** 0.545
fmax     = .35 ** 0.545

nhot     = 10
bins     = np.linspace(fmin, fmax, nhot)

model    = prep_model2DA(nhot)

for split in np.tile(np.arange(ntimes), 4): 
  zero   = np.int(split) * nsplit  
  
  ##  128 x 128 x 128 pixels, with 60K items in train and 10K in validation. 
  X      = np.zeros((nsplit * nslice, 128, 128, 1))

  for iid in np.arange(nsplit):
    mid  = zero + iid

    print('Loading %d' % mid)

    fpath           = '/global/cscratch1/sd/mjwilson/MLRSD/fastpm/fpm-%d-1.0000' % mid

    mesh            = BigFileMesh(fpath, dataset='1/Field', mode='real', header='Header').preview()

    for sslice in np.arange(nslice):
      X[iid + 10 * sslice,:,:, 0] = mesh[:,sslice,:]
  
  X_train         = X[:, :, :, :]    

  train_gen       = ImageDataGenerator(featurewise_center=False,\
                                       rotation_range=0,\
                                       width_shift_range=0.,\
                                       height_shift_range=0.,\
                                       horizontal_flip=True,\
                                       vertical_flip=True,\
                                       rescale=1.,\
                                       preprocessing_function=pprocess,\
                                       validation_split=0.15)

  ##  Fit whitening params. 
  train_gen.fit(X_train)

  _labels         = labels[zero * nslice: (zero + nsplit) * nslice]
  _y_train        = np.digitize(_labels[:,2], bins) 

  ##  One-hot encode target column.
  y_train         = to_categorical(_y_train, num_classes=nhot)

  ##  validation_data=(X_test, y_test)
  ##  history     = model.fit(X_train, y_train, epochs=20, batch_size=32, shuffle=True, validation_split=0.15)

  history         = model.fit_generator(train_gen.flow(X_train, y_train, batch_size=32, shuffle=True),\
                                        steps_per_epoch=10 * len(X_train) / 32, epochs=4)

  history         = history.history
  
  '''
  pickle.dump(history, open('history/history_%d.p' % zero, 'wb'))

  model_json = model.to_json()

  with open('model/model_%d.json' % zero, 'w') as json_file:
    json_file.write(model_json)

  ##  model.save_weights('model.h5')
  '''

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
'''
predictions = model.predict(X_test)
score       = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:',     score[0])
print('Test accuracy:', score[1])
'''
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
