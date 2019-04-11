import  os
import  json

import  pickle
import  matplotlib
import  tensorflow                            as      tf
import  matplotlib.pyplot                     as      plt
import  pylab                                 as      pl
import  numpy                                 as      np

from    tensorflow.keras.utils                import  to_categorical
from    tensorflow.keras.models               import  Sequential
from    tensorflow.keras.layers               import  Dense, SeparableConv2D, Conv2D, Flatten, Conv3D
from    tensorflow.keras.layers               import  Dropout, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU
from    tensorflow.keras.models               import  model_from_json
from    tensorflow.keras.optimizers           import  Adam, Nadam
from    tensorflow.keras.preprocessing.image  import  ImageDataGenerator
from    nbodykit.lab                          import  BigFileMesh
from    sklearn.model_selection               import  train_test_split


np.random.seed(seed=314)

##  Define optimizers. 
adam    =  Adam(lr=1.e-4, amsgrad=True, clipnorm=1.)
nadam   = Nadam(lr=1.e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=1.)

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

def prep_model_2D_B(nhot=1, optimizer=adam, regress=True, loss='mae', nmesh=128):
  print("\n\nPreparing model.\n\n")

  _input = (nmesh, nmesh, 1)
  
  model  = Sequential()
  model.add(SeparableConv2D(32, (3, 3), activation='linear', input_shape=(128, 128, 1))) 
  model.add(LeakyReLU(alpha=0.03))
  model.add(MaxPooling2D((2, 2)))
  model.add(SeparableConv2D(64, (3, 3), activation='linear')) 
  model.add(LeakyReLU(alpha=0.03))
  model.add(MaxPooling2D((2, 2)))
  model.add(SeparableConv2D(64, (3, 3), activation='linear'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())   ##  Batch renormalization should come late.  
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.1))

  if regress:
    ##  Appropriate [0, 1] output. 
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
  else:
    model.add(Dense(nhot, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  
  model.summary()
  
  return  model

def prep_model_2D_C(nhot=1, optimizer=adam, regress=True, loss='mae', nmesh=128):
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

def pprocess(X, nmesh=128):
  ##  Pre-process each 2D n-body slice:
  ##  --  Roll along random axis by random int npix.
  ##  --  Random sign (should be applied to delta).
  ##  --  To be added:  Poisson sample. 
  ##  --  To be added:  Rescale amplitude by an arbitrary growth factor. 
  
  axis = np.random.randint(2)
  npix = np.random.randint(nmesh)

  ##  Random assignment of delta -> -delta. 
  sign = -1. ** np.random.randint(2)

  return  sign * np.roll(X, npix, axis=axis)

def get_cosmos(nruns, nseeds, nslice):
  ##  Get list of available cosmologies.                                                                                                                                           
  cosmos = np.loadtxt('cosmo.txt').tolist()
  labels = []

  ##  Generate labels where each cosmology is matched to nslice slices of nseeds sims. 
  for x in cosmos:
    for i in np.arange(nseeds):
      for j in np.arange(nslice):
        labels.append(x)

  labels = np.array(labels)
  nruns  = np.minimum(nruns, len(labels) * nseeds)

  return  labels

def plot(y, yhat, save=False):
  pl.clf()
  
  fmin  = .25 ** 0.545
  fmax  = .35 ** 0.545

  fs    = np.linspace(fmin, fmax, 10)

  pl.plot(fs,  fs, 'k-',   alpha=0.6)
  pl.plot(y, yhat, 'o', markersize=2)

  pl.xlabel(r'$f$')
  pl.ylabel(r'$\hat f$')

  pl.savefig('mlf.pdf')
  os.system('xpdf mlf.pdf')

    
if __name__ == '__main__':
  print('\n\nWelcome to MLRSD-2D.\n\n')

  train, regress = True, True

  optimizer      =  adam
  
  nseeds         =     9     ##  Number of random (seed) sims available in each cosmology. 
  nslice         =    16     ##  Sliced 3D sim. into _nslice_ (x, z) slices.  
  nhot           =    10     ##  Supervised:  label sims by bin index in f;  nhot == # of bins. 
  nmesh          =   128
  nruns          =   900     ##  Limit the total number of mocks input;  Set to e.g. 1e99 for all. 
  nsplit         =   900     ##  Split loading, storing and learning of mocks into batches of size nsplit. 
  ntile          =     1     ##  Number of load, train epochs through the data.      
  epochs         =   500     ##  Number of actual (keras) epochs. 
  valid_frac     =  0.15     ##  Split data X into fractions for train, validate/test.    

  jump           =  np.floor(nmesh / nslice).astype(np.int)
  ntimes         =  np.floor(nruns / nsplit)                  ##  Number of splits required.      
  
  model          =  prep_model_2D_C(None, optimizer=optimizer, regress=regress)  
  labels         =  get_cosmos(nruns, nseeds, nslice)

  ##  Load sims and train in explicit batches.
  ##  Note:  **  Tile ** ntile times to go through sims (rank ordered in cosmology).
  ##         Currently, only a subset of f values are input with each tile.         
  for split in np.tile(np.arange(ntimes), ntile): 
    ##  Zero index for this split. 
    zero  = np.int(split) * nsplit  

    ##  Set y labels.                                                                                                                                                                                                                                                                                                
    y     = labels[zero * nslice: (zero + nsplit) * nslice, 2]
    
    ##  E.g.  128 x 128 pixels, for number of mocks in split x number of slices taken from 3D sim. 
    X     = np.zeros((nsplit * nslice, nmesh, nmesh, 1))
    
    ##  Loop over mocks in split. 
    for iid in np.arange(nsplit):
      ##  Load fastpm sim. 
      fpath           = os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000' % (zero + iid)
      _file           = BigFileMesh(fpath, dataset='1/Field', mode='real', header='Header')

      ##  attrs: {'Om0', 'Ode0', 'h', 'shotnoise'}.
      attrs           = _file.attrs
      mesh            = _file.preview()

      print('Loading %d (Om = %.3lf, f = %.3lf, h = %.3lf) with label %.3lf.' % (zero + iid, attrs['Om0'], attrs['Om0']**0.545, attrs['h'], y[iid * nslice]))
      
      for ii, sslice in enumerate(np.arange(0, nmesh, jump)):
        ##  Split 3D sim into _nslice_ 2D (x, z) slices;  Mesh returns (1 + delta). 
        X[iid + nsplit * ii, :, :, 0] = mesh[:, sslice, :] - 1.0
        
    if not regress:
      ##  Bin sims in f and use bin index as a supervised label.                                                                                                                                                                                                                                                       
      fmin  = .25 ** 0.545
      fmax  = .35 ** 0.545

      ##  Number of one-hot encodings == number of bins.                                                                                                                                                                                                                                                               
      bins  = np.linspace(fmin, fmax, nhot)

      y     = np.digitize(y, bins)

      ##  One-hot encode target column.                                                                                                                                                                                                                                                                               \
      y     = to_categorical(y, num_classes=nhot)
        
    if train:
      '''
      ##  Note:  horizontal and vertical flipping of 2D slices.  
      train_gen = ImageDataGenerator(featurewise_center=False,\
                                     rotation_range=0,\
                                     width_shift_range=0.,\
                                     height_shift_range=0.,\
                                     horizontal_flip=True,\
                                     vertical_flip=True,\
                                     rescale=1.,\
                                     preprocessing_function=pprocess,\
                                     validation_split=valid_frac)  ##  Last Validation split. 

      ##  Fit whitening params. 
      train_gen.fit(X)
      
      ##  Image generator for continous cretion with pre-processing;  steps_per_epoch=10 * len(X_train) / 32.
      history = model.fit_generator(train_gen.flow(X, y, batch_size=256, shuffle=True),\
                                    steps_per_epoch=8000, epochs=epochs, use_multiprocessing=True)
      '''

      history = model.fit(X, y, validation_split=0.3, epochs=epochs)

      ##  Get current predictions.                                                                                                                                                                                                                                                                                                                
      yhat = model.predict(X)

      ##  Plot current prediction against truth (regression or supervised).                                                                                                                                                                                                                                                                       
      plot(y, yhat)
    
      '''
      history         = history.history
      pickle.dump(history, open('history/history_%d.p' % zero, 'wb'))

      model_json      = model.to_json()

      with open('model/model_%d.json' % zero, 'w') as json_file:
        json_file.write(model_json)
      
      ##  model.save_weights('model.h5')
      '''
      
    else:
      history = pickle.load(open('history/history_%d.p' % zero, 'rb'))
    
      ##
      ofile   = open('model/model_%d.json' % zero, 'r')
      ojson   = ofile.read()

      ofile.close()

      ##  model  = model_from_json(ojson)
      ##  model.load_weights('model.h5')

      if regress:
         model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        
      else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    '''
    ##  Test the predictions;  
    X_train, X_test, y_train, y_test = train_test_split(X[::-1, :, : ,:], y[::-1], test_size=valid_frac)
    
    predictions  =   model.predict(X_test)
    score        =  model.evaluate(X_test, y_test, verbose=0)

    print('Test loss:',     score[0])
    print('Test accuracy:', score[1])

    for stat in history.keys():
      print(stat)

      tstat  = history[stat]
      epochs = range(1, 1 + len(tstat))

      plt.plot(epochs, tstat, 'bo', label='Training')
      plt.xlabel('EPOCHS')
      plt.ylabel(stat.upper())
      plt.legend()
      plt.show()
    '''

print('\n\nDone.\n\n')
