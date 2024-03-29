import  os
import  json
<<<<<<< HEAD
import  copy
import  tensorflow         as      tf
import  pylab              as      pl
import  numpy              as      np
import  matplotlib.pyplot  as      plt
=======
>>>>>>> 69c5653ba7f14eaed768b44b1fb4b3749d64da10

import  pickle
import  matplotlib
import  tensorflow                            as      tf
import  matplotlib.pyplot                     as      plt
import  pylab                                 as      pl
import  numpy                                 as      np

from    tensorflow.keras.models               import  model_from_json
from    tensorflow.keras.optimizers           import  Adam, Nadam
from    tensorflow.keras.preprocessing.image  import  ImageDataGenerator
<<<<<<< HEAD
from    nbodykit.lab                          import  LinearMesh, cosmology, BigFileMesh, BigFileCatalog
from    sklearn.model_selection               import  train_test_split


##  Define optimizers. 
adam  = Adam(lr=1.e-4, amsgrad=True, clipnorm=1.)
nadam = Nadam(lr=1.e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=1.)

def prep_model_2D_A(nhot):
  ##  Loosely based on:
  ##  https://arxiv.org/pdf/1807.08732.pdf

  model = Sequential()
=======
from    nbodykit.lab                          import  BigFileMesh
from    sklearn.model_selection               import  train_test_split
from    modelC                                import  prep_model_2D_C
from    preprocess                            import  pprocess
from    generator                             import  generator
>>>>>>> 69c5653ba7f14eaed768b44b1fb4b3749d64da10


np.random.seed(seed=314)

##  --  optimizers -- 
adam  =   Adam(lr=1.e-4, amsgrad=True, clipnorm=1.)
nadam =  Nadam(lr=1.e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=1.)

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

  return  nruns, labels

def plot(y, yhat, save=False):
  pl.clf()
  
  ymin  = 0.95 * y.min()
  ymax  = 1.05 * y.max()

  ys    = np.linspace(ymin, ymax, 10)

  pl.plot(ys,  ys, 'k-',   alpha=0.6)
  pl.plot(y, yhat, 'o', markersize=2)

<<<<<<< HEAD
def prep_model_2D_B_regress(optimizer=adam):
  model = Sequential()

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
  
  ##  Regression model.  Output constrained to [0., 1.] as appropriate for f. 
  model.add(layers.Dense(1), activation='sigmoid')

  model.compile(optimizer=adam, loss='mae', metrics=['accuracy'])

  model.summary()

  return  model


def pprocess(X):
  ##  Pre-process each 2D n-body slice:
  ##  --  Roll along random axis by random int npix
  ##  --  Random sign (should be applied to delta).
  ##  --  To be added:  Poisson sample. 
  ##  --  To be added:  Rescale amplitude by an arbitrary growth factor. 
  ##  
  axis = np.random.randint(2)
  npix = np.random.randint(128)
=======
  pl.xlabel(r'$f$')
  pl.ylabel(r'$\hat f$')
>>>>>>> 69c5653ba7f14eaed768b44b1fb4b3749d64da10

  pl.savefig('mlf.pdf')

  os.system('xpdf mlf.pdf')

    
if __name__ == '__main__':
  print('\n\nWelcome to MLRSD-2D.\n\n')

  train, regress = True, True

<<<<<<< HEAD
  ##  Set up the model.
  ##  model   = prep_model_2D_B(nhot, optimizer=adam)
  model   = prep_model_2D_B_regress(optimizer=adam)

  ##  Get list of available cosmologies. 
  cosmos  = np.loadtxt('cosmo.txt').tolist()
  labels  = []
=======
  optimizer      =  nadam
>>>>>>> 69c5653ba7f14eaed768b44b1fb4b3749d64da10
  
  nseeds         =      9     ##  Number of random (seed) sims available in each cosmology. 
  nslice         =     16     ##  Sliced 3D sim. into _nslice_ (x, z) slices.  
  nhot           =     10     ##  Supervised:  label sims by bin index in f;  nhot == # of bins. 
  nmesh          =    128
  nruns          =    900     ##  Limit the total number of mocks input;  Set to e.g. 1e99 for all. 
  epochs         =    500     ##  Number of actual (keras) epochs. 
  valid_frac     =   0.15     ##  Split data X into fractions for train, validate/test.    
  
  model          =  prep_model_2D_C(None, optimizer=optimizer, regress=regress)  
  nruns, labels  =  get_cosmos(nruns, nseeds, nslice)
          
  if train:
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
    ##  train_gen.fit(X)
      
<<<<<<< HEAD
      mesh            = BigFileMesh(fpath, dataset='1/Field', mode='real', header='Header').preview()
    
      for sslice in np.arange(nslice):
        ##  Split 3D sim into _nslice_ 2D (x, z) slices. 
        X[iid + nsplit * sslice, :, :, 0] = mesh[:, sslice, :]

    X_train           = X[:, :, :, :]    

    if train:
      ##  Note:  horizontal and vertical flipping of 2D slices.  
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

      ##  Digitze and one-hot encode target column.
      _y_train        = np.digitize(_labels[:,2], bins)
      y_train         = to_categorical(_y_train, num_classes=nhot)

      ##  Regress.
      y_train         = _labels

      ##  Image generator for continous cretion with pre-processing. 
      ##  Note:  Last fraction is saved for validation. 
      history         = model.fit_generator(train_gen.flow(X_train, y_train, batch_size=32, shuffle=True),\
                                            steps_per_epoch=10 * len(X_train) / 32, epochs=epochs)
      
      history         = history.history
  
      pickle.dump(history, open('history/history_%d.p' % zero, 'wb'))
=======
    ##  Image generator for continous creation with pre-processing;  steps_per_epoch=10 * len(X_train) / 32.
    ##  train_gen.flow(X, y, batch_size=256, shuffle=True)
    history = model.fit_generator(generator('./filelist.txt', batch_size=32, nmesh=nmesh, nslice=nslice, regress=regress, nhot=nhot),\
                                  steps_per_epoch=8000, epochs=epochs, use_multiprocessing=True)
>>>>>>> 69c5653ba7f14eaed768b44b1fb4b3749d64da10

    ##  history = model.fit(X, y, validation_split=0.3, epochs=epochs)
    (X_test, y_test) = generator('./filelist.txt', batch_size=32, nmesh=nmesh, nslice=nslice, regress=regress, nhot=nhot)
    
    ##  Get current predictions.
    y_hat = model.predict(X_test)

    ##  Plot current prediction against truth (regression or supervised).
    plot(y_test, y_hat)
    
    '''
    history         = history.history
    pickle.dump(history, open('history/history_%d.p' % zero, 'wb'))

    model_json      = model.to_json()

    with open('model/model_%d.json' % zero, 'w') as json_file:
      json_file.write(model_json)
      
    ##  model.save_weights('model.h5')
    '''
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
