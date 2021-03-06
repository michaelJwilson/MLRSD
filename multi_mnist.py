##  import  matplotlib;                   matplotlib.use('Agg')

import  tensorflow                    as      tf

from    tensorflow.keras.datasets     import  mnist
from    tensorflow.keras.utils        import  to_categorical
from    tensorflow.keras.models       import  Sequential
from    tensorflow.keras.layers       import  Dense, Conv2D, Flatten
from    tensorflow.keras.models       import  model_from_json
from    tensorflow.keras              import  backend
from    tensorflow.keras.utils        import  multi_gpu_model

##  import  matplotlib.pyplot  as      plt
##  import  pylab              as      pl
import  numpy                         as      np


train                           = True
'''
##  export CUDA_VISIBLE_DEVICES=1
config                          = tf.ConfigProto(device_count = {'GPU': 1})
session                         = tf.Session(config=config)
backend.set_session(session)
'''
print('\n\nWelcome.\n\n')

(_X_train, _y_train), (_X_test, _y_test) = mnist.load_data()

##  28 x 28 pixels, with 60K items in train and 10K in validation.  
X_train = _X_train.reshape(60000, 28, 28, 1)
X_test  =  _X_test.reshape(10000, 28, 28, 1)

##  one-hot encode target column.
y_train = to_categorical(_y_train)
y_test  = to_categorical(_y_test)

if train:
  model = Sequential()

  model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
  model.add(Conv2D(32, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(10, activation='softmax'))

  pmodel = multi_gpu_model(model, gpus=2)
  pmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  pmodel.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
                                                                                                                 
  model_json = pmodel.to_json()

  with open('model.json', 'w') as json_file:
    json_file.write(model_json)

  model.save_weights('model.h5')
'''
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

plt.figure(figsize=(10, 10))

for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(_X_train[i], cmap=plt.cm.binary)
  
  plt.ylabel(np.argmax(predictions[i], axis=None, out=None))
  plt.xlabel(_y_train[i])

plt.savefig('mnist.png')
'''
print('\n\nDone.\n\n')
