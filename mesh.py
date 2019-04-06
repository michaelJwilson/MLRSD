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


print('\n\nWelcome.\n\n')

nruns   = 900

LOS     = [0,0,1]
X       = np.zeros((nruns, 128, 128, 128))

for iid in np.arange(nruns):
  print('Loading %d' % iid)

  fpath             = '/global/cscratch1/sd/mjwilson/MLRSD/fastpm/fpm-%d-1.0000' % iid
  cat               = BigFileCatalog(fpath, dataset='1/', header='Header')

  rsd_factor        = cat.attrs['RSDFactor']
  cat['zPosition']  = cat['Position'] + rsd_factor * cat['Velocity'] * LOS

  mesh              = cat.to_mesh(Nmesh=128, BoxSize=1000., position='zPosition')
  X[iid,:,:,:]      = mesh.preview()

  mesh.save('/global/cscratch1/sd/mjwilson/MLRSD/fastpm/fpm-%d-1.0000' % iid, dataset='1/Field', mode='real')
