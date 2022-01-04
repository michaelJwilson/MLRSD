import  os
import  json
import  copy
import  matplotlib
import  matplotlib.pyplot             as      plt
import  pylab                         as      pl
import  numpy                         as      np

from    tensorflow.keras.datasets     import  mnist
from    tensorflow.keras.utils        import  to_categorical
from    tensorflow.keras.models       import  Sequential
from    tensorflow.keras.layers       import  Dense, Conv2D, Flatten, Conv3D, Dropout
from    tensorflow.keras.models       import  model_from_json
from    nbodykit.lab                  import  LinearMesh, cosmology, BigFileMesh, BigFileCatalog, FFTPower


print('\n\nWelcome.\n\n')

nruns   = 900

LOS     = [0,0,1]
X       = np.zeros((nruns, 128, 128, 128))

for iid in np.arange(nruns):
  print('Loading %d' % iid)

  fpath             = os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000' % iid
  cat               = BigFileCatalog(fpath, dataset='1/', header='Header')

  rsd_factor        = cat.attrs['RSDFactor']
  cat['zPosition']  = cat['Position'] + rsd_factor * cat['Velocity'] * LOS

  mesh              = cat.to_mesh(Nmesh=128, BoxSize=1000., position='zPosition')

  ##  delta         = mesh.paint(mode='real') - 1.0
  ##  deltak        = mesh.paint(mode='complex')
  
  X[iid,:,:,:]      = mesh.preview()

  ##  mesh.save(os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000' % iid, dataset='1/Field',  mode='real')
  mesh.save(os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000' % iid, dataset='1/kField', mode='complex')
  
  os.system('mkdir -p %s' % os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000/1/multipoles/' % iid)

  multipoles        = FFTPower(mesh, mode='2d', dk=0.005, kmin=0.01, Nmu=5, los=[0,0,1], poles=[0,2,4])
  multipoles.save(os.environ['SCRATCH'] + '/fastpm/fpm-%d-1.0000/1/multipoles/000000' % iid)
  
  ##  break
