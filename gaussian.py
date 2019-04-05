import os
import numpy as np
import pylab as pl

from   nbodykit.lab      import LinearMesh, cosmology, BigFileMesh
from   matplotlib        import pyplot as plt
from   astropy.cosmology import FlatLambdaCDM


read   = False

H0s    = np.linspace(65., 75., 10)
Oms    = np.linspace(.25, .35, 10)

runs   = []
ocosmo = []

for H0 in H0s:
  for Om in Oms:
    params = {'H0': H0, 'Om0': Om, 'Ob0':0.04, 'Tcmb0':2.7}
    acosmo = FlatLambdaCDM(**params)
    cosmo  = cosmology.Cosmology.from_astropy(acosmo)

    runs.append(cosmo)
    ocosmo.append([H0, Om, Om ** 0.545])

ocosmo  = np.array(ocosmo)
np.savetxt('cosmo.txt', ocosmo, fmt='%.6le')

scratch = os.environ['CSCRATCH'] 
iid     = 0

for cosmo in runs:
  Plin  = cosmology.EHPower(cosmo, redshift=0.0)

  for seed in np.arange(1, 10, 1):
    print('Solving for %d' % iid)

    if not read:
      os.system('mkdir -p %s/MLRSD/fields/field_%d' % (scratch, iid))

      mesh = LinearMesh(Plin, Nmesh=128, BoxSize=1.e3, seed=seed)
      mesh.save('%s/MLRSD/fields/field_%d' % (scratch, iid))

    else:
      mesh = BigFileMesh('%s/MLRSD/fields/field_%d' % (scratch, iid), 'Field')
  
    iid += 1

print('\n\nDone.\n\n')
