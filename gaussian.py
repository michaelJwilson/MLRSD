import os
import numpy as np
import pylab as pl

from nbodykit.lab import LinearMesh, cosmology, BigFileMesh
from matplotlib import pyplot as plt

cosmo  = cosmology.Planck15
Plin   = cosmology.EHPower(cosmo, redshift=0)

labels = [cosmo.Omega0_cdm, cosmo.h]

read   = False

for seed in np.arange(42, 52, 1):
  pl.clf()

  if not read:
    os.system('mkdir fields/field_%d' % seed)

    mesh = LinearMesh(Plin, Nmesh=128, BoxSize=1380, seed=42)
    mesh.save('fields/field_%d' % seed)

  else:
    mesh = BigFileMesh('fields/field_%d' % seed, 'Field')
  
  plt.imshow(mesh.preview(axes=[0,1]))

  ##  pl.show()
  pl.savefig('fields/pdfs/field_%d.pdf' % seed)
