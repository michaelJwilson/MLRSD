import numpy as np


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
