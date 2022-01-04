import  os
import  numpy              as      np
import  pylab              as      pl
import  matplotlib.pyplot  as      plt

from    nbodykit.lab       import  FFTPower



fpath      = os.environ['SCRATCH'] + '/fastpm/fpm-46-1.0000/1/multipoles/000000'

multipoles = FFTPower.load(fpath).poles

for ell in [0, 2, 4]:
  label = r'$\ell=%d$' % (ell)
  P     = multipoles['power_%d' %ell].real

  if ell == 0:
    P   = P - multipoles.attrs['shotnoise']

  plt.loglog(multipoles['k'], np.abs(P), label=label)

plt.legend(loc=0)
plt.xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
plt.ylabel(r'$P_\ell(k)$   [$(h^{-1} \mathrm{Mpc})^3$]')

plt.xlim(0.01, 0.6)
pl.savefig('plots/mutlipoles.pdf')

os.system('xpdf plots/mutlipoles.pdf')
