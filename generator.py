import  os
import  glob
import  numpy as np

from    nbodykit.lab                          import  BigFileMesh
from    tensorflow.keras.utils                import  to_categorical


def write_filelist():
  root   = os.environ['SCRATCH']
  files  = glob.glob(root + '/fastpm/*')

  files  = [x for x in files if x.split('-')[-1][:3] == '1.0']

  with open('filelist.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item)
  
def generator(fpath, batch_size=32, nmesh=128, nslice=16, mode='train', regress=True, nhot=10, _print=False): 
  f = open(fpath, 'r')

  while True:
    images = []
    labels = []

    jump   = np.floor(nmesh / nslice).astype(np.int)

    while len(images) < batch_size:
      line = f.readline()
      line = line.replace('\n', '')
      
      if line == '':
        ##  https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        f.seek(0)
        line = f.readline()

        if mode == 'eval':
          break

      _file = BigFileMesh(line, dataset='1/Field', mode='real', header='Header')    
      attrs = _file.attrs

      ##  (1. + delta) on preview. 
      mesh  = _file.preview() - 1.0

      if _print:
        print('Loading mock:  (h = %.3lf, Om = %.3lf, f = %.3lf).' % (attrs['h'], attrs['Om0'], attrs['Om0']**0.545))

      for ii, sslice in enumerate(np.arange(0, nmesh, jump)):
        ##  Split 3D sim into _nslice_ 2D (x, z) slices;  Mesh returns (1 + delta).
        images.append(mesh[:, sslice, :])
        labels.append([attrs['h'], attrs['Om0'], attrs['Om0']**0.545])

        ##  Bin sims in f and use bin index as a supervised label.

    images   = np.array(images).reshape(batch_size, nmesh, nmesh, 1)
    labels   = np.array(labels)  

    ## Use f as label.
    labels   = labels[:, 2]
    
    if not regress:
      fmin     = .25 ** 0.545
      fmax     = .35 ** 0.545

      ##  Number of one-hot encodings == number of bins.                                                                                                                                                                                                                                                  
      bins     = np.linspace(fmin, fmax, nhot)
    
      labels   = np.digitize(labels, bins)

      ##  One-hot encode target column.                                                                                                                                                                                                                                                                  
      labels   = to_categorical(labels, num_classes=nhot)

    ## 
    yield (images, labels)

  
if __name__ == '__main__':
  write_filelist()

  fpath = './filelist.txt'
  
  for x, y in generator(fpath, batch_size=32, nmesh=128, regress=True):
    pass
