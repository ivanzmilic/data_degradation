import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 
import sys

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

file_in = sys.argv[1]

data = fits.open(file_in)

# Hard-code wavelength grid:

ll0 = np.linspace(6301.0, 6303.0, 201) # in A

# Hard-code spectral profile:

# Resolution: 
dlambda = 30E-3 # in mA 

step = ll0[1] - ll0[0]

NX = data[0].data.shape[0]
NY = data[0].data.shape[1]
NS = data[0].data.shape[2]
NL = data[0].data.shape[3]

print ("Input data shape: ", NX, NY, NS, NL)


noise = 1E-3;

# Hinode wavelengths:

ll = np.linspace(0,111,112)
ll = 6302.08 + (ll-56.5)*0.0215
NLnew = len(ll)

new = np.zeros([NX//2, NY//2, NS, NLnew], dtype='float32')

for i in range(0,NX//2):
	for j in range(0,NY//2):
		for s in range(0,NS):

			og = data[0].data[i,j,s]

			convolved = gaussian_filter(og,dlambda/step)

			# resample

			f = interp1d(ll0, convolved, fill_value='extrapolate')

			# 

			new[i,j,s] = f(ll)

		print(i,j)


hdu_out = fits.PrimaryHDU(new)

hdu_out.writeto(sys.argv[2],overwrite=True)
