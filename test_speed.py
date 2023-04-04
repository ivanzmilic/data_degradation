import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits 
from scipy import signal
import sys


PSF = np.random.randn(201,201)
NX_new = 768
NY_new = 768
binning = 2

stokes_new = np.zeros([NX_new, NY_new,4,201])

from timeit import default_timer as timer

kek = fits.open(sys.argv[1])

for s in range(0,1):
	for l in range(0,11):
		start = timer()

		

		end1 = timer()
		print ("Loaded the frame", end1-start)
				
		convolved = signal.fftconvolve(kek[0].data[:,:,s,l], PSF, mode = 'same')
		end2 = timer()
		print ("convolved", end2-end1)

		convolved_binned = convolved.reshape(NX_new, binning, NY_new, binning)
		end3 = timer()
		print ("reshaped", end3-end2)
		convolved_binned = np.sum(convolved_binned,axis=(1,3)) / binning ** 2.0
		end4 = timer()
		print ("binned", end4-end3)

		#stokes_new[:,:,s,l] = np.copy(convolved_binned)
		stokes_new[:,:,s,l] = convolved_binned
		end5 = timer()
		print ("stored", end5-end4)