import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 
import sys 

import multiprocessing as mp

# This convolves a given Stokes cube spatially, either with a Gaussian, 
# Bessel, or with something else that you can read from the file and hard-code in there


# Some hard-coded stuff:
# ------------------------------------------------------------------------------------------
# Prepare the PSF:
D = 0.5
llambda = 630E-9

pix_size = 16E3 # in m

# PSF array size:
N_psf = 101

# -------------------------------------------------------------------------------------------
PSF = np.zeros([N_psf, N_psf])
x = np.linspace(0, N_psf-1.0, N_psf) - (N_psf - 1) * 0.5 
y = np.linspace(0, N_psf-1.0, N_psf) - (N_psf - 1) * 0.5 


pixel_scale_arcseconds = 16E3 / 725E3

# Under the assumption that sin theta = theta

k = 2.0 * np.pi / llambda

x *= (k* pixel_scale_arcseconds * D / 206265)
y *= (k* pixel_scale_arcseconds * D / 206265)

from scipy.special import jv

r = np.sqrt(x[:,None]**2.0 + y [None,:]**2.0) ** 0.5

PSF = (jv(1, r + 0.001 * pixel_scale_arcseconds)/(r + 0.001 * pixel_scale_arcseconds)) ** 2.0

PSF /= np.sum(PSF)

## that was PSF creation, let's move on:

input_file = sys.argv[1]

# We will partially load it because these things can be huge:

binning = int(sys.argv[2])

source_file = fits.open(input_file)

example_frame = source_file[0].data[:,:,0,0]

NX = example_frame.shape[0]
NY = example_frame.shape[1]

print ("Original image size :", NX, NY)

NX_new = NX // binning
NY_new = NY // binning

print ("Binned image size :", NX_new, NY_new)

# Figure out size of other stuff:

NS = len(source_file[0].data[0,0,:,0])
NL = len(source_file[0].data[0,0,0,:])

# New file:

stokes_new = np.zeros([NX_new, NY_new, NS, NL])

from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

profile_type = sys.argv[3]

print("Number of processors: ", mp.cpu_count())

pool = mp.Pool(mp.cpu_count())


# Use only one iterable:

def degrade(image): #PSF, psftype, modei, NX_new, NY_new, binning):
	
	#convolved = convolve(image, PSF,  mode = modei)

	#convolved_binned = convolved.reshape(NX_new, binning, NY_new, binning)
	#convolved_binned = np.sum(convolved_binned,axis=(1,3)) / binning ** 2.0

	#return convolved

	total = np.sum(image)

	return total

# ----------------------------------------------------	


data = np.zeros([8,16,13])

data[:,:,:] = 1.0

if __name__ == '__main__':
    # start 4 worker processes
    with mp.Pool(processes=1) as pool:

    	results = [pool.apply_async(degrade, args=(row)) for row in data]
    	pool.close()
    	print (results)

