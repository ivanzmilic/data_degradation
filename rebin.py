import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits 
from scipy.interpolate import interp2d
import sys

file_in = sys.argv[1]

angle = np.radians(float(sys.argv[2]))

axis = int(sys.argv[3])

NX, NY = fits.open(file_in)[0].data[0,:,:,0].shape
print (NX, NY)

step = 16.0

x = np.linspace(0,NX-1,NX) * step
y = np.linspace(0,NY-1,NY) * step

x_new = np.copy(x)
y_new = np.copy(y)

NX_new = NX
NY_new = NY

if (axis==0):
	x = x*np.cos(angle)
	NX_new = int(x[-1]/step)+1

if (axis==1):
	y = y * np.cos(angle)
	NY_new = int(y[-1]/step)+1

print (NX_new, NY_new)

x_new = np.linspace(0,NX_new-1,NX_new) * step
y_new = np.linspace(0,NY_new-1,NY_new) * step

cube_new = np.zeros([NX_new, NY_new, 4, 201])

for s in range(0,4):
	for l in range(0,201):

		f = interp2d(x,y,fits.open(file_in)[0].data[101+l,:,:,s])

		cube_new[:,:,s,l] = f(x_new,y_new).T

		print (s,l)

cube_new = cube_new.transpose(1,0,2,3)

kekbur = fits.PrimaryHDU(cube_new)

kekbur.writeto(file_in[:-5]+'_tilted.fits',overwrite=True)
