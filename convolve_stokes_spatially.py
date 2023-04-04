import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 
import sys 

# This convolves a given Stokes cube spatially, either with a Gaussian, 
# Bessel, or with something else that you can read from the file and hard-code in there


# Some hard-coded stuff:
# ------------------------------------------------------------------------------------------
# Prepare the PSF:
D = 0.1 # m 
llambda = 630E-9

pix_size = 16E3 # in m

noise = 0.0

# -------------------------------------------------------------------------------------------
pixel_scale_arcseconds = 16E3 / 725E3

# PSF from astropy! 

PSF = np.zeros([1,1])

profile_type = sys.argv[3]

if (profile_type == 'airy'):
	
	print ("Convolving with airy psf")

	from astropy.convolution import AiryDisk2DKernel
	zero = 1.22*llambda / D * 206265. / pixel_scale_arcseconds
	PSF = AiryDisk2DKernel(zero)
	PSF = np.asarray(PSF)
	N_psf = len(PSF[0])
	print ("PSF radius =", zero)

if (profile_type == 'gauss'):

	print ("Convolving with gauss psf")

	from astropy.convolution import Gaussian2DKernel

	sigma = 1.22 *llambda / D * 206265. / 2.35 / pixel_scale_arcseconds
	PSF = Gaussian2DKernel(sigma,sigma)
	PSF = np.asarray(PSF)
	N_psf = len(PSF[0])
	print ("PSF stddev =", sigma)

if (profile_type == 'given'):

	print ("Convolving with a prescribed PSF")

	PSF = fits.open("/home/milic/codes/data_degradation/hinodepsf.fits")[0].data

	N_psf = len(PSF[0])


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

stokes_new = np.zeros([NX_new, NY_new, NS, NL], dtype='float32')

from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy import signal
from timeit import default_timer as timer


for s in range(0,NS):
	start = timer()
	current_stokes = np.copy(source_file[0].data[:,:,s,:])
	current_stokes = current_stokes.transpose(2,0,1)
	end = timer()
	print("Loaded s = ",s, end-start)
	for l in range(0,NL):

		# assign a bigger frame, with the goal of making wrapping manually:
		padding = N_psf // 2 + 1
		NXw = NX + padding * 2
		NYw = NY + padding * 2


		current_frame = np.zeros([NXw,NYw])

		tmp = np.copy(current_stokes[l])
		current_frame[padding:-padding,padding:-padding] = tmp
			
		# bands
		current_frame[0:padding,padding:-padding] = np.copy(tmp[-padding:,:])
		current_frame[-padding:,padding:-padding] = np.copy(tmp[0:padding,:])
		current_frame[padding:-padding,0:padding] = np.copy(tmp[:,-padding:])
		current_frame[padding:-padding,-padding:] = np.copy(tmp[:,0:padding])

		# corners
		current_frame[0:padding,0:padding] = np.copy(tmp[-padding:,-padding:])
		current_frame[0:padding,-padding:] = np.copy(tmp[-padding:,0:padding])
		current_frame[-padding:,0:padding] = np.copy(tmp[0:padding,-padding:])
		current_frame[-padding:,-padding:] = np.copy(tmp[0:padding,0:padding])
		#end1 = timer()
		#print ("padded", end1-start)
			
			
		convolved = signal.fftconvolve(current_frame[:,:], PSF, mode = 'valid')
		#convolved = convolve(tmp, PSF, mode = 'wrap')
		del(current_frame)
			#end2 = timer()
			#print ("convolved", end2-end1)
		# This below is to polish the size, as the 'valid' outputs a bit higher dimensions 
		convolved = convolved[1:-1,1:-1]
		#print (convolved.shape)

		convolved_binned = convolved.reshape(NX_new, binning, NY_new, binning)
			#end3 = timer()
			#print ("reshaped", end3-end2)
		norm = binning ** 2.0
		convolved_binned = np.sum(convolved_binned,axis=(1,3)) / norm

		convolved_binned += np.random.normal(0,noise,(NX_new,NY_new))
			#end4 = timer()
			#print ("binned", end4-end3)

		stokes_new[:,:,s,l] = convolved_binned
			#end5 = timer()
			#print ("stored", end5-end4)
		
		if (s==0 and l==0):
			print ('mean:', np.mean(stokes_new[:,:,s,l]))
			print ('std:', np.std(stokes_new[:,:,s,l]))
			print ('contrast:', np.std(stokes_new[:,:,s,l])/np.mean(stokes_new[:,:,s,l]))
		del(convolved_binned)
		del(tmp)
		print (s,l)


output_name = sys.argv[4]

plt.figure(figsize=[9,8])
plt.imshow(stokes_new[:,:,0,0].T,cmap='magma',origin='lower',vmin=0.8,vmax=1.2)
plt.tight_layout()
contrast = np.std(stokes_new[:,:,0,0]) / np.mean(stokes_new[:,:,0,0])
plt.title("Contrast = "+str(contrast))
plt.savefig(output_name[:-5]+'.png',bbox_inches='tight')

plt.figure(figsize=[13,5])
plt.subplot(121)
plt.imshow(np.log10(PSF.T),cmap='viridis',origin='lower',vmin=-6,vmax=-2)
plt.subplot(122)
plt.plot(PSF[:,N_psf//2])
plt.tight_layout()
plt.savefig('psf.png',bbox_inches='tight')

myhdu = fits.PrimaryHDU(stokes_new)
myhdu.writeto(output_name,overwrite=True)
