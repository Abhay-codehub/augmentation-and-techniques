import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io

img = skimage.io.imread()  #reading image
img = skimage.img_as_float(img) #connverting image to float

sigma = 0.1 #noise Std
imgn = random_noise(img,var=sigma**2) #addding noise

sigma_est =estimate_sigma(imgn,average_sigmas=True) # noise estimation

#denoising using bytes

img_bayes = denoise_wavelet(imgn,method='BayesShrink',mode='soft',wavelet_levels=3,wavelet='bior6',rescale_sigma=True)

#denoising using visushrink
img_visushrink = denoise_wavelet(imgn,method='VisuShrink',mode='soft',sigma=sigma_est/3,wavelet_levels=5,wavelet='bior6.8',rescale_sigma=True)

#finding psnr
psnr_noisy = peak_signal_noise_ratio(img,imgn)
psnr_bayes =peak_signal_noise_ratio(img,img_bayes)
psnr_visu = peak_signal_noise_ratio(img,img_visushrink)

#plotting images
plt.figure(figsize=(30,30))

plt.subpoint(2,2,1)
plt.imshow(img,camp=plt.cm.gray)
plt.title('Original Image',fontsize=30)

plt.subplot(2,2,2)
plt.imshow(imgn,cmap=plt.cm.gray)
plt.title('Noisy Image',fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_bayes,cmap=plt.cm.gray)
plt.title('Denoised Image using Bayes',fontsize=30)

plt.subplot(2,2,4)
plt.imshow(img_visushrink,cmap=plt.cm.gray)
plt.title('Denoised Image using visushrink',fontsize=30)
plt.show()

#printing psnr
print('PSNR[ORIGINAL VS NOISY IMAGE]:',psnr_noisy)
print('PSNR[ORIGINAL VS DENOISED(visushrink)]:',psnr_visu)
print('PSNR[ORIGINAL VS DENOISED(BAYES)]:',psnr_bayes)










