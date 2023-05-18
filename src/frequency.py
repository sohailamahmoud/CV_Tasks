import numpy as np
import matplotlib.pyplot as plt

def Fourier(image):
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    return Fshift

def InverseFourier(Gshift):
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    return g

def LowPass(image):
    # image in frequency domain
    Fshift=Fourier(image)
    
    # Filter: Low pass filter
    M,N = image.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 25
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u,v] = 1
            else:
                H[u,v] = 0
    
    # Inverse Fourier Transform
    Gshift = Fshift * H
    low_frequencies= InverseFourier(Gshift)
    
    return low_frequencies

def HighPass(image):
    # image in frequency domain
    Fshift=Fourier(image)
     
    # Filter: High pass filter
    M,N = image.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 7
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u,v] = 1
            else:
                H[u,v] = 0
    H = 1 - H
    
    # Inverse Fourier Transform
    Gshift = Fshift * H
    high_frequencies= InverseFourier(Gshift)
    
    return high_frequencies
  

def hybrid(low,high):
    hybrid_image = low + high
    return hybrid_image
    

