
# coding: utf-8

# In[19]:


import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

#leitura do arquivo ecg.mat

mat = scipy.io.loadmat('ecg.mat')
print mat


#mat = scipy.io.loadmat('ecg.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame()



plt.show()


# In[8]:


"""DFT and FFT"""
import math

def iexp(n):
    return complex(math.cos(n), math.sin(n))

def is_pow2(n):
    return False if n == 0 else (n == 1 or is_pow2(n >> 1))

def dft(xs):
    "naive dft"
    n = len(xs)
    return [sum((xs[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]

def dftinv(xs):
    "naive dft"
    n = len(xs)
    return [sum((xs[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
            for i in range(n)]

def fft_(xs, n, start=0, stride=1):
    "cooley-turkey fft"
    if n == 1: return [xs[start]]
    hn, sd = n // 2, stride * 2
    rs = fft_(xs, hn, start, sd) + fft_(xs, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(-2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

def fft(xs):
    assert is_pow2(len(xs))
    return fft_(xs, len(xs))

def fftinv_(xs, n, start=0, stride=1):
    "cooley-turkey fft"
    if n == 1: return [xs[start]]
    hn, sd = n // 2, stride * 2
    rs = fftinv_(xs, hn, start, sd) + fftinv_(xs, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

def fftinv(xs):
    assert is_pow2(len(xs))
    n = len(xs)
    return [v / n for v in fftinv_(xs, n)]

if __name__ == "__main__":
    wave = [0, 1, 2, 3, 3, 2, 1, 0]
    dfreq = dft(wave)
    ffreq = fft(wave)
    dwave = dftinv(dfreq)
    fwave= fftinv(ffreq)
    print(dfreq)
    print(ffreq)
    print([v.real for v in dwave])
    print([v.real for v in fwave])
    pass




# In[32]:


#código simples de fft da internet

from scipy.fftpack import fft
import numpy as np
# Number of sample points
N = 100
# sample spacing
T = 1.0 / 500.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()


# In[33]:


#nseitolok

import matplotlib.pyplot as plt
import plot.plotly as py
import numpy as np
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')

