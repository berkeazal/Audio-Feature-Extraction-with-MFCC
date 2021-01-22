# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:26:10 2021

@author: burak.cekic
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import fft, fftshift
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct


samplingFrequency, signalData = wavfile.read(r'101_1305030823364_E.wav')
signal = signalData[0:int(3.5 * samplingFrequency)]
plt.plot(signal)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Ses Sinyali')
plt.show()
print(samplingFrequency)

#Önvurgulama
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
plt.plot(emphasized_signal)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Ön Vurgudan Geçmiş Ses Sinyali')
plt.show()


#Çerçeveleme
frame_stride = 0.01
frame_size = 0.025
frame_length, frame_step = frame_size * samplingFrequency, frame_stride * samplingFrequency   # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
plt.plot(frames)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Çerçevelenmiş Sinyal Görüntüsü')
plt.show()


#Pencereleme
frames *= np.hamming(frame_length)
plt.plot(frames)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Hamming Penceresi ile Ses Sinyali')
plt.show()

#Fourier Dönüşümü ve Güç Spektrumu
NFFT = 512 #FFT boyutu. Varsayılan 512'dir
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
plt.plot(mag_frames)
plt.xlabel('Frekans')
plt.ylabel('Genlik')
plt.title('Pencerelerin FFT Dönüşümü ile Büyüklük Spektrumu')
plt.show()
plt.plot(pow_frames)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Güç Spektrumu')
plt.show()

#FiltreBankaları
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (samplingFrequency / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / samplingFrequency)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1)))) #nfilt: filtre bankasındaki filtre sayısı, varsayılan 26.

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
print(filter_banks)
plt.imshow(filter_banks, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('Filter Bank Indesi')
plt.xlabel('Çerçeve İndeksi')
plt.show()
#Mel-frekans Cepstral Katsayıları (MFCC'ler)
num_ceps = 12
cep_lifter = 22 #son seferin katsayılarına bir kaldırıcı uygular. 0 kaldırıcı değildir. Varsayılan 22'dir
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift  #*
print(mfcc)

#OrtalamaNormalizasyon
filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

plt.imshow(mfcc, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('MFCC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()
