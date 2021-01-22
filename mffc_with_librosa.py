# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import fft, fftshift
import scipy.io.wavfile
from scipy.fftpack import dct
import librosa
from librosa.display import specshow
import soundfile

audio_path = 'KendineIyiBakGitar.wav'
gitarWav , srG = librosa.load(audio_path, sr=6000)
print('Sampling rate: {} samples/second'.format(srG))
print('Signal size: {} samples'.format(gitarWav.shape[0]))
print('Signal duration: {:.3f} seconds'.format(gitarWav.shape[0] / srG))
print(type(gitarWav), type(srG))
plt.figure(figsize=(10, 2))
librosa.display.waveplot(gitarWav, sr=srG)
plt.title('Gitar Sesi Ses Dalgası')
plt.show()

audio_path2 = 'kendine-iyi-bak-keman.wav'
kemanWav , srK = librosa.load(audio_path2, sr=6000)
print('Sampling rate: {} samples/second'.format(srK))
print('Signal size: {} samples'.format(kemanWav.shape[0]))
print('Signal duration: {:.3f} seconds'.format(kemanWav.shape[0] / srK))
print(type(kemanWav), type(srK))
plt.figure(figsize=(10, 2))
librosa.display.waveplot(kemanWav, sr=srK)
plt.title('Keman Sesi Ses Dalgası')
plt.show()

kemanWav2sn=kemanWav[5:6*srK]
gitarWav2sn=gitarWav[5:6*srG]
plt.figure(figsize=(7, 4))
plt.subplot(2,1,1)
librosa.display.waveplot(gitarWav2sn, sr=srG)
plt.xlabel('Gitar Ses Dalgası')
plt.subplot(2,1,2)
librosa.display.waveplot(kemanWav2sn, sr=srK)
plt.xlabel('Keman ile Beraber Gitar Ses Dalgası')
plt.show()
#Önvurgulama
#y( t ) = x(t) - α.x(t-1)
#Gitar
pre_emphasis = 0.97
emphasized_gitarWav = np.append(gitarWav[0], gitarWav[1:] - pre_emphasis * gitarWav[:-1])
plt.figure(figsize=(7, 4))
plt.subplot(2,1,1)
librosa.display.waveplot(emphasized_gitarWav, sr=srG)
#plt.plot(emphasized_gitarWav)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Ön Vurgudan Geçmiş Gitar Ses Sinyali')
#plt.show()

#Keman
emphasized_kemanWav = np.append(kemanWav[0], kemanWav[1:] - pre_emphasis * kemanWav[:-1])
plt.plot(emphasized_kemanWav)
plt.subplot(2,1,2)
librosa.display.waveplot(emphasized_kemanWav, sr=srK)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Ön Vurgudan Geçmiş Keman Ses Sinyali')
plt.show()

#Spektrogram
#Konuşma işlemedeki tipik kare boyutları, ardışık çerçeveler arasında % 50 (+/-% 10) örtüşmeyle 20 ms ila 40 ms arasında değişir. Popüler ayarlar çerçeve boyutu için 25 ms
#frame_size = 0.025ve 10 ms'lik adım (15 ms üst üste binme) frame_stride = 0.01
window_length = int(0.025 * srG)
hop_length = int(0.01 * srG)
spectrogram = np.abs(librosa.stft(gitarWav, hop_length=hop_length, win_length=window_length))
print('Pencere Uzunluğu:{}'.format(window_length))
# Plotting the spectrogram:
specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=srG, hop_length=hop_length, y_axis='linear', x_axis='time')
plt.title('Gitar Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
#Önvurgudan Geçmiş Gitar Sesi Sinyalinin Spektrumu
spectrogramGitar2 = np.abs(librosa.stft(emphasized_gitarWav, hop_length=hop_length, win_length=window_length))
print('Pencere Uzunluğu:{}'.format(window_length))
specshow(librosa.amplitude_to_db(spectrogramGitar2, ref=np.max), sr=srG, hop_length=hop_length, y_axis='linear', x_axis='time')
plt.title('Önvurgulanmış Gitar Sinyali Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

#Keman Spektrogram
window_lengthK = int(0.025 * srK)
hop_lengthK = int(0.01 * srK)
spectrogramK = np.abs(librosa.stft(kemanWav, hop_length=hop_lengthK, win_length=window_lengthK))
print('Pencere Uzunluğu:{}'.format(window_lengthK))
# Plotting the spectrogram:
specshow(librosa.amplitude_to_db(spectrogramK, ref=np.max), sr=srK, hop_length=hop_lengthK, y_axis='linear', x_axis='time')
plt.title('Keman Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
spectrogramKeman2 = np.abs(librosa.stft(emphasized_kemanWav, hop_length=hop_lengthK, win_length=window_lengthK))
print('Pencere Uzunluğu:{}'.format(window_lengthK))

#w2 = np.hamming (window_length)
#w3=np.convolve(w2,emphasized_gitarWav)
#p2=fft(w3,2048)/25.5
#frekans1 = np.linspace(-0.1, 0.1, len(p2))
#buyukluk1 = np.abs(fftshift(p2))
#response1 = 20 * np.log10(buyukluk1)
#response1 = np.clip(response1, -100, 100)
#plt.plot(frekans1, response1)
#plt.show()


# Plotting the spectrogram:
specshow(librosa.amplitude_to_db(spectrogramKeman2, ref=np.max), sr=srK, hop_length=hop_lengthK, y_axis='linear', x_axis='time')
plt.title('Önvurgulanmış Keman Spectrogramı')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

#MFCC
high_freq_melGitar = (2595 * np.log10(1 + (srG / 2) / 700)) #.m=2595*log10(1+f/700)
high_freq_melKeman = (2595 * np.log10(1 + (srK / 2) / 700))
mfccsGitar = librosa.feature.mfcc(emphasized_gitarWav, srG, n_mfcc=40)
mfccsKeman = librosa.feature.mfcc(emphasized_kemanWav, srK, n_mfcc=40)

plt.figure(figsize=(7,4))
librosa.display.specshow(mfccsGitar, x_axis='time')
plt.colorbar()
plt.title('Gitar Ses SinyaliMFCC')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
librosa.display.specshow(mfccsKeman, x_axis='time')
plt.colorbar()
plt.title('Keman Ses SinyaliMFCC')
plt.tight_layout()
plt.show()

#Melspectrogram
melspectrogramGitar =librosa.feature.melspectrogram(y=emphasized_gitarWav, sr=srG, n_mels=40,fmax=high_freq_melGitar)
melspectrogramKeman =librosa.feature.melspectrogram(y=emphasized_kemanWav, sr=srK, n_mels=40,fmax=high_freq_melKeman)

plt.figure(figsize=(7,4))
librosa.display.specshow(librosa.power_to_db(melspectrogramGitar,ref=np.max),y_axis='mel', fmax=high_freq_melGitar,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Gitar spectrogram')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
librosa.display.specshow(librosa.power_to_db(melspectrogramKeman,ref=np.max),y_axis='mel', fmax=high_freq_melGitar,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Keman spectrogram')
plt.tight_layout()
plt.show()

#Chromagram
chroma_stftGitar=librosa.feature.chroma_stft(y=emphasized_gitarWav, sr=srG,n_chroma=40)
plt.figure(figsize=(7,4))
librosa.display.specshow(chroma_stftGitar, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Gitar- Chromagram')
plt.tight_layout()
plt.show()

chroma_stftKeman=librosa.feature.chroma_stft(y=emphasized_kemanWav, sr=srK,n_chroma=40)
plt.figure(figsize=(7,4))
librosa.display.specshow(chroma_stftKeman, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Keman- Chromagram')
plt.tight_layout()
plt.show()

chroma_censGitar =librosa.feature.chroma_cens(y=emphasized_gitarWav, sr=41000,n_chroma=40)
chroma_censKeman =librosa.feature.chroma_cens(y=emphasized_kemanWav, sr=41000,n_chroma=40)
plt.figure(figsize=(7,4))
librosa.display.specshow(chroma_censGitar, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Gitar chroma_cens')
plt.tight_layout()
plt.show()
print(chroma_censGitar)
print(chroma_censKeman)
plt.figure(figsize=(7,4))
librosa.display.specshow(chroma_censKeman, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Keman Ses Sinyali chroma_cens')
plt.tight_layout()
plt.show()

#Zero Crossing
n0 = 7000
n1 = 8100
zero_crossingsGitar = librosa.zero_crossings(emphasized_gitarWav[n0:n1], pad=False)
print(sum(zero_crossingsGitar))
zero_crossingsKeman = librosa.zero_crossings(emphasized_kemanWav[n0:n1], pad=False)
print(sum(zero_crossingsKeman))

#Spectral Centroid
#Spektrumun kütle merkezinin nerede olduğunu gösterir.
#Spektral sentroid, bir spektrumu karakterize etmek için dijital sinyal işlemede kullanılan bir ölçüdür.
#Spektrumun kütle merkezinin nerede olduğunu gösterir.

spec_centGitar=librosa.feature.spectral_centroid(gitarWav)
spec_centKeman=librosa.feature.spectral_centroid(kemanWav)
print(spec_centGitar.shape)
print(spec_centKeman.shape)
plt.semilogy(spec_centGitar.T[:500], color='r')
plt.semilogy(spec_centKeman.T[:500], color='b')
plt.ylabel("Hz")
plt.show()

#spec_centGitar=librosa.feature.spectral_centroid(emphasized_gitarWav)
#spec_centKeman=librosa.feature.spectral_centroid(emphasized_kemanWav)
#print(spec_centGitar.shape)
#print(spec_centKeman.shape)
#plt.semilogy(spec_centGitar.T[:500], color='r')
#plt.semilogy(spec_centKeman.T[:500], color='b')
#plt.ylabel("Hz")

#chromagram = librosa.feature.chroma_stft(kemanWav, sr=srK, hop_length=hop_lengthK)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

###################### Farklı Deneme ile

frame_stride = 0.01
frame_size = 0.025
frame_length, frame_step = frame_size * srG, frame_stride * srG   # Convert from seconds to samples
signal_length = len(emphasized_gitarWav)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_gitarWav, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
plt.plot(frames)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Çerçevelenmiş Sinyal Görüntüsü')
plt.show()

#Çerçeveleme Keman
frame_strideK = 0.01
frame_sizeK = 0.025
frame_lengthK, frame_stepK= frame_sizeK * srK, frame_strideK * srK   # Convert from seconds to samples
signal_lengthK = len(emphasized_kemanWav)
frame_lengthK = int(round(frame_lengthK))
frame_stepK = int(round(frame_stepK))
num_framesK = int(np.ceil(float(np.abs(signal_lengthK - frame_lengthK)) / frame_stepK))  # Make sure that we have at least 1 frame

pad_signal_lengthK = num_framesK * frame_stepK + frame_lengthK
zK = np.zeros((pad_signal_lengthK - signal_lengthK))
pad_signalK = np.append(emphasized_kemanWav, zK) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indicesK = np.tile(np.arange(0, frame_lengthK), (num_framesK, 1)) + np.tile(np.arange(0, num_framesK * frame_stepK, frame_stepK), (frame_lengthK, 1)).T
framesK = pad_signalK[indices.astype(np.int32, copy=False)]
plt.plot(framesK)
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Çerçevelenmiş Sinyal Görüntüsü')
plt.show()

#Pencereleme Gitar
frames *= np.hamming(frame_length)

plt.plot(frames[30:100])
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Hamming Penceresi ile Gitar Ses Sinyali')
plt.show()


#Pencereleme Keman
framesK *= np.hamming(frame_lengthK)
print(framesK.shape)
print(frames.shape)
plt.plot(framesK[4000:4010])
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.title('Hamming Penceresi ile Keman Ses Sinyali')
plt.show()


