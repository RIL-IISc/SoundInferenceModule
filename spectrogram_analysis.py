import matplotlib.pyplot as plt 
import numpy as np

import librosa
import librosa.display 

audio_data = './final_data/0_4_0_90_channel4.wav'

x, sr = librosa.load(audio_data, sr=16000)

print("length of x:", len(x))

x = ((x - np.amin(x))*2)/(np.amax(x) - np.amin(x)) - 1

X = np.abs(librosa.stft(x, n_fft=512, hop_length=256, win_length=512))

# Get frequencies assiciated with STFT
freqs = librosa.fft_frequencies(sr=sr, n_fft=512)

# Apply filter on frequency range
upper = ([i for i in range(len(freqs)) if freqs[i] >= 5000])[0]
lower = ([i for i in range(len(freqs)) if freqs[i] <= 340])[-1]

#print(freqs)
freqs = freqs[lower:upper]
#print(freqs)
X = X[lower:upper,:]


Xdb = librosa.amplitude_to_db(X)

Xdb = (((Xdb - np.amin(Xdb))*2) / (np.amax(Xdb) - np.amin(Xdb))) / 2 


print('Sampling rate:', sr)
print('shape:', Xdb.shape)


#plt.figure(figsize=(14,5))
#librosa.display.waveplot(x, sr=sr)
#plt.show()



plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr,cmap='jet', x_axis='time', y_axis='hz')
#plt.imshow(np.transpose(Xdb), extent=[0,4.2,0,48000], cmap='jet', vmin=-100, vmax=0, origin='lowest', aspect='auto')
plt.colorbar()
plt.show()





