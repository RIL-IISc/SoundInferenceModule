import librosa 
import librosa.display 

import numpy as np
import cv2

class Spectrogram:

    def __init__(self, audio_wav_file):
        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = 1024
        self.audio_wav_file = audio_wav_file


    def wav2spec(self):

        x, sr = librosa.load(self.audio_wav_file, sr=16000)
        stft = librosa.stft(x, n_fft=1024, hop_length=256, win_length=1024, center=True)

        mag, phase = librosa.core.magphase(stft)
        mag = (librosa.power_to_db(
           mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1

        mag = cv2.resize(mag, (256, 256))
        mag = mag[:, :, np.newaxis]
        
        
        '''
        stft = np.abs(librosa.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length))

        # Get frequencies assiciated with STFT
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        # Filtering Noise
        # Apply filter on frequency range
        upper = ([i for i in range(len(freqs)) if freqs[i] >= 5000])[0]
        lower = ([i for i in range(len(freqs)) if freqs[i] <= 340])[0]

        #print(freqs)
        freqs = freqs[lower:upper]
        #print(freqs)
        stft = stft[lower:upper,:]

        Xdb = librosa.amplitude_to_db(stft)
        #Xdb = (((Xdb - np.amin(Xdb))*2) / (np.amax(Xdb) - np.amin(Xdb))) / 2 
        
        n = ((Xdb - np.amin(Xdb))*2)
        d = (np.amax(Xdb) - np.amin(Xdb))
    
        Xdb = 0.5 * (n / d)
        Xdb = np.log1p(Xdb)
        
        
        #stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
        #stft_logscale = np.log1p(stft)

        '''

        return mag


        
