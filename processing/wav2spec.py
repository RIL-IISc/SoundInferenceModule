import librosa 
import librosa.display 

import numpy as np
from skimage.measure import block_reduce 

#np.seterr(divide='ignore', invalid='ignore')


class Spectrogram:

    def __init__(self, audio_wav_file):
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.audio_wav_file = audio_wav_file


    def wav2spec(self):

        x, sr = librosa.load(self.audio_wav_file, sr=16000)
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

        return Xdb


        
