import librosa 
import librosa.display
import wave

import numpy as np

import os 
import glob 
import ntpath
from collections import defaultdict 

from processing.wav2spec import Spectrogram 

import tensorflow as tf 
from tensorflow.keras.models import Model


# Remove path address and retail wav file name

class Create_dataset:
    def __init__(self, model_file, audio_file_dir, x_sound, y_sound):
        self.model_file = model_file
        self.audio_file_dir = audio_file_dir
        self.x_sound = x_sound
        self.y_sound = y_sound

    def get_encoder_model(self):

        model = tf.keras.models.load_model(self.model_file)
        encoder = Model(inputs=model.input, outputs=model.layers[24].output)
        
        return encoder


    def path_leaf(self, path):
        ntpath.basename("a/b/c/d")
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    # West (+ve y-axis) -> 90, South (-ve x-axis) -> 180, East (-ve y-axis) -> 270, North (+ve x-axis) -> 0
    def get_relative_distance(self, x_robot, y_robot, x_sound, y_sound, orientation_robot):
        
        if orientation_robot == 0:
            x_relative = x_sound - x_robot
            y_relative = y_sound - y_robot

        elif orientation_robot == 90:
            x_relative = y_sound - y_robot
            y_relative = - x_sound + x_robot

        elif orientation_robot == 180:
            x_relative = -x_sound + x_robot
            y_relative = -y_sound + y_robot
        
        elif orientation_robot == 270:
            x_relative = -y_sound + y_robot
            y_relative = x_sound - x_robot

        return x_relative, y_relative
            


    def group_data(self):

        groups = defaultdict(list)
        labels = defaultdict(list)

        for filename in sorted(glob.glob(os.path.join(self.audio_file_dir, '*.wav'))):
            audio_file_name = self.path_leaf(filename)
            basename, extension = os.path.splitext(audio_file_name)
            #print(basename)

            sample, x_robot, y_robot, orientation_robot, channel_number = basename.split('_')

            
            
            x_relative, y_relative = self.get_relative_distance(int(x_robot), int(y_robot), self.x_sound, self.y_sound, int(orientation_robot))

            temp = []
            temp = [int(x_relative), int(y_relative)]

            labels[sample] = temp
            groups[sample].append(filename)
        
        return groups, labels

    def dataset(self):

        groups, labels = self.group_data()
        audio_data = defaultdict(list)
        encoder = self.get_encoder_model()

        for sample, list_channel_file in groups.items():
            for audio_file in list_channel_file:
                spec = Spectrogram(audio_file)

                spec_channel = spec.wav2spec()
                spec_channel = np.array(spec_channel)
                spec_channel = np.expand_dims(spec_channel, axis=0)

                embedding = encoder.predict(spec_channel)
                embedding = np.squeeze(embedding, axis=0)


                audio_data[sample].append(embedding)
                #print(audio_file)
                
        return audio_data, labels

    def stacked_spectrogram(self):
        audio_data, labels = self.dataset()
        stacked_audio_spectrograms = {}

        count = 0

        for s, list_channel_data in audio_data.items():
            #count += 1
            #print(s)
            stft_channel1 = list_channel_data[1]
            stft_channel2 = list_channel_data[2]
            stft_channel3 = list_channel_data[3]
            stft_channel4 = list_channel_data[4]

            #print(count)

            stacked_data = np.stack([stft_channel1, stft_channel2, stft_channel3, stft_channel4], axis=-1)
            stacked_audio_spectrograms[s] = stacked_data

        return stacked_audio_spectrograms, labels

    def generate_training_data(self):
        stacked_audio_spectrograms, labels = self.stacked_spectrogram()
        X, Y = [], []
        for s, stacked_spectrogram in stacked_audio_spectrograms.items():
            X.append(stacked_spectrogram)
        for sample, label in labels.items():
            Y.append(label)

        return X, Y 




    






    
    











