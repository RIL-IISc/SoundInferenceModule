import librosa 
import librosa.display
import wave

import numpy as np

import os 
import glob 
import ntpath
from collections import defaultdict 

from processing.wav2spec import Spectrogram 


# Remove path address and retail wav file name

class Create_dataset:
    def __init__(self, audio_file_dir):
        self.audio_file_dir = audio_file_dir

    def path_leaf(self, path):
        ntpath.basename("a/b/c/d")
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


    def group_data(self):

        dataset_files = []

        for filename in sorted(glob.glob(os.path.join(self.audio_file_dir, '*.wav'))):
            audio_file_name = self.path_leaf(filename)
            basename, extension = os.path.splitext(audio_file_name)
            #print(basename)

            sample, x_robot, y_robot, orientation_robot, channel_number = basename.split('_')

            if channel_number != 'channel0':
                dataset_files.append(filename)

        return dataset_files



    def dataset(self):

        dataset_files = self.group_data()
        dataset = []

        remove_file = []
        
        for file_name in dataset_files:
            
            spec = Spectrogram(file_name)
            #temp = spec.wav2spec()
            dataset.append(spec.wav2spec())
            
            '''
            #if temp == 'error':
            #    remove_file.append(file_name)
            #    print(file_name)
            #    os.remove(file_name)
            #    print("file removed")
            '''
                
        return dataset







    






    
    












