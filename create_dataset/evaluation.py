import librosa 
import librosa.display
import wave

import numpy as np

import os 
import glob 
import ntpath
from collections import defaultdict 

# Remove path address and retail wav file name

class Create_dataset:
    def __init__(self, audio_file_dir, x_sound, y_sound):
        self.audio_file_dir = audio_file_dir
        self.x_sound = x_sound
        self.y_sound = y_sound

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

    def generate_labels(self):
        groups, labels = self.group_data()
        Y = []
        for sample, label in labels.items():
            Y.append(label)
        return Y 
