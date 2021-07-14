import librosa 
import librosa.display
import wave 
from pydub import AudioSegment
from pydub.utils import make_chunks

import numpy as np 

import os
import glob 
import ntpath 
from collections import defaultdict 



# Remove path address and retain .wav audio file name


audio_dir = './data/exp2/alarm_conveyerbelt'

def path_leaf(path):
    ntpath.basename("a/b/c/d")
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def sample_audio_one_second(audio_dir):
    groups = defaultdict(list)
    
    for filename in sorted(glob.glob(os.path.join(audio_dir, '*.wav'))):
        audio_file_name = path_leaf(filename)
        #print(audio_file_name)
        basename, extension = os.path.splitext(audio_file_name)

        sample, x_robot, y_robot, orientation_robot, channel_number = basename.split('_')
        groups[sample].append(filename)
    
    return groups


def sample_audio_file(groups):
    count = 0
    for sample, list_channel_file in groups.items():
        for audio_file in list_channel_file:
            myaudio = AudioSegment.from_file(audio_file, 'wav')
            chunk_length_ms = 1000 # pydub calculates in millisec 
            chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 

            #get file name 
            file_name_channel = path_leaf(audio_file)
            basename, extension = os.path.splitext(file_name_channel)
            sample, x_robot, y_robot, orientation_robot, channel_number = basename.split('_')

            for i, chunk in enumerate(chunks):

                if i < 25:

                    iterator = count*25 + i 
                    chunk_name = "{iterator}_{x_robot}_{y_robot}_{orientation_robot}_{channel_number}.wav".format(iterator=iterator, x_robot=x_robot, 
                                                    y_robot=y_robot, orientation_robot=orientation_robot, channel_number=channel_number)

                    print("exporting", chunk_name)
                    chunk.export(chunk_name, format="wav")
        count = count + 1
        print(count)
        print(iterator)
            

groups = sample_audio_one_second(audio_dir)
print(len(groups))
sample_audio_file(groups)






