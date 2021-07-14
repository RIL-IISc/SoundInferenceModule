from create_dataset.create_dataset_autoencoder import Create_dataset
import matplotlib.pyplot as plt
import librosa

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.models import Model

import cv2 

f_audio = './final_data'

if __name__ == '__main__':

    dataset_autoencoder = Create_dataset(f_audio)

    data_train = dataset_autoencoder.dataset()
    print(data_train[0])
   
    print("length of dataset:", len(data_train))

    print(data_train[0].shape, len(data_train))

    dim = (64,160)

    

    resize_dataset = []
    for i in data_train:
        resize_dataset.append(cv2.resize(i, dim))


    #shape_spec = (160, 63, 1)
 
    print(resize_dataset[0].shape, len(resize_dataset))

    print(resize_dataset[0])


    input_img = Input(shape=(160, 64, 1))

    # Build the encoder

    encoded = Conv2D(128, kernel_size=(3, 3), input_shape=(160, 63, 1), padding='same')(input_img)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)

    encoded = Conv2D(64, kernel_size=(3, 3), padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)

    encoded = Conv2D(32, kernel_size=(3, 3), padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)

    encoded = Conv2D(16, kernel_size=(3, 3), padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)


    encoded = Conv2D(8, kernel_size=(3, 3), padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)

    encoder = Model(inputs=input_img, outputs=encoded)

    encoder.summary()

    encoded_shape = encoder.layers[-1].output.shape

    # Build the decoder

    decoded = Conv2D(8, kernel_size=(3, 3), input_shape=encoded_shape, padding='same')(encoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    #decoded = BatchNormalization()(decoded)
    decoded = UpSampling2D((2,2))(decoded)

    decoded = Conv2D(16, kernel_size=(3, 3), padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    #decoded = BatchNormalization()(decoded)
    decoded = UpSampling2D((2,2))(decoded)

    decoded = Conv2D(32, kernel_size=(3, 3), padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    #decoded = BatchNormalization()(decoded)
    decoded = UpSampling2D((2,2))(decoded)

    decoded = Conv2D(64, kernel_size=(3, 3), padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    #decoded = BatchNormalization()(decoded)
    decoded = UpSampling2D((2,2))(decoded)

    decoded = Conv2D(128, kernel_size=(3, 3), padding='same')(decoded)
    decoded = UpSampling2D((2,2))(decoded)

    decoded = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(decoded)

    decoder = Model(inputs=input_img, outputs=decoded)
    
    decoder.compile(optimizer= 'adam', loss = 'binary_crossentropy')
    decoder.summary()

    resize_dataset = np.array(resize_dataset)

    
    decoder.fit(resize_dataset, resize_dataset,
                 epochs=50,
                 batch_size=128, shuffle=True) 
    
    decoder.save('./model/autoencoder.h5')
    


















    




    

    