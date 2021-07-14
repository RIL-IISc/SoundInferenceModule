import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input

def get_cnn_model(input_shape):
    print("input_shape: " + str(input_shape))
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    #model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    
    #model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.25))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='linear'))
    #model.compile(loss=tensorflow.keras.losses.mean_squared_error, optimizer=tensorflow.keras.optimizers.Adam(0.009))
    model.compile(loss=tensorflow.keras.losses.mean_squared_error, optimizer=tensorflow.keras.optimizers.RMSprop(0.005))

    return model
