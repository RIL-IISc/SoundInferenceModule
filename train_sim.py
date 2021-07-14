from tensorflow.tools.docs.doc_controls import T
from create_dataset.create_dataset import Create_dataset
from processing.cnn_architecture import get_cnn_model

from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt

f_alarm_cobot = './data/alarm_cobot'
f_alarm_conveyer_belt = './data/alarm_conveyer_belt'
f_yasakawa = './data/alarm_yasakawa'
f_alarm_conveyer_tray = './data/alarm_conveyer_tray'
f_test_1 = './data/test_1'
f_test_2 = './data/test_2'



def get_absolute_coordinates(x_relative, y_relative, x_robot, y_robot, orientation_robot):
    if orientation_robot == 0:
        x_sound = x_relative + x_robot
        y_sound = y_relative + y_robot

    elif orientation_robot == 90:
        x_sound = -y_relative + x_robot
        y_sound = x_relative + y_robot

    elif orientation_robot == 180:
        x_sound = -x_relative + x_robot
        y_sound = -y_relative + y_robot

    elif orientation_robot == 270:
        x_sound = y_relative + x_robot
        y_sound = -x_relative + y_robot
    
    return x_sound, y_sound 


if __name__ == '__main__':

    dataset_cobot =  Create_dataset(f_alarm_cobot, 7, 4)
    dataset_conveyer = Create_dataset(f_alarm_conveyer_belt, 4, 1)
    dataset_yasakawa = Create_dataset(f_yasakawa, 0, 1)
    dataset_conveyer_tray = Create_dataset(f_alarm_conveyer_tray, 6, 2)
    dataset_test_1 = Create_dataset(f_test_1, 4, 1)
    dataset_test_2 = Create_dataset(f_test_2, 10, 3)


    x1, y1 = dataset_cobot.generate_training_data()
    x2, y2 = dataset_conveyer.generate_training_data()
    x3, y3 = dataset_yasakawa.generate_training_data()
    x4, y4 = dataset_conveyer_tray.generate_training_data()
    x_test_1, y_test_1 = dataset_test_1.generate_training_data()
    x_test_2, y_test_2 = dataset_test_2.generate_training_data()

    print('No. of samples in CobotRegion, ConveyerRegion, and Yasakawa:', len(x1), len(x2), len(x3))
    print(x1[0].shape, x2[0].shape, x3[0].shape)

    input_shape = x1[0].shape

    X, y = [], []
    X = x1 + x2 + x3 
    y = y1 + y2 + y3

    X = np.array(X)
    y = np.array(y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    x_test_1 = np.array(x_test_1)
    y_test_1 = np.array(y_test_1)

    x_test_2 = np.array(x_test_2)
    y_test_2 = np.array(y_test_2)



    model = get_cnn_model(input_shape)

    history = model.fit(X , y, epochs = 100, shuffle=True)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    



    y_pred_1 = model.predict(x_test_1)

    print(x_test_1[0].shape)

    print('x_relative and y_relative are as follows (ground truth):', y_test_1)
    print('_______________________________________________')


    print('x_relative_prediction and y_relative_prediction are as follows:', y_pred_1)