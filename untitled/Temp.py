import cv2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model


import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.misc import imread, imsave, imresize


class opencv:
    def __init__(self):
        self.face_pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_arr = []
        self.stop = 0
        self.camera()

    def camera(self):
        self.cap = cv2.VideoCapture(0)

        a = 0
        # 학습에 사용할 사진 개수
        while(a < 1):

            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceList = self.face_pattern.detectMultiScale(gray)
            for (x, y, w, h) in faceList:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                print(a)
                frame = frame[y: y+h, x: x + w]
                try:
                    self.face_arr.append(frame)
                    cv2.imshow('image', frame)
                except:
                    self.face_arr.pop()
                    print('Error')
                else:
                    a += 1
                if a % 10 == 0:
                    try:
                        self.img_input(frame)
                    except:
                        print('Not found model')

        self.cap.release()
        cv2.destroyAllWindows()

        input_food = 'gamja'
        if input_food == "gamja":
            label = [1, 0, 0, 0]
        if input_food == "goguma":
            label = [0, 1, 0, 0]
        if input_food == "apple":
            label = [0, 0, 1, 0]
        if input_food == "water":
            label = [0, 0, 0, 1]

        # 1(기존 모델 업데이트), 0(새 모델 만들기)

        if os.path.exists('./model/model.h5'):
            self.model(1, label)
        else:
            self.model(0, label)


    def model(self, update, label):
        x_train = []
        num_classes = 4

        for i in self.face_arr:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            x_train.append(imresize(i, (128, 128)))

        x_train = np.array(x_train)
        x_train = x_train.astype('float32')
        x_train /= 255.0
        x_train = np.expand_dims(x_train, axis=4)

        y_label = np.ones((len(self.face_arr), num_classes))


        for i in range(len(self.face_arr)):
            y_label[i] = label
        print(np.shape(x_train), np.shape(y_label))
        f = open('cnt.csv', 'r', encoding='utf-8')
        cnt = 0
        rdr = csv.reader(f)
        for line in rdr:
            cnt = int(line[0])
            cnt += 1
        f.close()

        input_shape = x_train[0].shape
        print(input_shape)

        if update == 0:
            model = Sequential()
            model.add(Conv2D(64, (8, 8), input_shape=input_shape, padding='same', activation='relu',
                         kernel_constraint=maxnorm(3)))
            model.add(Dropout(0.5))
            model.add(Conv2D(128, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            # Compile model
            epochs = 5
            lrate = 0.0003
            decay = lrate / epochs
            sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            print(model.summary())

            model_json = model.to_json()
            with open("./model/model.json", "w") as json_file:
                json_file.write(model_json)


            # Fit the model
            model.fit(x_train, y_label, epochs=epochs, batch_size=32)

            model.save_weights("./model/model.h5")
        else:
            json_file = open('./model/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("./model/model.h5")

            # Compile model
            epochs = 5
            lrate = 0.0006
            decay = lrate / epochs
            sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
            loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            print(loaded_model.summary())

            model_json = loaded_model.to_json()
            with open("./model/model.json", "w") as json_file:
                json_file.write(model_json)

            # Fit the model
            loaded_model.fit(x_train, y_label, epochs=epochs, batch_size=32)

            loaded_model.save_weights("./model/model.h5")


    def img_input(self, img):
        json_file = open('./model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./model/model.h5")
        print("Loaded model from disk")

        img = imresize(img, (128, 128))
        faceAligned = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceAligned = np.array(faceAligned)
        faceAligned = faceAligned.astype('float32')
        faceAligned /= 255.0
        faceAligned= np.expand_dims([faceAligned], axis=4)

        food = ['gamja', 'goguma', 'apple', 'water']

        Y_pred = loaded_model.predict(faceAligned)
        print(Y_pred)
        for index, value in enumerate(Y_pred[0]):
            result = food[index] + ': ' + str(int(value * 100)) + '%'
            print(result)


opencv()