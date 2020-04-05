import librosa
import tensorflow as tf

from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.models import model_from_json

from settings import CLASSES_MAP, MODEL_JSON, MODEL_H5, CLASSES
from src.metrics import *

class CNN(object):

    def __init__(self, most_shape):
        self.model = Sequential()
        self.input_shape=most_shape + (1,)
        self.model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=self.input_shape))
        self.model.add(MaxPooling2D((4, 2), strides=(4, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(48, (5, 5), padding="valid"))
        self.model.add(MaxPooling2D((4, 2), strides=(4, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(48, (5, 5), padding="valid"))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

    def __str__(self):
        return str(self.model.summary)

    def train(self, X_train, y_train, X_test, y_test):
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=['accuracy', precision, recall,fmeasure])

        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=70,
            batch_size=20,
            validation_data= (X_test, y_test))

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.score_test = self.model.evaluate(
            x=X_test,
            y=y_test)

        self.score_train = self.model.evaluate(
            x=X_train,
            y=y_train)

        print('Train loss:', self.score_train[0])
        print('Train precision:', self.score_train[2])
        print('Train recall:', self.score_train[3])
        print('Train f1-score:', self.score_train[4])

        print('Test loss:', self.score_test[0])
        print('Test precision:', self.score_test[2])
        print('Test recall:', self.score_test[3])
        print('Test f1-score:', self.score_test[4])

    @staticmethod
    def get_class(self, class_ID):
        return list(CLASSES_MAP.keys())[list(CLASSES_MAP.values()).index(class_ID)]

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_JSON, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_H5)
        print("Saved model to disk")

    def load_model(self):
         # load json and create model
        try:
            with open(MODEL_JSON, "r") as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(MODEL_H5)
            print("Loaded model from disk")
            loaded_model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=['accuracy', precision, recall, fmeasure])
            self.model = loaded_model
        except:
            print("Model not found")

    def predict(self, filepath):
        self.load_model()
        try:
            y, sr = librosa.load(filepath, duration=2)
            ps = librosa.feature.melspectrogram(y=y, sr=sr,)
            px = ps
            shape = + (1,) + self.input_shape
            ps = np.array(ps.reshape(shape))
            predictions = self.model.predict_classes(ps)
            class_id = predictions[0]
            print("The recorded chord is "+str(CLASSES[class_id]))
        except:
            print("File note found")