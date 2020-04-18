import os
import logging
import librosa
import datetime
import tensorflow as tf
import keras
import numpy as np

from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import Callback

from settings import CLASSES_MAP, MODEL_JSON, MODEL_H5, CLASSES, \
                                    MODEL_DIR, LOG_DIR_TRAINING
from src.metrics import *
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.model')

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)

class CNN(object):
    def __init__(self, most_shape):
        logger.info("Initializing CNN")
        self.model = Sequential()
        self.input_shape=most_shape + (1,)
        logger.info(f"Input shape = {self.input_shape}")
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
        logger.info("CNN Initialized")


    def __str__(self):
        return str(self.model.summary())

    def train(self, X_train, y_train, X_test, y_test):
        logger.info("Start training model")
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=['accuracy', precision, recall,fmeasure])

        # TensorBoard Logging
        log_dir = os.path.join(LOG_DIR_TRAINING, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        logger.info("Tensorboard Logging Started")
        logger.info("Use the following command in the terminal to view the logs during training: tensorboard --logdir logs/training")

        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=67,
            batch_size=20,
            validation_data= (X_test, y_test),
            callbacks=[tensorboard_callback, LoggingCallback(logger.info)])

        logger.info("Training completed")

    def evaluate(self, X_train, y_train, X_test, y_test):
        logger.info("Evaluating model")
        self.score_test = self.model.evaluate(
            x=X_test,
            y=y_test)

        self.score_train = self.model.evaluate(
            x=X_train,
            y=y_train)

        logger.info(f'Train loss: {self.score_train[0]}')
        logger.info(f'Train accuracy: {self.score_train[1]}')
        logger.info(f'Train precision: {self.score_train[2]}')
        logger.info(f'Train recall: {self.score_train[3]}')
        logger.info(f'Train f1-score: {self.score_train[4]}')

        logger.info(f'Test loss: {self.score_test[0]}')
        logger.info(f'Test accuracy: {self.score_test[1]}')
        logger.info(f'Test precision: {self.score_test[2]}')
        logger.info(f'Test recall: {self.score_test[3]}')
        logger.info(f'Test f1-score: {self.score_test[4]}')

    @staticmethod
    def get_class(self, class_ID):
        return list(CLASSES_MAP.keys())[list(CLASSES_MAP.values()).index(class_ID)]

    def save_model(self):
        logger.info('Saving model')
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_JSON, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_H5)
        logger.info("Saved model to "+MODEL_DIR)

    def load_model(self):
        logger.info('Loading saved model')
         # load json and create model
        try:
            with open(MODEL_JSON, "r") as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(MODEL_H5)
            loaded_model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=['accuracy', precision, recall, fmeasure])
            self.model = loaded_model
            logger.info('Model loaded from '+MODEL_DIR)
        except:
            logger.info("Model not found")

    def predict(self, filepath, loadmodel=True):
        logger.info('Prediction')
        if loadmodel:
            # self.load_model()
            pass
        else:
            # try:
                y, sr = librosa.load(filepath, duration=2)
                ps = librosa.feature.melspectrogram(y=y, sr=sr,)
                px = ps
                px
                shape = (1,) + self.input_shape
                ps = np.array(ps.reshape(shape))
                predictions = self.model.predict_classes(ps)
                class_id = predictions[0]
                chord = str(CLASSES[class_id])
                logger.info("The recorded chord is " + chord)
            # except:
                # logger.info("File note found")
                # chord = "N/A"
        return chord

# cnn = CNN((128, 87))