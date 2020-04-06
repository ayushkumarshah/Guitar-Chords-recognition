import random
import keras
import os, glob
import logging
import librosa, librosa.display

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras import backend as K

from src.metrics import *
from settings import *
from src.data import generate, augment
from src.processing import *
from src.model import CNN
from src.data.preprocessing import get_most_shape
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.train')

def main():
    logger.info("Start Training Pipeline")
    augmented = True
    if augmented:
        if not os.path.exists(os.path.join(METADATA_DIR_AUGMENTED_PROCESSED, 'data.pkl')):
            augment.main()
             # Read Data
        dataset = pd.read_pickle(os.path.join(METADATA_DIR_AUGMENTED_PROCESSED, 'data.pkl'))

    else:
        # Generate MetaData if not generated yet
        if not os.path.exists(os.path.join(METADATA_DIR_PROCESSED,'data.pkl')):
            generate.run()
        dataset = pd.read_pickle(os.path.join(METADATA_DIR_PROCESSED, 'data.pkl'))

    logger.info(f"Number of samples: {len(dataset)}")
    most_shape = get_most_shape(dataset)
    train_data, test_data = train_test_split(dataset, split_ratio=0.8)

    X_train, y_train = features_target_split(train_data)
    X_test, y_test = features_target_split(test_data)

    # Reshape for CNN input
    X_train, X_test = reshape_feature_CNN(X_train), reshape_feature_CNN(X_test)

    # Preserve y_test values
    y_test_values = y_test.copy()

    # One-Hot encoding for classes
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)

    # Instance of CNN model
    cnn = CNN(most_shape)
    logger.info(cnn)

    cnn.train(X_train, y_train, X_test, y_test)
    cnn.evaluate(X_train, y_train, X_test, y_test)

    predictions = cnn.model.predict_classes(X_test)
    conf_matrix=confusion_matrix(y_test_values, predictions, labels=range(10))
    logger.info(conf_matrix)
    cnn.save_model()

if __name__ == '__main__':
    main()