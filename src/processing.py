import logging
import keras
import numpy as np
import pandas as pd
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.processing')

def train_test_split(dataset, augmented=True, split_ratio=0.65):
    logger.info(f"Start train test split with split ratio: {split_ratio}")
    np.random.seed(42)
    sample = np.random.choice(dataset.index, size=int(len(dataset) * split_ratio), replace=False)
    if augmented:
        train_data = dataset.iloc[sample]
        test_data = dataset.drop(sample)
        test_data = test_data[test_data['augmentation']=='None']
    else:
        train_data, test_data = dataset.iloc[sample], dataset.drop(sample)
    logger.info(f"Number of training samples is {len(train_data)}")
    logger.info(f"Number of testing samples is {len(test_data)}")
    logger.info(f"Train test split completed")
    return train_data, test_data

def features_target_split(data):
    logger.info(f"Start feature target split")
    feature = data['spectrogram']
    target = data['class_ID']
    logger.info(f"Feature target split completed")
    return feature, target

def reshape_feature_CNN(features):
    logger.info(f"Features reshaped for CNN Input")
    return np.array([feature.reshape( (128, 87, 1) ) for feature in features])

def one_hot_encode(target):
    logger.info(f"Target one hot encoded")
    return np.array(keras.utils.to_categorical(target, 10))