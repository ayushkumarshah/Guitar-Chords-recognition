import keras
import numpy as np
import pandas as pd

def train_test_split(dataset, split_ratio=0.8):
    np.random.seed(42)
    sample = np.random.choice(dataset.index, size=int(len(dataset)*0.8), replace=False)
    train_data, test_data = dataset.iloc[sample], dataset.drop(sample)
    print("Number of training samples is", len(train_data))
    print("Number of testing samples is", len(test_data))
    return train_data, test_data

def features_target_split(data):
    feature = data['spectrogram']
    target = data['class_ID']
    return feature, target

def reshape_feature_CNN(features):
    return np.array([feature.reshape( (128, 87, 1) ) for feature in features])

def one_hot_encode(target):
    return np.array(keras.utils.to_categorical(target, 10))