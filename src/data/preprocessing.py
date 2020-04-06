import os, glob
import logging
import librosa, librosa.display
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from settings import *
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.data.preprocessing')

def df_info(f):
    def inner(df, *args, **kwargs):
        result = f(df, *args, **kwargs)
        logger.info(f"After applying {f.__name__}, shape of df = {result.shape }")
        logger.info(f"Columns of df are {df.columns}\n")
        return result
    return inner

@df_info
def construct_dataframe(df):
    """
    Construct Dataframe with all required values
    """
    logger.info("Construct DataFrame for raw data")
    file_path = glob.glob(DATA_DIR_GUITAR + "**/*.wav")
    df['file_path'] = file_path
    df['file_path'] = df['file_path'].map(lambda x: x[x.rindex('Only/')+len('Only/'):])
    df['file_name'] = df['file_path'].map(lambda x: x[x.rindex('/')+1:])
    df['class_name'] = df['file_path'].map(lambda x: x[:x.index('/')])
    df['class_ID'] = df['class_name'].map(lambda x: CLASSES_MAP[x])
    logger.info("Construct DataFrame for raw data completed")
    return df.copy()

@df_info
def get_spectrogram(df):
    logger.info("Extract spectrogram")
    """Extract spectrogram from audio"""
    df['audio_series'] = df['file_path'].map(lambda x: librosa.load(DATA_DIR_GUITAR \
                                                                    + x, duration=2))
    df['y'] = df['audio_series'].map(lambda x: x[0])
    df['sr'] = df['audio_series'].map(lambda x: x[1])
    df['spectrogram'] = df.apply(lambda row: librosa.feature.melspectrogram(y=row['y'],\
         sr=row['sr']), axis=1)
    df.drop(columns='audio_series', inplace=True)
    logger.info("Extract spectorgram completed")
    return df

@df_info
def add_shape(df):
    df['shape'] = df['spectrogram'].map(lambda x: x.shape)
    return df

def get_most_shape(df):
    most_shape = df['spectrogram'].map(lambda x: x.shape).value_counts().index[0]
    print(f"The most frequent shape is {most_shape}")
    return most_shape

# Maintain same shape
@df_info
def clean_shape(df):
    logger.info("Filter data using most frequent shape")
    most_shape = get_most_shape(df)
    df = df[df['shape']==most_shape]
    df.drop(columns='shape', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Filter data using most frequent shape completed")
    return df

# Create processed dataframe
@df_info
def process(df):
    logger.info("Process raw data")
    df = (df.pipe(clean_shape)
                .pipe(over_sample)
    )
    df = df[['spectrogram','class_ID', 'class_name']]
    logger.info("Process raw data completed")
    return df

def get_count(df):
    return df['class_name'].value_counts()

def get_class(class_ID):
    return list(CLASSES_MAP.keys())[list(CLASSES_MAP.values()).index(class_ID)]

@df_info
def over_sample(df):
    logger.info("Oversample data to balance classes")
    oversample = RandomOverSampler(sampling_strategy='auto')
    X, y = df['spectrogram'].values, df['class_ID'].values
    X = X.reshape(-1, 1)
    X, y = oversample.fit_resample(X, y)
    df = pd.DataFrame()
    df['spectrogram'] = pd.Series([np.array(x[0]) for x in X])
    df['class_ID'] = pd.Series(y)
    df['class_name'] = df['class_ID'].map(lambda x: get_class(x))
    logger.info("Oversample data to balance classes completed")
    return df