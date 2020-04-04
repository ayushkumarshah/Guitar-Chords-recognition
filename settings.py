import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_AUDIO = os.path.join(DATA_DIR, 'audio')
DATA_DIR_AUGMENTED = os.path.join(DATA_DIR, 'augmented')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_JSON = os.path.join(MODEL_DIR, 'model.json')
MODEL_H5 = os.path.join(MODEL_DIR, 'model.h5')


OUT_DIR = os.path.join(ROOT_DIR, 'output/')
RECORDING_DIR = os.path.join(OUT_DIR, 'recording')
IMAGE_DIR = os.path.join(OUT_DIR, 'images')

WAVE_OUTPUT_FILE = os.path.join(RECORDING_DIR, "recorded.wav")
SPECTROGRAM_FILE = os.path.join(RECORDING_DIR, "spectrogram.png")

# Features #################
CLASSES = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']

# # Model Config base
# MODEL_CONFIG_DIR = os.path.join(ROOT_DIR, 'training_config')
# DEFAULT_MODEL_CONFIG = os.path.join(MODEL_CONFIG_DIR, 'nfm.json')
# SAMPLING_CONFIG = os.path.join(MODEL_CONFIG_DIR, 'sampling.json')

