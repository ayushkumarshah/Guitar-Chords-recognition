import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_AUDIO = os.path.join(DATA_DIR, 'audio')
DATA_DIR_GUITAR = os.path.join(DATA_DIR_AUDIO, 'Guitar_Only/')
DATA_DIR_AUGMENTED = os.path.join(DATA_DIR_AUDIO, 'augmented')

METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
METADATA_DIR_RAW = os.path.join(METADATA_DIR, 'raw')
METADATA_DIR_PROCESSED = os.path.join(METADATA_DIR, 'processed')

METADATA_DIR_AUGMENTED = os.path.join(METADATA_DIR, 'augmented')
METADATA_DIR_AUGMENTED_RAW = os.path.join(METADATA_DIR_AUGMENTED, 'raw')
METADATA_DIR_AUGMENTED_PROCESSED = os.path.join(METADATA_DIR_AUGMENTED, 'processed')

LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR_TRAINING = os.path.join(LOG_DIR, 'training')

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
CLASSES_MAP = {'a':0, 'am':1, 'bm':2, 'c':3, 'd':4, 'dm':5, 'e':6, 'em':7, 'f':8, 'g':9}

# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100   # Default sample rate of microphone or recording device
DURATION = 3   # 3 seconds
CHUNK_SIZE = 1024