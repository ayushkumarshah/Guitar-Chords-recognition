import librosa
import numpy as np
import pandas as pd

from settings import *
from src.data.preprocessing import df_info, add_shape, clean_shape, over_sample,get_count
from src.data import generate

@df_info
def augment_data(df, kind='time', rate=1.07):
    if kind == 'time':
        df['y'] = df['y'].map(lambda y:librosa.effects.time_stretch(y, rate=rate))
        new_path = 'speed_' + str(int(rate*100))
    elif kind == 'pitch':
        df['y'] = df.apply(lambda row:librosa.effects.pitch_shift(row['y'], row['sr'], n_steps=rate), axis=1)
        new_path = 'pitch_' + str(int(rate*100))
    df['file_path'] = df.apply(lambda row: row['class_name']+'/'+new_path\
                                                      +"/"+row['file_name'], axis=1)
    for class_name in CLASSES:
        directory = os.path.join(DATA_DIR_AUGMENTED,class_name,new_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    path_exists_mask =  df['file_path'].map(lambda x: not os.path.exists(os.path.join(DATA_DIR_AUGMENTED, x)))
    df[path_exists_mask].apply(lambda row: librosa.output.write_wav(os.path.join(DATA_DIR_AUGMENTED, row['file_path']),\
                                                        row['y'], row['sr']), axis=1)
    df['spectrogram'] = df.apply(lambda row: librosa.feature.melspectrogram(y=row['y'], \
                                                sr=row['sr']),axis=1)
    df['augmentation'] = new_path
    return df

def get_augmentation_count(df):
    return df['augmentation'].value_counts()

@df_info
def process_augmented(df):
    df = (df.pipe(clean_shape)
                .pipe(over_sample))
    df = df[['spectrogram','class_ID', 'class_name']]
    return df

def main():
    if not os.path.exists(os.path.join(METADATA_DIR_RAW,'data.pkl')):
        generate.run()
    data_df_raw = pd.read_pickle(os.path.join(METADATA_DIR_RAW,'data.pkl'))
    data_df_time_inc = augment_data(data_df_raw.copy(), kind='time', rate=1.07)
    data_df_time_dec = augment_data(data_df_raw.copy(), kind='time', rate=0.81)
    data_df_shift_20 = augment_data(data_df_raw.copy(), kind='pitch', rate=2.5)
    data_df_shift_25 = augment_data(data_df_raw.copy(), kind='pitch', rate=2)
    data_df_raw['augmentation'] = 'None'
    data_df_augmented_raw = pd.concat([data_df_raw, data_df_time_inc, \
         data_df_time_dec, data_df_shift_20, data_df_shift_25], axis=0)
    data_df_augmented_raw = data_df_augmented_raw.pipe(add_shape)
    print(get_augmentation_count(data_df_augmented_raw))
    print(get_count(data_df_augmented_raw))

    data_df_augmented_raw.to_csv(os.path.join(METADATA_DIR_AUGMENTED_RAW, 'data.csv'), index=False)
    data_df_augmented_raw.to_pickle(os.path.join(METADATA_DIR_AUGMENTED_RAW, 'data.pkl'))

    data_df_augmented_processed = process_augmented(data_df_augmented_raw)
    data_df_augmented_processed.to_csv(os.path.join(METADATA_DIR_AUGMENTED_PROCESSED, 'data.csv'), index=False)
    data_df_augmented_processed.to_pickle(os.path.join(METADATA_DIR_AUGMENTED_PROCESSED, 'data.pkl'))

    print(get_count(data_df_augmented_processed))

if __name__ == '__main__':
    main()