import os, glob
import librosa, librosa.display
import pandas as pd

from settings import *
from src.data.preprocessing import *

def run():
    # Generate raw data
    file_path = glob.glob(DATA_DIR_GUITAR + "**/*.wav")
    data_df_raw = (pd.DataFrame().pipe(construct_dataframe(file_path))
                                .pipe(get_spectrogram)
                                .pipe(add_shape)
                )

    # Save raw data
    data_df_raw.to_csv(os.path.join(METADATA_DIR_RAW, 'data.csv'), index=False)
    print(get_count(data_df_raw))

    # Process and save processed data
    data_df_processed = process(data_df_raw)
    data_df_processed.to_csv(os.path.join(METADATA_DIR_PROCESSED, 'data.csv'), index=False)
    data_df_processed.to_pickle(os.path.join(METADATA_DIR_PROCESSED, 'data.pkl'))

    print(get_count(data_df_processed))

if  __name__ =='__main__':
    run()