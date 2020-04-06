import os, glob
import logging
import librosa, librosa.display
import pandas as pd

from settings import *
from src.data.preprocessing import *
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.data.generate')

def run():
    # Generate raw data
    logger.info("Generating Raw Metadata")
    data_df_raw = (pd.DataFrame().pipe(construct_dataframe)
                                .pipe(get_spectrogram)
                                .pipe(add_shape)
                )
    logger.info("Raw Metadata Generated")

    # Save raw data
    data_df_raw.to_csv(os.path.join(METADATA_DIR_RAW, 'data.csv'), index=False)
    data_df_raw.to_pickle(os.path.join(METADATA_DIR_RAW, 'data.pkl'))

    logger.info("Raw Metadata saved to"+METADATA_DIR_RAW)
    logger.info(get_count(data_df_raw))

    # Process and save processed data
    data_df_processed = process(data_df_raw)

    data_df_processed.to_csv(os.path.join(METADATA_DIR_PROCESSED, 'data.csv'), index=False)
    data_df_processed.to_pickle(os.path.join(METADATA_DIR_PROCESSED, 'data.pkl'))

    logger.info("Processed Metadata saved to"+METADATA_DIR_PROCESSED)
    logger.info(get_count(data_df_processed))

if  __name__ =='__main__':
    run()