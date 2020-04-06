import logging
import logging.config
import yaml
import time
from logging import Formatter

def setup_logging():
    with open('logging.yml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

class GMTFormatter(Formatter):
    """Formatter that converts time to GMT
    """
    converter = time.gmtime