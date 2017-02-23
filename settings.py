import os
import configparser
import logging
from util.logger import create_logger

setting_logger = create_logger('settings', logging.DEBUG, 'settings.log')
dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    # app_config.ini is a symbolic link to the config file
    config = configparser.ConfigParser()
    config.read(os.path.join(dir_path, 'app_config.ini'))

    IMG_PATH = config["file_path"]["img_path"]
    MODEL_DIR = config["file_path"]["model_dir"]
except Exception as e:
    setting_logger.exception(e)
