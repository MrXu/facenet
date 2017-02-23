# coding: utf-8

import logging

loggers = {}


def create_logger(logger_name, level, filename):
    logger = loggers.get(logger_name)
    if logger:
        return logger

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("./log/"+filename)
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    loggers[logger_name] = logger
    return logger
