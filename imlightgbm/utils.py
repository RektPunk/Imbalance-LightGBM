import logging


def init_logger() -> logging.Logger:
    logger = logging.getLogger("imlightgbm")
    logger.setLevel(logging.INFO)
    return logger


logger = init_logger()
