from loguru import logger

logger.add("app.log", rotation="10 MB")


def get_logger():
    return logger
