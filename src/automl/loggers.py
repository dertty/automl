import logging
import sys
from datetime import datetime

import optuna

from .constants import PATH, is_ml_data_dir_exists


def center_text(text: str, max_len: int = 12):
    """Pads the text with whitespaces to center it."""
    pad_right = (max_len - len(text)) // 2
    pad_left = max_len - len(text) - pad_right
    return " " * pad_right + text + " " * pad_left


_log_format = f"[%(asctime)s] - %(message)s"
msg_types_reprs = {
    "start": center_text("START"),
    "end": center_text("END"),
    "score": center_text("SCORE"),
    "best_params": center_text("BEST PARAMS"),
    "fit": center_text("FIT"),
    "optuna": center_text("OPTUNA"),
    "new_best": center_text("NEW BEST"),
    "best": center_text("BEST  MODEL"),
    "model": center_text("MODEL"),
    "params": center_text("PARAMS")
}

# disable default optuna logs
# custom optuna logs are implemented
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LoggerAdapterWithMessageTypes(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        msg_type = kwargs.pop("msg_type", self.extra["msg_type"])

        if msg_type is not None:
            return "[%s] - %s" % (msg_types_reprs[msg_type], msg), kwargs

        return "%s" % (msg), kwargs


class LoggingStream:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = ""
        self.first_line_flag = True

    def write(self, message):
        if self.first_line_flag:
            self.logger.error(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
            self.first_line_flag = False

        self.buffer += message

        if self.buffer.endswith("\n"):
            self.logger.error(self.buffer[:-1])
            self.buffer = ""

    def flush(self):
        pass


def get_info_file_handler():
    file_handler = logging.FileHandler(PATH / "logs.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    file_handler.addFilter(lambda record: record.levelno == logging.INFO)
    return file_handler


def get_error_file_handler():
    file_handler = logging.FileHandler(PATH / "error.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    return file_handler


def get_info_stream_handler():
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    stream_handler.addFilter(lambda record: record.levelno == logging.INFO)
    return stream_handler


def get_error_stream_handler():
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    return stream_handler


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # add stdout handlers
    logger.addHandler(get_info_stream_handler())
    logger.addHandler(get_error_stream_handler())
    
    if is_ml_data_dir_exists():
        # add file handlers
        logger.addHandler(get_info_file_handler())
        logger.addHandler(get_error_file_handler())

    logger = LoggerAdapterWithMessageTypes(logger, {"msg_type": None})

    # redirect std:err to logger
    sys.stderr = LoggingStream(logger)

    return logger


# def configure_root_logger():
#     # for LightAutoML logs
#     root_logger = logging.getLogger()
#     root_logger.addHandler(get_info_file_handler())
#     root_logger.addHandler(get_error_file_handler())