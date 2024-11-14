import os
import sys


class SuppressWarnings:
    def __enter__(self):
        os.environ["PYTHONWARNINGS"] = "ignore"
        return

    def __exit__(self, type, value, traceback):
        os.environ["PYTHONWARNINGS"] = "default"
        return


class LoggerWriter:
    """Logger with stdout interface."""

    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message != "\n":
            self.logger.info(message[:-1])

    def flush(self):
        pass


class CatchLamaLogs:
    """
    Context manager that redirects stdout to logger while LightAutoML is working.
    This allows to save LightAutoML logs in file.
    """

    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.save_stdout = sys.stdout
        sys.stdout = LoggerWriter(self.logger)

    def __exit__(self, type, value, traceback):
        sys.stdout = self.save_stdout
