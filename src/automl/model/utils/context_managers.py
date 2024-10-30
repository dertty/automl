import os


class SuppressWarnings:
    def __enter__(self):
        os.environ["PYTHONWARNINGS"] = "ignore"
        return

    def __exit__(self, type, value, traceback):
        os.environ["PYTHONWARNINGS"] = "default"
        return
