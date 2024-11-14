from datetime import datetime
from pathlib import Path

PATH = Path().resolve() / "ml_data" / datetime.now().strftime("%Y_%m_%d___%H_%M_%S")


def create_ml_data_dir():
    PATH.mkdir(parents=True, exist_ok=True)
    
def is_ml_data_dir_exists():
    """Checks whether ml data dir for the program exists.

    Returns:
        bool
    """
    return PATH.exists()


# def is_child():
#     parent_pid = os.getppid()
#     current_pid = os.getpid()

#     return parent_pid != 1 and parent_pid == current_pid

# if __name__ == '__main__':
#     if is_child():
#         print("Я дочерний процесс")
#     else:
#         print("Я родительский процесс")
