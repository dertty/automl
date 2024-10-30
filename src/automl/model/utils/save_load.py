import yaml


def save_yaml(d, file_name):
    with open(file_name, "w") as f:
        yaml.dump(d, f, sort_keys=False)
