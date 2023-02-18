import json


def write_json(data, path_to_file: str, indent=2):
    with open(path_to_file, 'w') as f:
        json.dump(data, f, indent=indent)


def read_json(path_to_file: str):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data
