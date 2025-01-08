import json

def write_json(json_data, filepath):

    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)

def read_json(filepath):

    with open(filepath, 'r') as f:
        json_data = json.load(f)
    return json_data
