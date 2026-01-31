import yaml

## used for reading YAML files
def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out

## combine multiple metadata files
def combine_yaml(files):
    metadata = {}
    for f in files:
        metadata.update(read_yaml(f))
    return metadata    
