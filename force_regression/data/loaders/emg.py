from typing import Dict, Any
import scipy.io as sio
import os
import mat73


def load_matfile(filepath: str) -> Dict[str, Any]:
    """
    This function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")
    if os.path.splitext(filepath)[1] != '.mat':
        raise ValueError("Invalid file format. Only .mat files are supported.")
    try:
        data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

        print(f"File {os.path.basename(filepath)} loaded successfully with sio")
        return _check_keys(data)
    except Exception as e1:
        try:
            data = mat73.loadmat(filepath)
            print(f"File {os.path.basename(filepath)} loaded successfully with mat73")
            return _check_keys(data)
        except Exception as e2:
            raise Exception(f"Error loading file {os.path.basename(filepath)} with both sio and mat73: {e1}, {e2}")


def _check_keys(dict_func: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert mat-objects in the dictionary to nested dictionaries.
    """
    for key in dict_func:
        if isinstance(dict_func[key], sio.matlab.mat_struct):
            dict_func[key] = _todict(dict_func[key])
    return dict_func


def _todict(matobj: sio.matlab.mat_struct) -> Dict[str, Any]:
    """
    Recursively convert mat-objects to nested dictionaries.
    """
    reformat_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            reformat_dict[strg] = _todict(elem)
        else:
            reformat_dict[strg] = elem
    return reformat_dict


def check_filename(path: str, filename: str) -> str:
    """
    Validate and construct the full file path.
    """
    if not filename.endswith(".mat"):
        raise ValueError("File should end with .mat, is this a MATLAB file?")

    return path.rstrip("/") + "/" + filename

