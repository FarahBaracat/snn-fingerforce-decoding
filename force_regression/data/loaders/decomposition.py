import pickle as pkl
import torch
import io


class CpuUnpickler(pkl.Unpickler):
    """
    A custom unpickler class to load pickled data onto CPU memory.
    """
    def find_class(self, module: str, name: str):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            return super().find_class(module, name)


def load_pickle(filename: str) -> dict:
    with open(filename, 'rb') as file:
        contents = CpuUnpickler(file).load()

    return contents
