import os
import typing
import numpy as np
from scipy import ndimage
from base.file_util import open, cache_file


def read_tensor(path: str) -> typing.Optional[np.ndarray]:
    """
    Load a saved a tensor, saved either as an image file for standard RGB images or as a numpy archive for more general
    tensors.
    """
    (_, ext) = os.path.splitext(path)
    ext = ext.lower()
    if ext in {'.png', '.jpg', '.jpeg'}:
        with open(path) as f:
            res = ndimage.imread(f, mode='RGB')
        assert len(res.shape) == 3
        assert res.shape[2] == 3
        return res
    elif ext in {'.npz'}:
        try:
            data = np.load(cache_file(path))
            assert len(list(data.items())) == 1
        except Exception as e:
            print('Error unzipping %s' % path)
            print(e, flush=True)
            return None
        return data['arr_0']
    else:
        raise RuntimeError(
            'Extension %s for file %s not supported' % (ext, path)
        )


def write_tensor(dst: str, value: np.ndarray) -> None:
    """Save a numpy tensor to a given location."""
    (_, ext) = os.path.splitext(dst)
    assert (ext == '' or ext == '.npz')
    with open(dst, 'wb') as f:
        np.savez_compressed(f, value)
