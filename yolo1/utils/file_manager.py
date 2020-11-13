"""
Manage the downloading of external files that are used in YOLO networks.
"""

# from __future__ import annotations

import io
import os

from typing import Union

# define PathABC type
try:
    PathABC = Union[bytes, str, os.PathLike]
except AttributeError:
    # not Python 3.6+
    import pathlib
    PathABC = Union[bytes, str, pathlib.Path]


def get_size(path: Union[PathABC, io.IOBase]) -> int:
    """
    A unified method to find the size of a file, either by its path or an open
    file object.

    Arguments:
        path: a path (as a str or a Path object) or an open file (which must be
              seekable)

    Return:
        size of the file

    Raises:
        ValueError: the IO object given as path is not open or it not seekable
        FileNotFoundError: the given path is invalid
    """
    if isinstance(path, io.IOBase):
        if path.seekable():
            currentPos = path.tell()
            path.seek(-1, io.SEEK_END)
            size = path.tell() + 1
            path.seek(currentPos)
            return size
        else:
            raise ValueError(
                "IO object must be seekable in order to find the size.")
    else:
        return os.path.getsize(path)


def open_if_not_open(file: Union[PathABC, io.IOBase], *args,
                     **kwargs) -> io.IOBase:
    """
    Takes an input that can either be a file or a path. If the input is given as
    a file, it is returned without modification. If it is a path, it is opened
    as a file.

    Arguments:
        file: a path or file that is being opened if it already isn't
        *args, **kwargs: to see the potential additional arguments or keywords
                         that can be based into this function, consult the open
                         builtin function.

    Returns:
        opened file

    Raises:
        IOError: consult the open builtin function for the potential IO errors
                 that can be raised when opening the file
    """
    if isinstance(file, io.IOBase):
        return file
    return open(file, *args, **kwargs)


# URL names that can be accessed using the download function
urls = {
    'yolov1.cfg':
    ('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1.cfg',
     'cfg',
     '8b4b951dd646478ea4214cb389d152793ca98e5c6e67266884908ba084b6211e'),
    'yolov2.cfg':
    ('https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov2.cfg',
     'cfg',
     '57d85d77262c840b56ad5418faae4950d9c7727e0192fb70618eeaac26a19817'),
    'yolov3.cfg':
    ('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
     'cfg',
     '22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953'),
    'yolov3-spp.cfg':
    ('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg',
     'cfg',
     '7a4ec2d7427340fb12059f2b0ef76d6fcfcac132cc287cbbf0be5e3abaa856fd'),
    'yolov3-tiny.cfg':
    ('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
     'cfg',
     '84eb7a675ef87c906019ff5a6e0effe275d175adb75100dcb47f0727917dc2c7'),
    'yolov4.cfg':
    ('https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
     'cfg',
     'a6d0f8e5c62cc8378384f75a8159b95fa2964d4162e33351b00ac82e0fc46a34'),
    'yolov4-tiny.cfg':
    ('https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
     'cfg',
     '6cbf5ece15235f66112e0bedebb324f37199b31aee385b7e18f0bbfb536b258e'),
    'yolov1.weights':
    ('http://pjreddie.com/media/files/yolov1/yolov1.weights', 'weights',
     'df414df832ed10e3f2788df1e0e0ae573976573b97b5eec2c824ab9e5a8ae6d6'),
    'yolov2.weights':
    ('https://pjreddie.com/media/files/yolov2.weights', 'weights',
     'd9945162ed6f54ce1a901e3ec537bdba4d572ecae7873087bd730e5a7942df3f'),
    'yolov3.weights':
    ('https://pjreddie.com/media/files/yolov3.weights', 'weights',
     '523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297'),
    'yolov3-spp.weights':
    ('https://pjreddie.com/media/files/yolov3-spp.weights', 'weights',
     '87a1e8c85c763316f34e428f2295e1db9ed4abcec59dd9544f8052f50de327b4'),
    'yolov3-tiny.weights':
    ('https://pjreddie.com/media/files/yolov3-tiny.weights', 'weights',
     'dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c'),
    'yolov4.weights':
    ('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
     'weights',
     'e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562'),
    'yolov4-tiny.weights':
    ('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
     'weights',
     '037676f0d929c24e1bd9a0037fe30dc416fc26e0ca2a4491a44d024873316061'),
}


def download(name: str, trust: bool = False) -> str:
    """
    Download a predefined file named `name` from the original repository.

    For example, yolov3.weights is a file that defines the pretrained YOLOv3
    model. It can be downloaded from https://pjreddie.com/media/files/yolov3.weights
    so it is downloaded from there.

    Args:
        name: Name of the file that will be downloaded
        trust: Trust the cache even if the file's hash is inconsistent
               This option can speed the loading of the file at the expense of
               security. Default value is False.

    Returns:
        The path of the downloaded file as a `str`

    Raises:
        KeyError:       Name of file is not found in the `urls` variable.
        ValueError:     Name or URL stored in the `urls` variable is invalid.
        OSError:        There was a problem saving the file when it was being
                        downloaded.
        HTTPException:  The file was not able to be downloaded.
        Exception:      Any other undocumented error that ks.utils.get_file may
                        have thrown to indicate that the file was inaccessible.
    """
    import tensorflow.keras as ks
    from http.client import HTTPException

    url, type, hash = urls[name]

    cache_dir = os.path.abspath('cache')
    full_path = os.path.join(cache_dir, type, name)
    if trust and os.path.exists(full_path):
        return full_path

    try:
        if hash is None:
            return ks.utils.get_file(name,
                                     url,
                                     cache_dir=cache_dir,
                                     cache_subdir=type)
        else:
            return ks.utils.get_file(name,
                                     url,
                                     cache_dir=cache_dir,
                                     cache_subdir=type,
                                     file_hash=hash,
                                     hash_algorithm='sha256')
    except Exception as e:
        if 'URL fetch failure on' in str(e):
            raise HTTPException(str(e)) from e
        else:
            raise
