"""

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 24-02-2022
"""
import requests
import gzip
import os
import hashlib
import numpy as np

from pathlib import Path



def fetch_mnist():
    __allurls__ = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    datasets = [_fetch(url) for url in __allurls__]
    for i, dataset in enumerate(datasets):
        if i % 2 == 0:
            datasets[i] = dataset[0x10:].reshape((-1, 28, 28))
        else:
            datasets[i] = dataset[8:]
    
    return datasets


def _fetch(url):
    __root__ = Path(os.getcwd())
    datafolder = Path(__root__, '.data/')

    if not datafolder.exists():
        datafolder.mkdir(parents=True)

    urlpath = url.replace('/', '-')
    fpath = Path(__root__, f'.data/{urlpath}')
    if fpath.exists():
        print(f'{url} already exists, loading it...')
        with open(fpath, 'rb') as f:
            data = f.read()
    else:
        print(f'{url} does not exist, downloading it...')
        fpath.touch()
        with open(fpath, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

