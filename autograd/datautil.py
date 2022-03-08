import requests
import gzip
import os
import sys
import numpy as np

from pathlib import Path


def _fetch(url):
    root = Path(os.getcwd())
    datafolder = Path(root, '.data/')

    if not datafolder.exists():
        datafolder.mkdir(parents=True)

    filepath = url.replace('/', '-')
    filepath = Path(root, f'.data/{filepath}')
    if filepath.exists():
        sys.stdout.write(f'{url=} already exists, reading it... ')
        with open(filepath, 'rb') as f:
            data = f.read()
    else:
        sys.stdout.write(f'{url=} was not found, downloading it... ')
        filepath.touch()
        with open(filepath, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    sys.stdout.write('done!\n')
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def _remove(url):
    root = Path(os.getcwd())
    datafolder = Path(root, '.data/')

    if not datafolder.exists():
        raise FileNotFoundError(
        f'trying to remove downloaded file but no data folder exists...')

    filepath = url.replace('/', '-')
    filepath = Path(root, f'.data/{filepath}')
    if filepath.exists():
        sys.stdout.write(f'removing {url=} ')
        filepath.unlink(filepath)
        sys.stdout.write('done!\n')
        return True
    raise FileNotFoundError(
    f'{url=} was not found in .data/ directory... skipping')
    return False


def fetch_mnist(version='digits', clean=False):
    urls = {}
    urls['digits'] = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    urls['fashion'] = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']

    if version not in urls:
        raise ValueError(
        f'provided MNIST version does not exist, {version=}')

    datasets = [_fetch(url) for url in urls[version]]
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[8:] if i % 2 else dataset[0x10:].reshape((-1, 28, 28))

    if clean:
        cleaned = [_remove(url) for url in urls[version]]
        if not all(cleaned):
            print('could not remove all downloaded files, manually inspect this')

    return datasets
  
