from contextlib import contextmanager
import os
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from typing import IO, Generator, Iterable

import springs as sp
import boto3


LOGGER = sp.configure_logging(logger_name=__name__, logging_level='INFO')


@contextmanager
def read_file(
    path: str, mode: str = 'r', **kwargs
) -> Generator[IO, None, None]:
    parse = urlparse(path)
    remove = False

    assert 'r' in mode, 'Only read mode is supported'

    if parse.scheme == 's3':
        client = boto3.client('s3')
        LOGGER.info(f'Downloading {path} to a temporary file')
        with NamedTemporaryFile(delete=False) as f:
            path = f.name
            client.download_fileobj(parse.netloc, parse.path.lstrip('/'), f)
            remove = True
    elif parse.scheme == 'file' or parse.scheme == '':
        pass
    else:
        raise ValueError(f'Unsupported scheme {parse.scheme}')

    try:
        with open(path, mode=mode, **kwargs) as f:
            yield f
    finally:
        if remove:
            os.remove(path)


@contextmanager
def write_file(
    path: str, mode: str = 'w', **kwargs
) -> Generator[IO, None, None]:
    parse = urlparse(path)
    local = None

    assert 'w' in mode or 'a' in mode, 'Only write/append mode is supported'

    try:
        if parse.scheme == 'file' or parse.scheme == '':
            with open(path, mode=mode, **kwargs) as f:
                yield f
        else:
            with NamedTemporaryFile(delete=False, mode=mode) as f:
                yield f
                local = f.name
    finally:
        if local is None:
            pass
        elif parse.scheme == 's3':
            LOGGER.info(
                f'Uploading {local} to {parse.netloc}{parse.path.lstrip("/")}'
            )
            client = boto3.client('s3')
            client.upload_file(local, parse.netloc, parse.path.lstrip('/'))
            os.remove(local)
        else:
            raise ValueError(f'Unsupported scheme {parse.scheme}')


def recursively_list_files(path: str) -> Iterable[str]:
    parse = urlparse(path)

    if parse.scheme == 's3':
        cl = boto3.client('s3')
        prefixes = [parse.path.lstrip('/')]

        while len(prefixes) > 0:
            prefix = prefixes.pop()
            response = cl.list_objects_v2(Bucket=parse.netloc, Prefix=prefix)
            for obj in response['Contents']:
                if obj['Key'][-1] == '/':
                    prefixes.append(obj['Key'])
                else:
                    yield f's3://{parse.netloc}/{obj["Key"]}'

    elif parse.scheme == 'file' or parse.scheme == '':
        for root, _, files in os.walk(parse.path):
            for f in files:
                yield os.path.join(root, f)
    else:
        raise NotImplementedError(f'Unknown scheme: {parse.scheme}')
