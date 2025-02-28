import os
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Callable, Generator, Iterable, Optional
from urllib.parse import urlparse

import boto3
import springs as sp

LOGGER = sp.configure_logging(logger_name=__name__, logging_level="INFO")


@contextmanager
def read_file(
    path: str,
    mode: str = "r",
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    **kwargs,
) -> Generator[IO, None, None]:
    logger = logger or LOGGER
    open_fn = open_fn or open
    parse = urlparse(path)
    remove = False

    assert "r" in mode, "Only read mode is supported"

    if parse.scheme == "s3":
        client = boto3.client("s3")
        logger.info(f"Downloading {path} to a temporary file")
        with NamedTemporaryFile(delete=False) as f:
            path = f.name
            client.download_fileobj(parse.netloc, parse.path.lstrip("/"), f)
            remove = True
    elif parse.scheme == "file" or parse.scheme == "":
        pass
    else:
        raise ValueError(f"Unsupported scheme {parse.scheme}")

    try:
        with open_fn(file=path, mode=mode, **kwargs) as f:
            yield f
    finally:
        if remove:
            os.remove(path)


@contextmanager
def write_file(
    path: str,
    mode: str = "w",
    skip_if_empty: bool = False,
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    **kwargs,
) -> Generator[IO, None, None]:

    parse = urlparse(path)
    local = None
    logger = logger or LOGGER
    open_fn = open_fn or open

    assert "w" in mode or "a" in mode, "Only write/append mode is supported"

    try:
        if parse.scheme == "file" or parse.scheme == "":
            # make enclosing directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open_fn(file=path, mode=mode, **kwargs) as f:
                yield f
        else:
            with NamedTemporaryFile(delete=False, mode=mode) as f:
                yield f
                local = f.name
    finally:
        if local is None:
            if skip_if_empty and os.stat(path).st_size == 0:
                logger.info(f"Skipping empty file {path}")
                os.remove(path)
        elif parse.scheme == "s3":
            dst = f'{parse.netloc}{parse.path.lstrip("/")}'
            if skip_if_empty and os.stat(local).st_size == 0:
                logger.info(f"Skipping upload to {dst} since {local} is empty")
            else:
                logger.info(f"Uploading {local} to {dst}")
                client = boto3.client("s3")
                client.upload_file(local, parse.netloc, parse.path.lstrip("/"))
            os.remove(local)
        else:
            raise ValueError(f"Unsupported scheme {parse.scheme}")


def recursively_list_files(
    path: str, ignore_hidden_files: bool = True
) -> Iterable[str]:
    parse = urlparse(path)

    if parse.scheme == "s3":
        cl = boto3.client("s3")
        prefixes = [parse.path.lstrip("/")]

        while len(prefixes) > 0:
            prefix = prefixes.pop()
            paginator = cl.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=parse.netloc, Prefix=prefix)
            for page in pages:
                for obj in page["Contents"]:
                    if obj["Key"][-1] == "/":
                        prefixes.append(obj["Key"])
                    else:
                        yield f's3://{parse.netloc}/{obj["Key"]}'

    elif parse.scheme == "file" or parse.scheme == "":
        for root, _, files in os.walk(parse.path):
            for f in files:
                if ignore_hidden_files and f.startswith("."):
                    continue
                yield os.path.join(root, f)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")
