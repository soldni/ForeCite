from collections import Counter
from contextlib import contextmanager
import hashlib
import multiprocessing
import json
import gzip
import os
from string import punctuation
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from functools import partial
from typing import IO, Generator, Sequence, Optional, Iterable, Union, Set

import springs as sp
import boto3
from cached_path import cached_path
from tqdm import tqdm
import spacy
from spacy.tokens import Doc, Span
from necessary import necessary
from textacy.extract.basics import noun_chunks as textacy_noun_chunks


LOGGER = sp.configure_logging(__name__)


class NounChunkExtractor:
    # _TEXTACY_POS_EXPAND_NC = {'NOUN', 'VERB', 'PROPN'}
    # _TEXTACY_POS_INITIAL_NC = {'NOUN', 'ADJ', 'VERB', 'PROPN'}
    _SPACY_DISABLE = ['ner', 'textcat']

    def __init__(
        self,
        spacy_model_name: str = "en_core_web_sm",
        spacy_disable: Optional[Sequence[str]] = None,
        # textacy_pos_initial_nc: Optional[Sequence[str]] = None,
        # textacy_pos_expand_nc: Optional[Sequence[str]] = None,
        return_lemma: bool = False,
        return_lower: bool = True,
        pbar_name: str = 'NounChunkExtractor',
        pbar_pos: int = 0,
    ):
        with necessary(
            modules='en_core_web_sm',
            message='Please run `python -m spacy download en_core_web_sm`'
        ):
            self.nlp = spacy.load(
                spacy_model_name,
                disable=(spacy_disable or self._SPACY_DISABLE)
            )

        # self.textacy_pos_expand_nc = set(
        #     textacy_pos_expand_nc or self._TEXTACY_POS_EXPAND_NC
        # )
        # self.textacy_pos_initial_nc = set(
        #     textacy_pos_initial_nc or self._TEXTACY_POS_INITIAL_NC
        # )

        self.return_lemma = return_lemma
        self.return_lower = return_lower

        self.pbar_name = pbar_name
        self.pbar_pos = pbar_pos

    # def find_chunks(self, doc: Doc) -> Iterable[Span]:
    #     extracted_noun_chunks = textacy_noun_chunks(doc, drop_determiners=True)
    #     for noun_chunk in extracted_noun_chunks:
    #         not_acceptable_start_word = True
    #         while not_acceptable_start_word and len(noun_chunk) > 0:
    #             not_acceptable_start_word = \
    #                 noun_chunk[0].pos_ not in self.textacy_pos_initial_nc
    #             noun_chunk = noun_chunk[1 if not_acceptable_start_word else 0:]
    #         if len(noun_chunk) > 0:
    #             yield noun_chunk

    def get_lemma(self, span: Span) -> str:
        return span.lemma_ if self.return_lemma else span.text

    def get_lower(self, span_text: str) -> str:
        return (
            span_text.lower() if self.return_lower else span_text
        ).strip()

    def __call__(
        self, text: Union[str, Sequence[str]]
    ) -> Iterable[Sequence[str]]:
        texts_it = tqdm(
            [text] if isinstance(text, str) else text,
            desc=self.pbar_name,
            position=self.pbar_pos
        )
        for doc in self.nlp.pipe(texts_it):
            yield [
                self.get_lower(self.get_lemma(chunk))
                # for chunk in self.find_chunks(doc)
                for chunk in textacy_noun_chunks(doc, drop_determiners=True)
            ]


def get_stopwords(
    url: str = (
        'https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/'
        'lucas/Alir3z4-stop-words-6dedf5e/english.txt'
    )
) -> Set[str]:
    with open(cached_path(url), 'r') as f:
        return set(f.read().splitlines())


def escape_line_breaks(text: str) -> str:
    """Adapted from ftfy.fixes.fix_line_breaks; escapes non-unix line breaks
    instead of replacing them with unix line breaks."""
    return (
        text.replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\u2028", "\\n")
        .replace("\u2029", "\\n")
        .replace("\u0085", "\\n")
    )


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


def process_single(path: str, cfg: 'Config') -> Counter[str]:
    if len(pid := multiprocessing.current_process()._identity) > 0:
        pos = pid[0] - 1
    else:
        pos = 0

    name = (_p := os.path.basename(path))[:10] + '...' + _p[-10:]
    (stopwords := get_stopwords()).update(set(punctuation))

    ids, years, texts = [], [], []

    h = hashlib.md5()
    with read_file(path, mode='rb') as f:
        raw_content = gzip.decompress(f.read()).decode('utf-8')
        for ln in escape_line_breaks(raw_content).splitlines():
            h.update(ln.encode('utf-8'))
            data = json.loads(ln)

            ids.append(data['id'])
            years.append(data['year'])
            texts.append(
                (
                    '{title}\n\n\n{abstract}\n\n\n{full_text}'
                ).format(**data).strip()
            )

    nce = NounChunkExtractor(pbar_name=f'Processing {name}', pbar_pos=pos)
    vocab: Counter[str] = Counter()

    out_filepath = os.path.join(cfg.output, 'np', f'{h.hexdigest()}.jsonl')

    with write_file(out_filepath) as f:
        for id_, year, nc in zip(ids, years, nce(texts)):
            nc_dict = Counter(w for w in nc if w not in stopwords)
            vocab.update(nc_dict)
            f.write(
                json.dumps(
                    dict(id=id_, year=year, noun_chunks=sorted(nc_dict))
                ) + '\n'
            )
    return vocab


def list_all_subfiles(path: str) -> Iterable[str]:
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
        for root, dirs, files in os.walk(parse.path):
            for f in files:
                yield os.path.join(root, f)

    else:
        raise NotImplementedError(f'Unknown scheme: {parse.scheme}')


@sp.dataclass
class Config:
    prefix: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_content/'
    output: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases/'
    n_proc: int = max(multiprocessing.cpu_count() - 1, 1)
    debug: bool = False
    freq: int = 3


@sp.cli(Config)
def main(cfg: Config):
    '''
    Tokenize the corpus and build an index of noun phrases.

    This program operates roughly as follows:

    1. Get a list of all files in the corpus.
    2. For each paper in each file, run spacy to get the noun phrases.
    3. Write id, year, and noun phrases for each paper to a file.
    4. Build a vocabulary of noun phrases, maybe with some cutoff for
       frequency.
    '''

    # get file system depending on protocol in the prefix
    sources = list(list_all_subfiles(cfg.prefix))

    if cfg.debug:
        part_vocabs = [process_single(path=path, cfg=cfg) for path in sources]
    else:
        with multiprocessing.Pool(cfg.n_proc) as pool:
            part_vocabs = pool.map(partial(process_single, cfg=cfg), sources)

    vocab: Counter[str] = Counter()
    for part_vocab in part_vocabs:
        vocab.update(part_vocab)

    with write_file(os.path.join(cfg.output, 'vocab.txt'), 'w') as f:
        for word, count in vocab.most_common():
            if count < cfg.freq:
                break
            f.write(f'{word}\t{count}\n')


if __name__ == '__main__':
    main()
