from collections import Counter
import hashlib
import multiprocessing
import json
import gzip
import os
from string import punctuation
from urllib.parse import urlparse
from functools import partial
from typing import Sequence, Optional, Iterable, Union, Set

import springs as sp
import fsspec
from cached_path import cached_path
from tqdm import tqdm
import spacy
from spacy.tokens import Doc, Span
from necessary import necessary
from textacy.extract.basics import noun_chunks as textacy_noun_chunks


class NounChunkExtractor:
    _TEXTACY_POS_EXPAND_NC = {'NOUN', 'VERB', 'PROPN'}
    _TEXTACY_POS_INITIAL_NC = {'NOUN', 'ADJ', 'VERB', 'PROPN'}
    _SPACY_DISABLE = ['ner', 'textcat']

    def __init__(
        self,
        name: str = 'NounChunkExtractor',
        spacy_model_name: str = "en_core_web_sm",
        spacy_disable: Optional[Sequence[str]] = None,
        textacy_pos_initial_nc: Optional[Sequence[str]] = None,
        textacy_pos_expand_nc: Optional[Sequence[str]] = None,
        return_lemma: bool = False,
        return_lower: bool = True,
    ):
        with necessary(
            modules='en_core_web_sm',
            message='Please run `python -m spacy download en_core_web_sm`'
        ):
            self.nlp = spacy.load(
                spacy_model_name,
                disable=(spacy_disable or self._SPACY_DISABLE)
            )

        self.textacy_pos_expand_nc = set(
            textacy_pos_expand_nc or self._TEXTACY_POS_EXPAND_NC
        )
        self.textacy_pos_initial_nc = set(
            textacy_pos_initial_nc or self._TEXTACY_POS_INITIAL_NC
        )

        self.name = name
        self.return_lemma = return_lemma
        self.return_lower = return_lower

    def find_chunks(self, doc: Doc) -> Iterable[Span]:
        extracted_noun_chunks = textacy_noun_chunks(doc, drop_determiners=True)
        for noun_chunk in extracted_noun_chunks:
            not_acceptable_start_word = True
            while not_acceptable_start_word and len(noun_chunk) > 0:
                not_acceptable_start_word = \
                    noun_chunk[0].pos_ not in self.textacy_pos_initial_nc
                noun_chunk = noun_chunk[1 if not_acceptable_start_word else 0:]
            if len(noun_chunk) > 0:
                yield noun_chunk

    def get_lemma(self, span: Span) -> str:
        return span.lemma_ if self.return_lemma else span.text

    def get_lower(self, span_text: str) -> str:
        return (
            span_text.lower() if self.return_lower else span_text
        ).strip()

    def __call__(
        self, text: Union[str, Sequence[str]]
    ) -> Iterable[Sequence[str]]:
        for doc in tqdm(
            self.nlp.pipe([text] if isinstance(text, str) else text),
            desc=self.name,
            total=len(text) if isinstance(text, Sequence) else 1
        ):
            yield [
                self.get_lower(self.get_lemma(chunk))
                for chunk in self.find_chunks(doc)
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


def process_single(path: str, cfg: 'Config') -> Counter[str]:
    in_fs = fsspec.filesystem(urlparse(cfg.prefix).scheme)
    (stopwords := get_stopwords()).update(set(punctuation))

    ids, years, texts = [], [], []

    h = hashlib.md5()

    with in_fs.open(path, 'rb') as f:
        raw_content = gzip.decompress(f.read()).decode('utf-8')
        for ln in escape_line_breaks(raw_content).splitlines():
            if not ln:
                continue

            h.update(ln.encode('utf-8'))
            data = json.loads(ln)

            ids.append(data['id'])
            years.append(data['year'])
            texts.append(
                (
                    '{title}\n\n\n{abstract}\n\n\n{full_text}'
                ).format(**data).strip()
            )

    name = (_p := os.path.basename(path))[:10] + '...' + _p[-10:]
    nce = NounChunkExtractor(name=f'Processing {name}')
    vocab: Counter[str] = Counter()

    out_fs = fsspec.filesystem((out_scheme := urlparse(cfg.output).scheme))
    output = cfg.output.lstrip(f'{out_scheme}://' if out_scheme else '')
    out_filepath = os.path.join(output, 'np', f'{h.hexdigest()}.jsonl')

    with out_fs.open(out_filepath, 'w') as f:
        for id_, year, nc in zip(ids, years, nce(texts)):
            nc_dict = Counter(w for w in nc if w not in stopwords)
            vocab.update(nc_dict)
            f.write(
                json.dumps(
                    dict(id=id_, year=year, noun_chunks=sorted(nc_dict))
                ) + '\n'
            )
    return vocab


@sp.dataclass
class Config:
    prefix: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_content/'
    output: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases/'
    n_proc: int = multiprocessing.cpu_count()
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
    fs = fsspec.filesystem(urlparse(cfg.prefix).scheme)
    sources = [path['name'] for path in fs.listdir(cfg.prefix)]

    if cfg.debug:
        part_vocabs = [process_single(path=path, cfg=cfg) for path in sources]
    else:
        with multiprocessing.Pool(cfg.n_proc) as pool:
            part_vocabs = pool.map(partial(process_single, cfg=cfg), sources)

    vocab: Counter[str] = Counter()
    for part_vocab in part_vocabs:
        vocab.update(part_vocab)

    out_fs = fsspec.filesystem((out_scheme := urlparse(cfg.output).scheme))
    output = cfg.output.lstrip(f'{out_scheme}://' if out_scheme else '')
    out_filepath = os.path.join(output, 'vocab.txt')

    with out_fs.open(out_filepath, 'w') as f:
        for word, count in vocab.most_common():
            if count < cfg.freq:
                break
            f.write(f'{word}\t{count}\n')


if __name__ == '__main__':
    main()
