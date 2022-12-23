import gzip
import hashlib
import json
import multiprocessing
import os
from collections import Counter
import re
from string import punctuation
from typing import Counter as CounterType, cast
from typing import Iterable, Optional, Sequence, Set, Union

import spacy
import springs as sp
from cached_path import cached_path
from necessary import necessary
from spacy.tokens import Span
from textacy.extract.basics import noun_chunks as textacy_noun_chunks
from ftfy import fix_text

from .io_utils import read_file, recursively_list_files, write_file
from .mp_utils import Map, Reduce


class NounChunkExtractor:
    _SPACY_DISABLE = ["ner", "textcat"]

    def __init__(
        self,
        spacy_model_name: str = "en_core_web_sm",
        spacy_disable: Optional[Sequence[str]] = None,
        return_lemma: bool = False,
        return_lower: bool = True,
    ):
        with necessary(
            modules="en_core_web_sm",
            message="Please run `python -m spacy download en_core_web_sm`",
        ):
            self.nlp = spacy.load(
                spacy_model_name,
                disable=(spacy_disable or self._SPACY_DISABLE),
            )

        self.return_lemma = return_lemma
        self.return_lower = return_lower
        self.leading_digits_re = re.compile(r"^[\d,\.]+\b")
        self.stopwords = self.get_stopwords()
        self.punctuation = set(punctuation)

    @staticmethod
    def get_stopwords(
        url: str = (
            "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/"
            "lucas/Alir3z4-stop-words-6dedf5e/english.txt"
        ),
    ) -> Set[str]:

        with open(cached_path(url), "r") as f:
            stopwords = set(f.read().splitlines())
        stopwords.update({
            'table', 'tables', 'tab', 'tab.', 'tabs', 'tabs.',
            'figure', 'figures', 'fig', 'fig.', 'figs', 'figs.'
        })
        return stopwords

    def process_chunk(self, chunk: Span) -> Union[str, None]:
        if (chunk.end_char - chunk.start_char) < 3 or len(chunk) >= 10:
            return None

        if self.leading_digits_re.match(chunk[0].text):
            # first token is a number
            return None

        text = chunk.lemma_ if self.return_lemma else chunk.text
        text = text.lower() if self.return_lower else text

        if text in self.stopwords:
            # full chunk is a stopword
            return None

        if text[0] in self.punctuation:
            # first character is punctuation
            return None

        if text.startswith('http://') or text.startswith('https://'):
            # is a URL
            return None

        text = fix_text(text)

        return text

    def get_lemma(self, span: Span) -> str:
        return span.lemma_ if self.return_lemma else span.text

    def get_lower(self, span_text: str) -> str:
        return (span_text.lower() if self.return_lower else span_text).strip()

    def __call__(
        self, text: Union[str, Sequence[str]]
    ) -> Iterable[Set[str]]:
        text = [text] if isinstance(text, str) else text
        for doc in self.nlp.pipe(text):
            chunks = set(
                self.process_chunk(chunk)
                for chunk in textacy_noun_chunks(doc, drop_determiners=True)
            )
            chunks.discard(None)
            yield cast(Set[str], chunks)


def get_stopwords(
    url: str = (
        "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/"
        "lucas/Alir3z4-stop-words-6dedf5e/english.txt"
    ),
) -> Set[str]:
    with open(cached_path(url), "r") as f:
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


def process_single(path: str, cfg: "Config") -> CounterType[str]:
    ids, years, texts = [], [], []

    h = hashlib.md5()
    with read_file(path, mode="rb") as f:
        raw_content = gzip.decompress(f.read()).decode("utf-8")
        for ln in escape_line_breaks(raw_content).splitlines():
            h.update(ln.encode("utf-8"))
            data = json.loads(ln)

            ids.append(data["id"])
            years.append(data["year"])
            texts.append(
                ("{title}\n\n\n{abstract}\n\n\n{full_text}")
                .format(**data)
                .strip()
            )

    nce = NounChunkExtractor()
    vocab: CounterType[str] = Counter()

    out_filepath = os.path.join(cfg.output, "np", f"{h.hexdigest()}.jsonl")

    with write_file(out_filepath) as f:
        for id_, year, nc in zip(ids, years, nce(texts)):
            nc_dict = Counter(w for w in nc)
            vocab.update(nc_dict)
            f.write(
                json.dumps(
                    dict(id=id_, year=year, noun_chunks=sorted(nc_dict))
                )
                + "\n"
            )
    return vocab


@sp.dataclass
class Config:
    prefix: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_content/"
    output: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases/"
    n_proc: int = max(multiprocessing.cpu_count() - 1, 1)
    debug: bool = False
    freq: int = 3


def merge_vocabs(*vocabs: CounterType[str]):
    first, *rest = vocabs
    for v in rest:
        first.update(v)
    return first


@sp.cli(Config)
def main(cfg: Config):
    """
    Tokenize the corpus and build an index of noun phrases.

    This program operates roughly as follows:

    1. Get a list of all files in the corpus.
    2. For each paper in each file, run spacy to get the noun phrases.
    3. Write id, year, and noun phrases for each paper to a file.
    4. Build a vocabulary of noun phrases, maybe with some cutoff for
       frequency.
    """

    # get file system depending on protocol in the prefix
    sources = list(recursively_list_files(cfg.prefix))

    map_pool = Map(n_proc=cfg.n_proc, debug=cfg.debug, pbar="Running spacy...")
    part_vocabs = map_pool(process_single, sources, cfg=cfg)

    reduce_pool = Reduce(
        n_proc=cfg.n_proc, debug=cfg.debug, pbar="Merging vocab..."
    )
    vocab = reduce_pool(merge_vocabs, part_vocabs)

    with write_file(os.path.join(cfg.output, "vocab.txt"), "w") as f:
        for word, count in vocab.most_common():
            if count < cfg.freq:
                break
            f.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    main()
