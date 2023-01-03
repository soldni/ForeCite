import gzip
import hashlib
import json
import multiprocessing
import os
import string
from queue import Queue
from typing import (
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Union
)

import spacy
import springs as sp
import unidecode
from necessary import necessary
from spacy.parts_of_speech import DET
from spacy.tokens import Span

from .io_utils import read_file, recursively_list_files, write_file
from .mp_utils import Map


class ExtractedNounChunk(NamedTuple):
    text: str
    start: int
    end: int


class NounChunkExtractor:
    _SPACY_DISABLE = ["ner", "textcat"]

    def __init__(
        self,
        spacy_model_name: str = "en_core_web_sm",
        spacy_disable: Optional[Sequence[str]] = None,
    ):
        with necessary(
            modules=spacy_model_name,
            message=f"Run `python -m spacy download {spacy_model_name}`",
        ):
            self.nlp = spacy.load(
                spacy_model_name,
                disable=(spacy_disable or self._SPACY_DISABLE),
            )
        self.punctuation = set(string.punctuation)
        self.letters = set(string.ascii_lowercase)
        self.pl_tags = {self.nlp.vocab.strings[tag] for tag in ("NNS", "NNPS")}

        self.nlp.Defaults.stop_words |= {
            "abstract",
            "al",
            "appendices",
            "appendix",
            "author",
            "authors",
            "e.g.",
            "eq",
            "eqs",
            "equation",
            "equations",
            "et",
            "fig",
            "figs",
            "figure",
            "figures",
            "i.e.",
            "sec",
            "section",
            "subsection",
            "tab",
            "table",
            "tables",
            "tabs",
            "title",
        }

    def process_chunk(self, chunk: Span) -> Union[ExtractedNounChunk, None]:
        if (
            chunk[0].like_num
            or chunk[0].like_email
            or chunk[0].like_url
            or chunk[0].is_punct
        ):
            # first token is a number
            return None

        if chunk[0].pos == DET:
            return self.process_chunk(chunk[1:])

        if len(chunk) >= 10:
            return None

        if all(len(t) < 3 for t in chunk):
            return None

        if len(chunk) == 1 and chunk[0].is_stop:
            return None

        # use the lemma if it's a plural noun, otherwise use the text
        text = "".join(
            (t.lemma_ if t.tag in self.pl_tags else t.text) + t.whitespace_
            for t in chunk
        ).lower().strip()

        if text[0] in self.punctuation:
            # first character is punctuation
            return None

        if not text.isascii():
            text = unidecode.unidecode_expect_nonascii(text)

        if not any(c in self.letters for c in text):
            # no letters
            return None

        return ExtractedNounChunk(
            text=text, start=chunk.start_char, end=chunk.end_char
        )

    def __call__(
        self,
        text: Union[str, Iterable[str]]
    ) -> Iterable[ExtractedNounChunk]:

        text = [text] if isinstance(text, str) else text
        offset = 0

        for doc in self.nlp.pipe(text):
            for chunk in doc.noun_chunks:
                extracted = self.process_chunk(chunk=chunk)
                if extracted is None:
                    continue
                yield ExtractedNounChunk(
                    text=extracted.text,
                    start=extracted.start + offset,
                    end=extracted.end + offset
                )
            offset += len(doc.text_with_ws)


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


def dump_and_gzip(data: Union[dict, list], newline: bool = True) -> bytes:
    return gzip.compress(
        (
            json.dumps(data) + ("\n" if newline else "")
        ).encode("utf-8")
    )


def process_single(
    path: str,
    cfg: "Config",
    docs_progress_bar: Queue[int]
):
    nce = NounChunkExtractor()

    # texts, nps_to_write = [], []

    nps_to_write: List[bytes] = []
    details_to_write: List[bytes] = []

    part = 0

    (h := hashlib.md5()).update(path.encode("utf-8"))
    path_hash = h.hexdigest()
    prefix = path_hash[:cfg.prefix_len]

    def make_path(
        part: int, root: str = cfg.output, group: Optional[str] = None
    ) -> str:
        parts = [root]
        parts.append(str(group)) if group else None
        parts.append(prefix) if prefix else None
        parts.append(f"{path_hash}_{part}.jsonl.gz")
        return os.path.join(*parts)

    with read_file(path, mode="rb") as in_f:
        raw_content = gzip.decompress(in_f.read()).decode("utf-8")
        for ln in escape_line_breaks(raw_content).splitlines():
            data = json.loads(ln)

            noun_chunks: Set[str] = set()
            noun_chunks_details: List[dict] = []
            for sec in ("title", "abstract", "full_text"):
                if not data[sec]:
                    continue

                # we add the newline back in so that the start/end offsets
                # are correct
                texts = [f"{s}\n" for s in data[sec].split("\n")]
                for nc in nce(texts):
                    noun_chunks.add(nc.text)
                    noun_chunks_details.append(
                        dict(
                            text=nc.text, start=nc.start, end=nc.end, sec=sec)
                    )
            nps_to_write.append(
                dump_and_gzip(
                    {
                        "id": data["id"],
                        "year": data["year"],
                        "citations": [
                            c for c in data.get("citations", [])
                            if c is not None
                        ],
                        "noun_chunks": list(noun_chunks),
                    }
                )
            )

            details_to_write.append(
                dump_and_gzip(
                    {
                        "id": data["id"],
                        "noun_chunks": noun_chunks_details,
                    }
                )
            )
            docs_progress_bar.put(1)

            if len(nps_to_write) >= cfg.per_file:
                np_path = make_path(part=part, group='np')
                with write_file(np_path, mode="wb") as out_f:
                    out_f.writelines(nps_to_write)

                details_path = make_path(part=part, group='details')
                with write_file(details_path, mode="wb") as out_f:
                    out_f.writelines(details_to_write)

                part += 1
                nps_to_write, details_to_write = [], []

    if nps_to_write:
        # write the remaining lines if any
        np_path = make_path(part=part, group='np')
        with write_file(np_path, mode="wb") as out_f:
            out_f.writelines(nps_to_write)

        details_path = make_path(part=part, group='details')
        with write_file(details_path, mode="wb") as out_f:
            out_f.writelines(details_to_write)


@sp.dataclass
class Config:
    prefix: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_content_cits_clean/"
    output: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_np_cits_ascii_clean/"
    n_proc: int = max(multiprocessing.cpu_count() - 1, 1)
    debug: bool = False
    per_file: int = 1000
    prefix_len: int = 1


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

    # per-file must be an integer > 0
    assert isinstance(cfg.per_file, int) and cfg.per_file > 0, \
        "per_file must be an integer > 0"

    # prefix_len must be an 0 <= integer <= (length md5 hash / 2)
    assert isinstance(cfg.prefix_len, int) and 0 <= cfg.prefix_len <= 16, \
        "prefix_len must be an integer between 0 and 16"

    # get file system depending on protocol in the prefix
    sources = list(recursively_list_files(cfg.prefix))

    docs_progress_bar : Queue[int] = Queue()

    with Map(
        n_proc=cfg.n_proc, debug=cfg.debug, pbar="Extracting NPs..."
    ) as map_pool:

        # adding a second progress bar for documents; the first one
        # is for files.
        docs_progress_bar = map_pool.add_progress_bar(
            desc="Processed...",
            unit=" docs",
            unit_scale=True
        )
        map_pool(
            process_single,
            sources,
            cfg=cfg,
            docs_progress_bar=docs_progress_bar
        )


if __name__ == "__main__":
    main()
