import gzip
import hashlib
import json
import multiprocessing
import os
import string
from queue import Queue
from collections import OrderedDict
from typing import (
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
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
        text: Union[str, Sequence[str]]
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


def process_single(
    path: str,
    cfg: "Config",
    docs_progress_bar: Queue[int]
):
    nce = NounChunkExtractor()

    texts, to_write = [], []

    with read_file(path, mode="rb") as in_f:
        raw_content = gzip.decompress(in_f.read()).decode("utf-8")
        for ln in escape_line_breaks(raw_content).splitlines():
            data = json.loads(ln)
            t, a, ft = data["title"], data["abstract"], data["full_text"]

            texts = [
                (t.strip() if t else ""),
                (a.strip() if a else ""),
                *(map(str.strip, ft.split("\n")) if ft else ""),
            ]

            noun_chunks: Dict[str, List[List[int]]] = OrderedDict()
            for nc in nce(texts):
                noun_chunks.setdefault(nc.text, []).append([nc.start, nc.end])

            docs_progress_bar.put(1)

            data = dict(
                id=data["id"],
                year=data["year"],
                # not all files have citation fields
                citations=[
                    ct for ct in (
                        data.get("citations", None) or []
                    ) if ct is not None
                ],
                # save the individual chunks in noun_chunks...
                noun_chunks=list(noun_chunks.keys()),
                # ...and the locations of each chunk in nc_locations
                nc_locations=list(noun_chunks.values()),
            )
            to_write.append(json.dumps(data) + "\n")

    (h := hashlib.md5()).update(path.encode("utf-8"))
    out_filepath = os.path.join(cfg.output, "np", f"{h.hexdigest()}.jsonl")
    with write_file(out_filepath, mode="w") as out_f:
        out_f.writelines(to_write)


@sp.dataclass
class Config:
    prefix: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_content_cits_clean/"
    output: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_np_cits_ascii_clean/"
    n_proc: int = max(multiprocessing.cpu_count() - 1, 1)
    debug: bool = False


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

    with Map(
        n_proc=cfg.n_proc, debug=cfg.debug, pbar="Extracting NPs with spacy..."
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
