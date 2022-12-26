import gzip
import hashlib
import json
import multiprocessing
import os
import string
from collections import Counter
from queue import Queue
from time import sleep
from typing import Counter as CounterType
from typing import Iterable, Optional, Sequence, Set, Union, cast

import spacy
import springs as sp
import unidecode
from necessary import necessary
from spacy.parts_of_speech import DET
from spacy.tokens import Span

from .io_utils import read_file, recursively_list_files, write_file
from .mp_utils import Bag, Map


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

    def process_chunk(self, chunk: Span) -> Union[str, None]:
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
        ).lower()

        if text[0] in self.punctuation:
            # first character is punctuation
            return None

        if not text.isascii():
            text = unidecode.unidecode_expect_nonascii(text)

        if not any(c in self.letters for c in text):
            # no letters
            return None

        return text

    def __call__(self, text: Union[str, Sequence[str]]) -> Iterable[Set[str]]:
        text = [text] if isinstance(text, str) else text
        for doc in self.nlp.pipe(text):
            chunks = set(self.process_chunk(nc) for nc in doc.noun_chunks)
            chunks.discard(None)
            yield cast(Set[str], chunks)


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
    vocab_queue: Queue[Union[CounterType[str], "Stop"]],
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

            noun_chunks = Counter(nc for ncs in nce(texts) for nc in ncs)

            vocab_queue.put(noun_chunks)

            data = dict(
                id=data["id"],
                year=data["year"],
                noun_chunks=sorted(noun_chunks),
            )
            to_write.append(json.dumps(data) + "\n")

    (h := hashlib.md5()).update(path.encode("utf-8"))
    out_filepath = os.path.join(cfg.output, "np", f"{h.hexdigest()}.jsonl")
    with write_file(out_filepath, mode="w") as out_f:
        out_f.writelines(to_write)


@sp.dataclass
class Config:
    prefix: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_content/"
    output: str = "s3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases_ascii/"
    n_proc: int = max(multiprocessing.cpu_count() - 1, 1)
    debug: bool = False
    freq: int = 3


class Stop:
    ...


def build_vocab(
    vocab_queue: Queue[Union[CounterType[str], Stop]],
    pbar_queue: Queue[int],
    output_path: str,
    timeout: float = 0.1,
    min_freq: int = 1,
):
    logger = sp.configure_logging("build_vocab")

    vocab: CounterType[str] = Counter()

    while True:
        if vocab_queue.empty():
            sleep(timeout)

        partial_vocab = vocab_queue.get()
        if isinstance(partial_vocab, Stop):
            break

        pbar_queue.put_nowait(1)
        vocab.update(partial_vocab)

    logger.info("Writing vocab to disk")
    with write_file(os.path.join(output_path, "vocab.txt"), "w") as f:
        for word, count in vocab.most_common():
            if count < min_freq:
                break
            f.write(f"{word}\t{count}\n")
    logger.info("Done writing vocab to disk!")


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
    ) as m:

        b = m.stack.enter_context(Bag(1, manager=m.manager, debug=cfg.debug))
        vocab_queue = m.manager.Queue()
        pbar_queue = m.add_progress_bar(
            desc="Making vocab", unit=" docs", unit_scale=True
        )
        b.add(
            build_vocab,
            vocab_queue=vocab_queue,
            pbar_queue=pbar_queue,
            timeout=0.1,
            output_path=cfg.output,
            min_freq=cfg.freq,
        )

        if not cfg.debug:
            b.start(block=False)

        m(process_single, sources, cfg=cfg, vocab_queue=vocab_queue)

        vocab_queue.put(Stop())

        if cfg.debug:
            b.start()

        b.results()


if __name__ == "__main__":
    main()
