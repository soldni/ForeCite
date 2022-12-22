import bisect
import json
from multiprocessing import Manager, cpu_count
from queue import Queue
import os
from time import sleep
from typing import Callable, NamedTuple, Union

import ftfy

import springs as sp
from .io_utils import read_file, write_file, recursively_list_files
from .mp_utils import Bag


LOGGER = sp.configure_logging(__name__, logging_level='INFO')


class EOP:
    '''End of processing sentinel value; after this value is popped from
    a queue, a worker should stop.'''
    def __init__(self, success: bool):
        self.success = success


class NounChunkTuple(NamedTuple):
    text: str
    p_id: int
    year: int


def partition_by_term(term: str, length: int = 2) -> str:
    key = ''.join(
        ch if ch.isascii() and ch.isalnum() else '_'
        for ch in term[:length].lower()
    )
    return key


def partition_by_paper(paper_id: Union[int, str]) -> str:
    return str(paper_id)[:2]


def term_enhance(
    term: str,
    /,
    min_length: int = 3,
    normalize: bool = True,
) -> Union[str, None]:
    if not term[0].isalnum():
        return None
    if len(term) < min_length:
        return None
    if normalize:
        term = ftfy.fix_text(term)
    return term


def term_to_paper_index_worker(
    terms_queue: Queue[Union[NounChunkTuple, EOP]],
    dest: str,
    timeout: float = 1.0,
    partition_fn: Callable[[str], str] = partition_by_term,
    min_papers: int = 1,
):
    '''
    Worker that builds the term to paper index.

    The terms_queue should be a queue of TermPaperTuple tuples.
    The worker will pop items from the queue and write them to the
    destination files. The worker will stop when it pops an EOS object
    from the queue.

    Args:
        terms_queue (Queue[Union[NounChunkTuple, EOS]]): Queue of
            tuples to process. tuples should be of the form (term,
            paper_id, year). The worker will stop when it pops an EOS
            object from the queue.
        dest (str): Destination directory to write the index files to.
            The worker will write two sets of files: ...
        timeout (float, optional): How long to wait for a new item to
            appear in the queue before checking again. Defaults to 1 second.
        partition_fn (Callable[[str], str], optional): Function that
            partitions terms into groups. Defaults to partitioning by
            the first two characters of the term, lowercased and with
            spaces and slashes replaced with underscores.
    '''

    logger = sp.configure_logging(
        logger_name='term_to_paper_index_worker',
        logging_level='INFO'
    )

    term_papers_index: dict[str, dict[str, list[int]]] = {}
    term_years_index: dict[str, dict[str, dict[int, int]]] = {}

    logger.info('Starting worker...')

    while True:
        if terms_queue.empty():
            sleep(timeout)
            continue

        nc = terms_queue.get()

        if isinstance(nc, EOP):
            break

        key = partition_fn(nc.text)

        li = term_papers_index.setdefault(key, {}).setdefault(nc.text, [])
        bisect.insort(li, nc.p_id)
        (
            di := term_years_index.setdefault(key, {}).setdefault(nc.text, {})
        ).setdefault(nc.year, 0)
        di[nc.year] += 1

    logger.info('Finished processing queue; writing files...')
    for key, terms in term_papers_index.items():
        path = os.path.join(dest, 'term_papers', f'{key}.jsonl')
        with write_file(path, skip_if_empty=True, logger=logger) as f:
            for term, papers in terms.items():
                if len(papers) < min_papers:
                    continue
                f.write(json.dumps(dict(term=term, papers=papers)) + '\n')

    for key, terms in term_years_index.items():
        path = os.path.join(dest, 'term_years', f'{key}.jsonl')
        with write_file(path, skip_if_empty=True, logger=logger) as f:
            for term, years in terms.items():
                if sum(years.values()) < min_papers:
                    continue
                f.write(json.dumps(dict(term=term, years=years)) + '\n')

    logger.info('Finished writing files; exiting...')


def process_single_nps_file(
    path: str,
    terms_queue: Queue[Union[NounChunkTuple, EOP]],
    term_enhancer_fn: Callable[[str], str] = term_enhance   # type: ignore
):
    logger = sp.configure_logging(
        logger_name='process_nps_file',
        logging_level='INFO'
    )

    logger.info(f'Processing {path}...')
    with read_file(path) as f:
        for line in f:
            paper = json.loads(line)
            for np in paper['noun_chunks']:
                term = term_enhancer_fn(np)
                if term:
                    nc = NounChunkTuple(
                        text=np, p_id=paper['id'], year=paper['year']
                    )
                    terms_queue.put(nc)

    logger.info(f'Finished processing {path}')


@sp.dataclass
class BuildNpsIndicesConfig:
    src: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases/np/'
    dst: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_forecite_indices/'
    n_proc: int = max(cpu_count() - 1, 2)
    min_papers: int = 3


@sp.cli(BuildNpsIndicesConfig)
def main(cfg: BuildNpsIndicesConfig):
    """
    Let's walk trough the functionality we wish to support by creating
    indexes; that will inform which indices to create.

    To build ForeCite, we want to calculate:

    for each term t:
        for each paper p that contains the term t:
            let y := the year p was published
            f_t = (
                # of paper p'
                where p' is a paper containing t
                and p' is a paper published in y' >= y
            )
            f^p_t = (
                # of times t appears in p'
                where p' is a paper containing t
                and p' is cites p
            )

    indices you need to create:
        - for each term t, a list of papers containing t
        - for each term t, the number of papers containing t per year (f_t)
        - for each paper p, the list of papers that cite p, grouped by year
    """

    assert cfg.n_proc > 1, 'Must have at least two processes assigned'

    # these are the files to process
    np_files = recursively_list_files(cfg.src)

    with Manager() as m:
        terms_queue = m.Queue()

        with Bag(
            # save one core for the writer
            n_proc=(cfg.n_proc - 1),
            # we need to share the manager so that the terms queue can
            # be shared between processes in the two bags
            manager=m,
            callback_if_success=lambda: terms_queue.put(EOP(True)),
            callback_if_failure=lambda: terms_queue.put(EOP(False)),
            error_msg='One of the reader processes failed'
        ) as readers_bag, Bag(
            # one lone core for the writer
            n_proc=1,
            # again, we need to share the manager for the terms queue
            manager=m,
            error_msg='Writer process failed'
        ) as writer_bag:

            writer_bag.add(
                target=term_to_paper_index_worker,
                kwargs=dict(
                    terms_queue=terms_queue,
                    dest=cfg.dst,
                    min_papers=cfg.min_papers
                )
            )

            for f in np_files:
                readers_bag.add(
                    target=process_single_nps_file,
                    kwargs=dict(path=f, terms_queue=terms_queue)
                )

            # we don't care about blocking on a successful read, but
            # we want to wait until the writer is done
            readers_bag.start(block=False)
            writer_bag.start(block=True)


if __name__ == '__main__':
    main()
