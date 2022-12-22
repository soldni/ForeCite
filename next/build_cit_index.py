import bisect
from contextlib import ExitStack, contextmanager
from curses.ascii import isascii
import json
from multiprocessing import Manager, cpu_count, Process
from threading import Semaphore
from queue import Queue
import os
from time import sleep
from typing import Callable, Tuple, Union

import ftfy

from typing_extensions import Concatenate, ParamSpec

import springs as sp
from .io_utils import read_file, write_file, recursively_list_files
from .mp_utils import monitor_processes, Bag


LOGGER = sp.configure_logging(__name__, logging_level='INFO')


class EOP:
    '''End of processing sentinel value; after this value is popped from
    a queue, a worker should stop.'''
    def __init__(self, success: bool):
        self.success = success


def partition_by_paper(paper_id: Union[int, str]) -> str:
    return str(paper_id)[:2]


def paper_to_citation_index_worker(
    citations_queue: Queue[Union[Tuple[int, int, int], EOP]],
    dest: str,
    timeout: float = 1.0,
    partition_fn: Callable[[int], str] = partition_by_paper
):
    '''
    Worker that builds the paper to citation index.

    The citations_queue should be a queue of tuples of the form
    (paper_id, cited_paper_id, year). The worker will pop items from
    the queue and write them to the destination files. The worker will
    stop when it pops an EOS object from the queue.

    Args:
        citations_queue (Queue[Union[Tuple[int, int, int], EOS]]): Queue
            of tuples to process. tuples should be of the form (paper_id,
            cited_paper_id, year). The worker will stop when it pops an
            EOS object from the queue.
        dest (str): Destination directory to write the index files to.
            The worker will write one file: paper_citations.jsonl.
        timeout (float, optional): How long to wait for a new item to
            appear in the queue before checking again. Defaults to 1 second.
        partition_fn (Callable[[int], str], optional): Function that
            partitions papers into groups. Defaults to partitioning by
            the first two numbers of the paper_id.
    '''
    logger = sp.configure_logging(
        logger_name='term_to_paper_index_worker',
        logging_level='INFO'
    )

    paper_cit_index: dict[str, dict[int, dict[int, list[int]]]] = {}

    while True:
        if citations_queue.empty():
            sleep(timeout)
            continue

        pc = citations_queue.get()

        if isinstance(pc, EOP):
            if pc.success:
                break
            else:
                raise RuntimeError('Error processing citations queue')

        paper_id, cited_paper_id, year = pc
        key = partition_fn(paper_id)

        (
            p := paper_cit_index.setdefault(key, {}).setdefault(paper_id, {})
        ).setdefault(year, [])
        p[year].append(cited_paper_id)

    logger.info('Finished processing queue; writing files...')
    for key, papers in paper_cit_index.items():
        path = os.path.join(dest, 'paper_citations', f'{key}.jsonl')
        with write_file(path, skip_if_empty=True) as f:
            for pid, y in papers.items():
                f.write(json.dumps(dict(paper=pid, years=y)) + '\n')

        logger.info(f'Wrote {path} ({len(papers)} papers)')


def process_cit_file(
    path: str,
    citations_queue: Queue[Union[Tuple[int, int, int], EOP]]
):
    logger = sp.configure_logging(
        logger_name='process_cit_file',
        logging_level='INFO'
    )

    logger.info(f'Processing {path}...')
    with read_file(path) as f:
        for line in f:
            cit = json.loads(line)
            for c in cit['cited']:
                citations_queue.put((cit['paper'], cit['cited'], cit['year']))


@sp.dataclass
class BuildCitationIndicesConfig:
    prefix: str = 's3://ai2-s2-lucas/s2orc_20221211/edge_graph/'
    dest: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_forecite_indices/'
    n_proc: int = max(cpu_count() - 1, 2)
    min_papers: int = 3


@sp.cli(BuildCitationIndicesConfig)
def main(cfg: BuildCitationIndicesConfig):
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
    np_files = recursively_list_files(cfg.prefix)

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
                    dest=cfg.dest,
                    min_papers=cfg.min_papers
                )
            )

            for f in np_files:
                readers_bag.add(
                    target=process_nps_file,
                    kwargs=dict(path=f, terms_queue=terms_queue)
                )

            # we don't care about blocking on a successful read, but
            # we want to wait until the writer is done
            readers_bag.start(block=False)
            writer_bag.start(block=True)


if __name__ == '__main__':
    main()
