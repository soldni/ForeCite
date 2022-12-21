from multiprocessing import Pool, Manager

import springs as sp
from .io_utils import read_file, write_file, recursively_list_files
from cached_path import cached_path


def load_vocab(path: str):
    with read_file(cached_path(path)) as f:     # pyright: ignore
        for ln in f:
            ...


@sp.dataclass
class BuildIndicesConfig:
    nps_prefix: str = 's3://ai2-s2-lucas/s2orc_20221211/acl_noun_phrases/'
    cit_prefix: str = 's3://ai2-s2-lucas/s2orc_20221211/edge_graph/'


@sp.cli(BuildIndicesConfig)
def main(cfg: BuildIndicesConfig):
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

    ...


if __name__ == '__main__':
    main()
