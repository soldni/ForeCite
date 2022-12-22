from multiprocessing.managers import ValueProxy, SyncManager
from time import sleep
from functools import partial, reduce
from typing import (
    Optional, Sequence, Tuple, TypeVar, Callable, Iterable, Union
)
from concurrent.futures import ProcessPoolExecutor, Future
from queue import Queue

from typing_extensions import ParamSpec, Concatenate

from tqdm import tqdm

T = TypeVar('T')
V = TypeVar('V')
P = ParamSpec('P')


class MpTqdm(tqdm):
    def set_total(self, value: int) -> None:
        self.total = value


class Map:
    def __init__(
        self,
        n_proc: int = 1,
        debug: bool = False,
        timeout: Union[int, float] = 0.1,
        pbar: Optional[str] = None
    ) -> None:
        self.timeout = float(timeout)
        self.n_proc = n_proc
        self.pbar = pbar
        self.debug = debug

    @staticmethod
    def _when_completed(
        _: Future,
        pbar: Union[tqdm, None],
        count: ValueProxy
    ):
        count.value -= 1
        if pbar is not None:
            pbar.update(1)

    def __call__(
        self,
        fn: Callable[Concatenate[T, P], V],
        seq: Sequence[T],
        *_: P.args,
        **kwargs: P.kwargs
    ) -> Sequence[V]:
        if self.n_proc == 1 or self.debug:
            _fn = partial(fn, **kwargs)
            return [_fn(x) for x in seq]
        else:

            SyncManager.register('MpTqdm', MpTqdm)

            with ProcessPoolExecutor(self.n_proc) as pool, \
                    SyncManager() as manager:

                count = manager.Value('i', 0)
                pbar: Optional[MpTqdm] = (
                    manager.MpTqdm(total=0, desc=self.pbar)  # type: ignore
                    if self.pbar else None
                )

                response = []
                for elem in seq:
                    count.value += 1
                    if pbar is not None:
                        pbar.set_total(count.value)

                    response.append(
                        f := pool.submit(partial(fn, **kwargs), elem)
                    )
                    f.add_done_callback(
                        partial(self._when_completed, pbar=pbar, count=count)
                    )

                if pbar:
                    pbar.refresh()

                while count.value > 0:
                    sleep(self.timeout)

                if pbar:
                    pbar.close()
                return [f.result() for f in response]


class Reduce:
    def __init__(
        self,
        n_proc: int = 1,
        debug: bool = False,
        timeout: Union[int, float] = 0.1,
        pbar: Optional[str] = None,
    ) -> None:
        """This is a parallel implementation of the reduce function. It
        works by recursively calling a function on pairs of elements in
        the sequence until there is only one element left. The function
        is called in parallel using a process pool unless the debug flag
        is set to True or the number of processes is set to 1.

        Args:
            n_proc (int, optional): Number of processes to use for the
                reduction. Defaults to 1.
            debug (bool, optional): Whether to use multiprocessing to
                perform the reduction; if set to True, the reduction will
                be performed sequentially. Defaults to False.
            timeout (Union[int, float], optional): Time to wait for each
                step of the reduction to complete, in seconds. Defaults to 0.1.
        """
        self.n_proc = n_proc
        self.debug = debug
        self.timeout = float(timeout)
        self.pbar = pbar

    def _partition_queue(self, queue: Queue[T]) -> Iterable[Tuple[T, T]]:
        # keep first element of each pair to yield here until we have popped
        # a second element from the queue.
        first = None

        while not queue.empty():
            elem = queue.get()
            if first is None:
                # we don't have a pair yet, so we store the first element
                first = elem
            else:
                # we have a pair, so we yield it and reset the first element
                yield first, elem
                first = None

        if first is not None:
            # if we have an odd number of elements, we put the last one
            # in the queue to be processed by the next partition
            queue.put(first)

    @staticmethod
    def _when_completed(
        future: Future,
        queue: Queue,
        steps: ValueProxy[int],
        pbar: Optional[tqdm] = None,
    ) -> None:
        # completed the operation, so we can decrement the number of steps
        # and put the result back in the queue
        queue.put(future.result())
        if pbar:
            pbar.update(1)
        steps.value -= 1

    def _num_steps(self, n: int) -> int:
        # number of operations to reduce n elements; should be
        # n - 1 but we compute here just to be sure
        if n < 2:
            return 0
        return n // 2 + self._num_steps(n // 2 + n % 2)

    def __call__(
        self,
        fn: Callable[Concatenate[T, T, P], T],
        seq: Sequence[T],
        *_: P.args,
        **kwargs: P.kwargs
    ) -> T:

        if self.n_proc == 1 or self.debug:
            return reduce(partial(fn, **kwargs), seq)
        else:
            SyncManager.register('tqdm', tqdm)
            with ProcessPoolExecutor(self.n_proc) as pool, \
                    SyncManager() as manager:

                # populate the queue with the elements of the sequence
                queue: Queue[T] = manager.Queue()
                for elem in seq:
                    queue.put(elem)

                # number of steps to perform; we use a manager.Value
                # so that we can access it from the callback function
                # in a separate process. `i` is the type code for an
                # integer.
                steps = manager.Value('i', self._num_steps(queue.qsize()))

                pbar: Optional[tqdm] = (
                    manager.tqdm(   # type: ignore
                        desc=self.pbar, total=steps.value
                    ) if self.pbar is not None else None
                )

                while steps.value > 0:
                    if queue.empty():
                        # wait in case the queue is empty but we still
                        # have steps to do
                        sleep(self.timeout)
                        continue

                    # do pairwise reduction; the callback will take care
                    # of adding the result back into the queue for the next
                    # round of reduction; it will also reduce the number of
                    # steps that we still have to do
                    for one, two in self._partition_queue(queue=queue):
                        f = pool.submit(partial(fn, **kwargs), one, two)
                        f.add_done_callback(
                            partial(
                                self._when_completed,
                                queue=queue,
                                steps=steps,
                                pbar=pbar
                            )
                        )

                if pbar:
                    pbar.close()

                return queue.get()


def f(x, y):
    sleep(.1)
    return x + y


def g(x, y: int = 0):
    sleep(.1)
    return x * 2 + y


if __name__ == '__main__':

    r = Reduce(n_proc=4, debug=False, pbar='reduce')
    o = r(f, range(10))
    print(o)

    m = Map(n_proc=10, debug=False, pbar='map')
    o = m(g, range(10))
    print(o)

    o = r(f, m(g, range(10), y=1))
    print(o)
