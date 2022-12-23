from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial, reduce
from multiprocessing import Process, cpu_count, get_context
from multiprocessing.managers import SyncManager, ValueProxy
from queue import Queue
from threading import Event, Lock, Semaphore
from time import sleep
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from tqdm import tqdm
from typing_extensions import Concatenate, ParamSpec

T = TypeVar("T")
V = TypeVar("V")
P = ParamSpec("P")
MAX_INT = 2147483647


class MpTqdm(tqdm):
    def set_total(self, value: int) -> None:
        self.total = value


class Bag:
    """A bag of callable objects that can be executed in parallel.
    Results are not guaranteed to be in the same order as the callables."""

    def __init__(
        self,
        n_proc: int = cpu_count(),
        debug: bool = False,
        timeout: Union[int, float] = 0.1,
        manager: Optional[SyncManager] = None,
        callback_if_success: Optional[Callable] = None,
        callback_if_failure: Optional[Callable] = None,
    ) -> None:
        """Initialize a bag of callables.

        Args:
            n_proc (int, optional): Number of processes to use. Defaults to
                the number of CPUs on the machine.
            debug (bool, optional): If True, run the callables sequentially
                instead of in parallel. Defaults to False.
            pbar (Optional[str], optional): Description for a tqdm progress
                bar. If not provided, no progress bar will be shown. Defaults
                to None.
            timeout (Union[int, float], optional): How long to wait for a
                between checking if the processes are done. Defaults to 0.1
                seconds.
            manager (Optional[SyncManager], optional): A multiprocessing
                manager to use. If none is provided, a new one will be
                created. Defaults to None.
            callback_if_success (Optional[Callable], optional): A callback
                to run if all the callables succeed. Defaults to None.
            callback_if_failure (Optional[Callable], optional): A callback
                to run if any of the callables fail. Defaults to None.
        """

        self.timeout = float(timeout)
        self.n_proc = n_proc if n_proc > 0 else cpu_count()
        self.debug = debug

        self._callable: List[Callable] = []
        self._stack = ExitStack()
        self._futures: List[Future] = []

        self.callback_if_success = callback_if_success or self._null_callback
        self.callback_if_failure = callback_if_failure or self._null_callback

        self.manager = (
            self._stack.enter_context(SyncManager(ctx=get_context()))
            if manager is None
            else manager
        )

        # this queue is used to accumulate the results of the processes
        self._results_queue: Queue[Any] = self.manager.Queue()

        # this event is used to signal that an exception has been raised in
        # one of the processes.
        self._exception_event = self.manager.Event()

        # this event is used to signal that all the processes have finished
        # successfully, and the results can be retrieved.
        self._success_event = self.manager.Event()

    @staticmethod
    def _null_callback():
        """A callback that does nothing; used as a default callback
        if none is provided."""
        pass

    def add(self, fn: Callable, *args, **kwargs) -> None:
        """Add a callable to the bag.

        Args:
            fn (Callable): The callable to add.
            *args: Arguments to pass to the callable.
            **kwargs: Keyword arguments to pass to the callable.
        """
        self._callable.append(partial(fn, *args, **kwargs))

    def _start_debug(self):
        """Run the callables sequentially in the main process.
        Do not use this method directly; use the start() method instead."""

        while self._callable:
            try:
                # pop the last callable from the list and call it;
                # append the result to the list of results.
                fn = self._callable.pop(0)
                self._results_queue.put_nowait(fn())
            except Exception as e:
                # if an exception is raised, call the failure callback
                self.callback_if_failure()
                raise e

        # no failures, so call the success callback
        self.callback_if_success()

        # set the success event so that the results can be retrieved
        self._success_event.set()

    def results(self) -> List[Any]:
        """Get the results of the callables.
        This method will block until all the callables have finished."""

        while not self._success_event.is_set():
            # wait for the success event to be set
            sleep(self.timeout)

        # empty the queue of results and return them
        results = []
        while not self._results_queue.empty():
            results.append(self._results_queue.get())
        return results

    @staticmethod
    def _process_wrapper(
        fn: Callable,
        results_queue: Queue,
        exception_event: Event,
    ):
        """A wrapper for the callables that are run in the processes. It is
        responsible of adding the results to the results queue after the
        callable has finished, and toggling the exception event if an
        exception is raised.

        Args:
            fn (Callable): The callable to run.
            results_queue (Queue): The queue to add the results to.
            exception_event (Event): The event to toggle if an exception
                is raised.
        """
        try:
            results_queue.put(fn())
        except Exception as e:
            exception_event.set()
            raise e

    @staticmethod
    def _start_processes(
        processes: List[Callable],
        n_proc: int,
        timeout: float,
        exception_event: Event,
        success_event: Event,
        callback_if_success: Callable,
        callback_if_failure: Callable,
    ):
        """Start the processes and wait for them to finish. If any fails,
        terminate all the other processes and call the failure callback;
        if all succeed, call the success callback and set the success event
        to allow the results to be retrieved.

        Args:
            processes (List[Callable]): The callables to run.
            n_proc (int): The number of processes to run.
            timeout (float): The timeout to use when checking if the
                processes are done.
            exception_event (Event): The event to toggle if an exception
                is raised.
            success_event (Event): The event to set if all the processes
                succeed.
            callback_if_success (Callable): The callback to call if all
                the processes succeed.
            callback_if_failure (Callable): The callback to call if any
                of the processes fail.
        """

        # keep track of the processes that are running in this list;
        # this ensures that we don't start more processes than we are
        # allowed to, and that we are able to terminate them if any
        # exception is raised.
        running: List[Process] = []

        while True:
            while len(running) < n_proc and len(processes) > 0:
                # we have more processes to run, and we have room for more!
                # let's start a few.
                running.append(p := Process(target=processes.pop(0)))
                p.start()

            for i, p in enumerate(running):
                # let's check if any of the processes has finished
                if not p.is_alive():
                    # oh, the i-th has! let's remove it from the list
                    # of running processes.
                    running.pop(i)

            if exception_event.is_set():
                # uh oh, an exception has been raised in one of the
                # processes. let's terminate all the other processes
                # and call the failure callback.
                for p in running:
                    p.terminate()

                # after the callback is done, we return immediately.
                callback_if_failure()
                return

            if len(running) == 0 and len(processes) == 0:
                # we have no more processes to run, and all the processes
                # finished successfully. let's call the success callback,
                # set the success event, and return.
                success_event.set()
                callback_if_success()
                return

            # take a break before checking again
            sleep(timeout)

    def start(self, block: bool = True):
        """Start the processes.

        Args:
            block (bool): Whether to block the main process until all the
                processes have finished. If False, the main process will
                continue running while the processes are running in the
                background. To retrieve the results, use the results()
                method.
        """

        if self.debug:
            # shortcut to run the functions in the main process
            return self._start_debug()

        # we need to remove the processes that have been added so far, and
        # wrap them into a function that takes care of keeping their return
        # value and signalling any exceptions.
        processes = []
        for fn in self._callable:
            processes.append(
                partial(
                    self._process_wrapper,
                    fn=fn,
                    results_queue=self._results_queue,
                    exception_event=self._exception_event,
                )
            )

        # let's pass all arguments to the executor function; this is needed
        # in case we want to run the processes in a separate thread.
        execute_fn = partial(
            self._start_processes,
            processes=processes,
            n_proc=self.n_proc,
            timeout=self.timeout,
            exception_event=self._exception_event,
            success_event=self._success_event,
            callback_if_success=self.callback_if_success,
            callback_if_failure=self.callback_if_failure,
        )

        if block:
            # block the main process until done.
            execute_fn()
        else:
            # start the processes in a separate thread.
            self._stack.enter_context(ThreadPoolExecutor(1)).submit(execute_fn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.close()


class Map:
    def __init__(
        self,
        n_proc: int = 1,
        debug: bool = False,
        timeout: Union[int, float] = 0.1,
        pbar: Optional[str] = None,
    ) -> None:
        self.timeout = float(timeout)
        self.n_proc = n_proc
        self.pbar = pbar
        self.debug = debug

    @staticmethod
    def _when_completed(_: Future, pbar: Union[tqdm, None], count: ValueProxy):
        count.value -= 1
        if pbar is not None:
            pbar.update(1)

    def __call__(
        self,
        fn: Callable[Concatenate[T, P], V],
        seq: Sequence[T],
        *_: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[V]:
        if self.n_proc == 1 or self.debug:
            _fn = partial(fn, **kwargs)
            return [_fn(x) for x in seq]
        else:

            SyncManager.register("MpTqdm", MpTqdm)

            with ProcessPoolExecutor(
                self.n_proc
            ) as pool, SyncManager() as manager:

                count = manager.Value("i", 0)
                pbar: Optional[MpTqdm] = (
                    manager.MpTqdm(total=0, desc=self.pbar)  # type: ignore
                    if self.pbar
                    else None
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
            queue.put_nowait(first)

    @staticmethod
    def _when_completed(
        future: Future,
        queue: Queue,
        steps: ValueProxy[int],
        pbar: Optional[tqdm] = None,
    ) -> None:
        # completed the operation, so we can decrement the number of steps
        # and put the result back in the queue
        queue.put_nowait(future.result())
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
        **kwargs: P.kwargs,
    ) -> T:

        if self.n_proc == 1 or self.debug:
            return reduce(partial(fn, **kwargs), seq)
        else:
            SyncManager.register("tqdm", tqdm)
            with ProcessPoolExecutor(
                self.n_proc
            ) as pool, SyncManager() as manager:

                # populate the queue with the elements of the sequence
                queue: Queue[T] = manager.Queue()
                for elem in seq:
                    queue.put_nowait(elem)

                # number of steps to perform; we use a manager.Value
                # so that we can access it from the callback function
                # in a separate process. `i` is the type code for an
                # integer.
                steps = manager.Value("i", self._num_steps(queue.qsize()))

                pbar: Optional[tqdm] = (
                    manager.tqdm(  # type: ignore
                        desc=self.pbar, total=steps.value
                    )
                    if self.pbar is not None
                    else None
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
                                pbar=pbar,
                            )
                        )

                if pbar:
                    pbar.close()

                return queue.get()


# def f(x, y):
#     sleep(.1)
#     return x + y


# def g(x, y: int = 0):
#     sleep(.1)
#     return x * 2 + y


# def h(x, do_print: bool = True, do_raise: bool = True):
#     sleep(.1)
#     if do_print:
#         print(f'step: {x}')
#     if x >= 5 and do_raise:
#         raise ValueError('x must be less than or equal to 5')
#     return x


# if __name__ == '__main__':
#     r = Reduce(n_proc=4, debug=False, pbar='reduce')
#     o = r(f, range(10))
#     print(o)

#     m = Map(n_proc=10, debug=False, pbar='map')
#     o = m(g, range(10))
#     print(o)

#     o = r(f, m(g, range(10), y=1))
#     print(o)

#     with Bag(
#         n_proc=2,
#         timeout=.1,
#         callback_if_success=lambda: print('hello')
#     ) as b:
#         for i in range(100):
#             b.add(fn=h, x=i, do_print=False, do_raise=False)
#         b.start(block=False)
#         o = b.results()
#         print(f'result: {o[:5]}...')

#     with Bag(
#         n_proc=0,
#         timeout=1,
#         callback_if_failure=lambda: print('sad'),
#         callback_if_success=lambda: print('you should not see this')
#     ) as b:
#         for i in range(10):
#             b.add(fn=h, x=i)

#         b.start(block=False)

#         sleep(1)
#         print('caught error!')
