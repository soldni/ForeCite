from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from multiprocessing import Process, cpu_count, get_context
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Event
from time import sleep
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from tqdm import tqdm
from typing_extensions import Concatenate, ParamSpec

T = TypeVar("T")
V = TypeVar("V")
P = ParamSpec("P")


class Bag:
    """A bag of callable objects that can be executed in parallel.
    Results are not guaranteed to be in the same order as the callables."""

    def __init__(
        self,
        n_proc: int = cpu_count(),
        debug: bool = False,
        pbar: Optional[str] = None,
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
        self.pbar = pbar

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

        pbar = (
            tqdm(desc=self.pbar, total=len(self._callable))
            if self.pbar else None
        )

        while self._callable:
            try:
                # pop the last callable from the list and call it;
                # append the result to the list of results.
                fn = self._callable.pop(0)
                self._results_queue.put_nowait(fn())
                pbar.update(1) if pbar else None
            except Exception as e:
                # if an exception is raised, call the failure callback
                pbar.close() if pbar else None
                self.callback_if_failure()
                raise e

        # no failures, so call the success callback
        pbar.close() if pbar else None
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
        counter_queue: Queue[int],
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
            counter_queue.put(1)
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
                raise RuntimeError("One or more processes failed.")

            if len(running) == 0 and len(processes) == 0:
                # we have no more processes to run, and all the processes
                # finished successfully. let's call the success callback,
                # set the success event, and return.
                success_event.set()
                callback_if_success()
                return

            # take a break before checking again
            sleep(timeout)

    @staticmethod
    def _progress_bar_thread(
        counter_queue: Queue[int],
        total: int,
        timeout: float,
        exception_event: Event,
        success_event: Event,
        description: Optional[str] = None,
    ):
        """A thread that updates the progress bar. It will stop when the
        success event is set, or when the exception event is set.

        Args:
            counter_queue (Queue[int]): The queue to get the updates from.
            total (int): The total number of updates.
            timeout (float): The timeout to use when checking if the
                progress bar should be updated.
            exception_event (Event): The event to check if an exception
                has been raised.
            success_event (Event): The event to check if all the
                processes have finished successfully.
            description (Optional[str]): The description to show in the
                progress bar. If None, the progress bar is not shown.
        """

        if description is None:
            # no description was provided, so we don't show the progress bar
            return

        pbar = tqdm(total=total, desc=description)

        # loop until we find the stop signal
        while True:
            if exception_event.is_set() or success_event.is_set():
                # found the stop signal, so we can stop the thread
                pbar.close()
                return

            if counter_queue.empty():
                # no updates to the progress bar, so we wait a bit
                sleep(timeout)
                continue
            else:
                # this is how much we should increment the progress bar
                pbar.update(counter_queue.get())

    def _start_progress_bar(self) -> Queue[int]:
        # progress bar has to run in a separate thread, since it is not
        # thread/process safe. We use a queue to communicate with the
        # progress bar thread; the queue is used to send the number of
        # steps to increment the progress bar by.

        counter_queue: Queue[int] = self.manager.Queue()

        self._stack.enter_context(ThreadPoolExecutor(1)).submit(
            partial(
                self._progress_bar_thread,
                counter_queue=counter_queue,
                total=len(self._callable),
                timeout=self.timeout,
                description=self.pbar,
                exception_event=self._exception_event,
                success_event=self._success_event,
            )
        )

        return counter_queue

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

        # create a queue to store the progress in how many processes have
        # finished; this is used to update the progress bar
        counter_queue = self._start_progress_bar()

        # we need to remove the processes that have been added so far, and
        # wrap them into a function that takes care of keeping their return
        # value and signalling any exceptions.
        processes = []
        while self._callable:
            processes.append(
                partial(
                    self._process_wrapper,
                    fn=self._callable.pop(0),
                    results_queue=self._results_queue,
                    exception_event=self._exception_event,
                    counter_queue=counter_queue,
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


class Map(Bag):
    def __call__(
        self,
        fn: Callable[Concatenate[T, P], V],
        seq: Iterable[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[V]:
        for elem in seq:
            self.add(fn, elem, *args, **kwargs)
        self.start(block=True)
        return self.results()


class Reduce(Bag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reduce_success_event = self.manager.Event()
        self._len_sequence_to_reduce: Union[int, None] = None
        self._counter_queue: Union[Queue[int], None] = None

    def _start_progress_bar(self) -> Queue[int]:
        if self._counter_queue is not None:
            # we already showed the progress bar, so we don't need to show
            return self._counter_queue

        self._counter_queue = self.manager.Queue()

        # get the total number of steps to reduce the sequence
        def ts(n: int) -> int:
            return 0 if n < 2 else (n // 2 + ts(n // 2 + n % 2))
        assert self._len_sequence_to_reduce is not None
        total = ts(self._len_sequence_to_reduce)

        # progress bar has to run in a separate thread, since it is not
        # thread/process safe. We use a queue to communicate with the
        # progress bar thread; the queue is used to send the number of
        # steps to increment the progress bar by.
        self._stack.enter_context(ThreadPoolExecutor(1)).submit(
            partial(
                self._progress_bar_thread,
                counter_queue=self._counter_queue,
                total=total,
                timeout=self.timeout,
                description=self.pbar,
                exception_event=self._exception_event,
                success_event=self._reduce_success_event,
            )
        )
        return self._counter_queue

    def __call__(
        self,
        fn: Callable[Concatenate[T, T, P], T],
        seq: Iterable[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:

        seq = list(seq)
        self._len_sequence_to_reduce = len(seq)

        while len(seq) > 1:
            while len(seq) > 1:
                self.add(fn, seq.pop(0), seq.pop(0), *args, **kwargs)
            self.start(block=True)
            seq.extend(self.results())

        self._reduce_success_event.set()
        self._counter_queue = None
        return seq[0]


# def f(x, y):
#     sleep(.1)
#     return x + y


# def g(x, y: int = 0):
#     sleep(.1)
#     return x * 2 + y


# def h(x, do_print: bool = True, do_raise: bool = True):
#     sleep(.01)
#     if do_print:
#         print(f'step: {x}')
#     if x >= 5 and do_raise:
#         raise ValueError('x must be less than or equal to 5')
#     return x


# if __name__ == '__main__':
#     r = Reduce(n_proc=4, debug=False, pbar='reduce')
#     o = r(f, range(10))
#     sleep(2)
#     print(o)

#     m = Map(n_proc=10, debug=False, pbar='map')
#     o = m(g, range(10))
#     sleep(2)
#     print(o)

#     with Map(n_proc=4, pbar='m2') as m, Reduce(n_proc=4, pbar='r2') as r:
#         o = r(f, m(g, range(10), y=1))
#         sleep(2)
#         print(o)

    # with Bag(
    #     n_proc=2,
    #     timeout=.1,
    #     callback_if_success=lambda: print('hello')
    # ) as b:
    #     for i in range(100):
    #         b.add(fn=h, x=i, do_print=False, do_raise=False)
    #     b.start(block=False)
    #     o = b.results()
    #     print(f'result: {o[:5]}...')

    # with Bag(
    #     n_proc=0,
    #     timeout=1,
    #     callback_if_failure=lambda: print('sad'),
    #     callback_if_success=lambda: print('you should not see this')
    # ) as b:
    #     for i in range(6):
    #         b.add(fn=h, x=i)

    #     try:
    #         b.start(block=True)
    #     except Exception:
    #         sleep(1)
    #         print('caught error!')
