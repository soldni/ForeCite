from concurrent.futures import ThreadPoolExecutor
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
        self._n_proc = n_proc if n_proc > 0 else cpu_count()
        self._debug = debug
        self.pbar = pbar
        self.stack = ExitStack()

        self._pbar_count = 0
        self._callable: List[Callable] = []

        self._callback_if_success = callback_if_success or self._null_callback
        self._callback_if_failure = callback_if_failure or self._null_callback

        self.manager = (
            self.stack.enter_context(SyncManager(ctx=get_context()))
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

    def add(
        self, fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
    ) -> None:
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
            if self.pbar
            else None
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
                self._callback_if_failure()
                raise e

        # no failures, so call the success callback
        pbar.close() if pbar else None
        self._callback_if_success()

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
        desc: Optional[str] = None,
        unit: str = "it",
        unit_scale: bool = False,
        position: int = 0,
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

        redraw_every_n_checks = steps_until_redraw = 100

        if desc is None:
            # no description was provided, so we don't show the progress bar
            # note that we can't just return here, because we still need to
            # consume the updates from the counter queue.
            pbar = None
        else:
            # create the progress bar with the given parameters
            pbar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                position=position,
            )

        # loop until we find the stop signal
        while True:
            if exception_event.is_set() or success_event.is_set():
                # found the stop signal, so we can stop the thread
                if pbar is not None:
                    sleep(timeout)
                    pbar.close()
                return

            if steps_until_redraw == 0:
                # we have to redraw the progress bar
                if pbar is not None:
                    pbar.refresh()
                steps_until_redraw = redraw_every_n_checks
            steps_until_redraw -= 1

            if counter_queue.empty():
                # no updates to the progress bar, so we wait a bit
                sleep(timeout)
                continue
            else:
                # this is how much we should increment the progress bar
                count = counter_queue.get()
                if pbar is not None:
                    pbar.update(count)

    def add_progress_bar(
        self,
        desc: Union[str, None],
        counter_queue: Optional[Queue[int]] = None,
        unit: str = "it",
        unit_scale: bool = False,
        success_event: Optional[Event] = None,
        total: Optional[int] = None,
    ) -> Queue[int]:
        # progress bar has to run in a separate thread, since it is not
        # thread/process safe. We use a queue to communicate with the
        # progress bar thread; the queue is used to send the number of
        # steps to increment the progress bar by.

        counter_queue = counter_queue or (
            self.manager.Queue() if not self._debug else Queue()
        )
        desc = desc or self.pbar
        success_event = success_event or self._success_event
        total = total or len(self._callable)

        self.stack.enter_context(ThreadPoolExecutor(1)).submit(
            partial(
                self._progress_bar_thread,
                counter_queue=counter_queue,
                total=total,
                timeout=self.timeout,
                desc=desc,
                exception_event=self._exception_event,
                success_event=success_event,
                unit=unit,
                unit_scale=unit_scale,
                position=self._pbar_count,
            )
        )
        self._pbar_count += 1
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

        if self._debug:
            # shortcut to run the functions in the main process
            return self._start_debug()

        # create a queue to store the progress in how many processes have
        # finished; this is used to update the progress bar
        counter_queue = self.add_progress_bar(desc=self.pbar)

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
            n_proc=self._n_proc,
            timeout=self.timeout,
            exception_event=self._exception_event,
            success_event=self._success_event,
            callback_if_success=self._callback_if_success,
            callback_if_failure=self._callback_if_failure,
        )

        if block:
            # block the main process until done.
            return execute_fn()
        else:
            # start the processes in a separate thread.
            return self.stack.enter_context(ThreadPoolExecutor(1)).submit(
                execute_fn
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.close()


class Map(Bag):
    def __call__(
        self,
        fn: Callable[Concatenate[T, P], V],
        seq: Iterable[T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[V]:
        for elem in seq:
            self.add(fn, elem, *args, **kwargs)  # type: ignore
        self.start(block=False)
        return self.results()


class Reduce(Bag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reduce_success_event = self.manager.Event()
        self._len_sequence_to_reduce: Union[int, None] = None
        self._counter_queue: Union[Queue[int], None] = None

    def add_progress_bar(
        self,
        desc: Union[str, None],
        *args,
        **kwargs,
    ) -> Queue[int]:

        if desc != self.pbar:
            return super().add_progress_bar(  # type: ignore
                desc=desc, *args, **kwargs
            )

        if self._counter_queue is not None:
            # we already showed the progress bar, so we don't need to show
            return self._counter_queue

        self._counter_queue = self.manager.Queue()

        # get the total number of steps to reduce the sequence
        def ts(n: int) -> int:
            return 0 if n < 2 else (n // 2 + ts(n // 2 + n % 2))

        assert self._len_sequence_to_reduce is not None
        total = ts(self._len_sequence_to_reduce)

        return super().add_progress_bar(
            desc=desc,
            counter_queue=self._counter_queue,
            total=total,
            success_event=self._reduce_success_event,
        )

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
                self.add(
                    fn,  # type: ignore
                    seq.pop(0),  # type: ignore
                    seq.pop(0),  # type: ignore
                    *args,
                    **kwargs,
                )
            self.start(block=False)
            seq.extend(self.results())

        self._reduce_success_event.set()
        self._counter_queue = None
        return seq[0]
