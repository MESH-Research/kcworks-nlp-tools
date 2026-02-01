from timeit import Timer
from contextlib import contextmanager


@contextmanager
def timed(operation_name: str | None = None):
    """Initializa a timing context manager"""
    timer = Timer()
    start_time: int | None = None

    try:
        start_time = timer.timeit()
        yield

    finally:
        end_time = timer.timeit()
        print(f"** {operation_name + ' ' or ' '}took {end_time - start_time} seconds")
