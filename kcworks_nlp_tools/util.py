import importlib.util
import os
from contextlib import contextmanager
from timeit import Timer


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


def overwrite(text: str, **kwargs):
    """Overwrite the last line of stdout with a string."""
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"

    print(LINE_UP, end=LINE_CLEAR, flush=True)
    print(text, **kwargs, flush=True)


def get_package_root():
    if __package__:
        spec = importlib.util.find_spec(__package__.split(".")[0])
        if spec is not None and spec.origin is not None:
            return os.path.dirname(spec.origin)  # package's __init__.py dir
    return os.path.dirname(os.path.abspath(__file__))
