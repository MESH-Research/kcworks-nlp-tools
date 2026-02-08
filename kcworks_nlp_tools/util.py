import importlib.util
import os
import time
from contextlib import contextmanager


@contextmanager
def timed(operation_name: str | None = None):
    """Context manager that times the block and prints elapsed seconds."""
    start_time = time.perf_counter()
    try:
        print(f"** starting {operation_name}")
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"** {operation_name + ' ' or ' '}took {elapsed} seconds")


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
