from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator


class Tee:
    """Write logs to both console and file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def redirect_output(out_path: str, append: bool = False) -> Iterator[None]:
    """Redirect stdout/stderr to both terminal and a line-buffered log file."""
    out_mode = "a" if append else "w"
    out_file = open(out_path, out_mode, encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, out_file)
    sys.stderr = Tee(sys.stderr, out_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        out_file.close()
