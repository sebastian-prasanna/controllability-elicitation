"""Tee a run's stdout/stderr into its own run directory.

Every entrypoint that owns a run folder should wrap its body in ``tee_stdio``
so the log lands *inside* that folder rather than wherever the caller happened
to redirect it. That keeps a run self-describing: config, artifacts and log in
one directory, and no orphan .log files left behind when a run dir is removed.

    with tee_stdio(run_dir / "train.log"):
        ...  # everything printed here also goes to the file

Writes are flushed per call so progress logs are readable live (monitoring
loops tail these). Only Python-level writes are captured: the tee keeps the
real ``fileno()``, so output from child processes goes to the terminal only.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


class _Tee:
    """File-like proxy writing to both a stream and an open file handle."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._fh.write(data)
        self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        # tqdm and friends ask this; answer for the real terminal so
        # interactive rendering is unchanged.
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextmanager
def tee_stdio(path: str | Path, mode: str = "w"):
    """Duplicate sys.stdout/sys.stderr into ``path`` for the block's duration.

    Parent dirs are created. mode="a" appends (use for a second phase writing
    to an existing log). Yields the resolved Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, mode, buffering=1)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(real_out, fh), _Tee(real_err, fh)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        fh.close()
