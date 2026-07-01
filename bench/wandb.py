"""Minimal no-op wandb stub for offline SSD benchmarking (real wandb not
installed on the target). Provides just the surface bench.py uses:
init / log / Histogram / finish."""


class Histogram:
    def __init__(self, *args, **kwargs):
        pass


class _Run:
    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


run = None


def init(*args, **kwargs):
    global run
    run = _Run()
    return run


def log(*args, **kwargs):
    pass


def finish(*args, **kwargs):
    pass
