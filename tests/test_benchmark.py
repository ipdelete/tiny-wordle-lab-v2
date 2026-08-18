from tiny_wordle.benchmark import run_benchmark


def test_benchmark_entrypoint_is_importable():
    assert callable(run_benchmark)
