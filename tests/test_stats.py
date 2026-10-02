import numpy as np

from swothr.utils.stats import boot_median, binned


def test_boot_median_brackets_median():
    x = np.random.default_rng(3).normal(5, 1, 400)
    med, lo, hi, n = boot_median(x)
    assert n == 400 and lo <= med <= hi and abs(med - 5) < 0.2


def test_small_sample_returns_nan():
    assert np.isnan(boot_median([1, 2, 3])[0])


def test_binned_counts():
    x = np.arange(100.0)
    b = binned(x, x, [0, 50, 100])
    assert list(b["n"]) == [50, 50]
