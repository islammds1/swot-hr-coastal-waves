"""Robust binned statistics."""
import numpy as np
import pandas as pd


def boot_median(x, n=1000, seed=0, min_n=5):
    """Median and 95 % percentile-bootstrap interval. Returns (med, lo, hi, n)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return np.nan, np.nan, np.nan, int(x.size)
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, x.size)), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi), int(x.size)


def binned(x, y, edges, n_boot=1000):
    """Per-bin median of y with bootstrap CI. Columns: centre, lo_edge, hi_edge,
    median, ci_lo, ci_hi, n. Bins are [lo, hi)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        med, l, h, n = boot_median(y[s], n=n_boot)
        rows.append(dict(centre=0.5 * (lo + hi), lo_edge=lo, hi_edge=hi,
                         median=med, ci_lo=l, ci_hi=h, n=n))
    return pd.DataFrame(rows)
