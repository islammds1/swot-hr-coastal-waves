#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. S3 -- Sensitivity of the SWOT HR retrievals to the instrument-response
mask threshold, from the sens_mXX_* columns written by the v4.3 FFT pipeline.

For every box the pipeline re-picks the dominant partition with mask
thresholds 0, 0.10, 0.15, 0.20, 0.25 (adopted), 0.30 and 0.40, and stores
lambda, theta, response and energy ratio (relative to 0.25). Boxes are those
flagged usable (internal quality passed, grade A/B) at the adopted threshold.

(a) SWOT - WW3 wavelength bias (median, mean, 95% bootstrap CI)
(b) Per-box wavelength change relative to the 0.25 retrieval
(c) Per-box direction change relative to the 0.25 retrieval
(d) Dominant-partition energy removed by the adopted 0.25 mask
"""

import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swothr.config import path, apply_overrides
from swothr.utils import apply_paper_style


apply_paper_style()

CSV_GLOB = f"{path('root')}/Results/FFT/new/swot_5km_FFT_*_FFT_v4*.csv"
OUTFILE = f"{path('root')}/Results/FFT/new/Figure_S3"
T_MASK, T_FLAG = 0.25, 0.50
N_BOOT = 2000
rng = np.random.default_rng(1)
apply_overrides(globals(), "mask_sensitivity", final=True)


def load():
    frames = []
    for f in sorted(glob.glob(CSV_GLOB)):
        storm = re.search(r"FFT_(.+?)_FFT_v4", f).group(1)
        frames.append(pd.read_csv(f, low_memory=False).assign(storm=storm))
    df = pd.concat(frames, ignore_index=True)
    tags = sorted({c.split("_")[1] for c in df.columns if c.startswith("sens_m")})
    thr = np.array([int(t[1:]) / 100 for t in tags])
    return df, tags, thr


def boot_ci(x, stat=np.median):
    x = x[np.isfinite(x)]
    b = stat(rng.choice(x, (N_BOOT, x.size)), axis=1)
    return np.percentile(b, [2.5, 97.5])


def main():
    df, tags, thr = load()
    u = df[df["usable"].astype(bool)].copy()
    ref = f"m{int(T_MASK * 100):02d}"
    print(f"boxes: {len(df)} total, {len(u)} usable; thresholds {thr}")

    med, mean, ci_med = [], [], []
    frac_dl, p99_dl, max_dl = [], [], []
    frac_dt, p99_dt, max_dt = [], [], []

    for t in tags:
        lam = u[f"sens_{t}_lambda_m"].to_numpy()
        bias = lam - u["ww3_lambda_p_m"].to_numpy()
        bias = bias[np.isfinite(bias)]
        med.append(np.median(bias))
        mean.append(np.mean(bias))
        ci_med.append(boot_ci(bias))

        dl = np.abs(lam - u[f"sens_{ref}_lambda_m"].to_numpy())
        dl = dl[np.isfinite(dl)]
        frac_dl.append(100 * np.mean(dl > 1.0))
        p99_dl.append(np.percentile(dl, 99))
        max_dl.append(dl.max())

        dt = np.abs((u[f"sens_{t}_theta_degN"] - u[f"sens_{ref}_theta_degN"]
                     + 180) % 360 - 180).to_numpy()
        dt = dt[np.isfinite(dt)]
        frac_dt.append(100 * np.mean(dt > 1.0))
        p99_dt.append(np.percentile(dt, 99))
        max_dt.append(dt.max())


    ci_med = np.array(ci_med)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    (a, b), (c, d) = axs

    a.fill_between(thr, ci_med[:, 0], ci_med[:, 1], color="tab:blue", alpha=0.25,
                   label="median, 95% CI")
    a.plot(thr, med, "o-", color="tab:blue", label="median")
    a.plot(thr, mean, "s--", color="tab:blue", mfc="w", label="mean")
    a.set_ylabel("SWOT − WW3 wavelength (m)")
    a.set_title("wavelength bias against WW3")

    b.plot(thr, p99_dl, "o-", color="tab:purple", label="99th percentile")
    b.plot(thr, max_dl, "^:", color="tab:purple", mfc="w", label="maximum")
    b.set_ylabel(r"$|\lambda - \lambda_{0.25}|$ (m)")
    b2 = b.twinx()
    b2.bar(thr, frac_dl, width=0.03, color="0.6", alpha=0.5)
    b2.set_ylabel(r"boxes with $|\Delta\lambda| > 1$ m (%)", color="0.4")
    b.set_title("per-box wavelength change")

    c.plot(thr, p99_dt, "o-", color="tab:orange", label="99th percentile")
    c.plot(thr, max_dt, "^:", color="tab:orange", mfc="w", label="maximum")
    c.set_ylabel(r"$|\theta - \theta_{0.25}|$ (°)")
    c2 = c.twinx()
    c2.bar(thr, frac_dt, width=0.03, color="0.6", alpha=0.5)
    c2.set_ylabel(r"boxes with $|\Delta\theta| > 1°$ (%)", color="0.4")
    c.set_title("per-box direction change")

    m_all = 100 * df["p1_masked_energy_frac"].dropna()
    m_use = 100 * u["p1_masked_energy_frac"].dropna()
    levels = [0.1, 1, 5, 10, 25]
    xs = np.arange(len(levels))
    d.bar(xs - 0.2, [100 * np.mean(m_all > L) for L in levels], 0.4,
          color="0.6", label=f"all boxes (n={m_all.size})")
    d.bar(xs + 0.2, [100 * np.mean(m_use > L) for L in levels], 0.4,
          color="tab:green", label=f"usable boxes (n={m_use.size})")
    d.set_xticks(xs, [f">{L:g}%" for L in levels])
    d.set_xlabel("dominant-partition energy removed by the 0.25 mask")
    d.set_ylabel("boxes (%)")
    d.set_title("energy removed by the mask")

    for ax, lab in zip(axs.flat, "abcd"):
        ax.set_title(f"({lab})", loc="left", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left" if lab != "d" else "upper right")
        if lab == "d":
            continue
        ax.axvline(T_MASK, color="0.3", lw=1, ls=":")
        ax.set_xlabel("power-response mask threshold")
        ax.set_xlim(-0.03, thr.max() + 0.03)

    fig.savefig(OUTFILE + ".jpg", dpi=300)
    #fig.savefig(OUTFILE + ".pdf")

    # numbers for the SM text
    i0, i25 = 0, list(thr).index(T_MASK)
    print(f"bias median: {med[i0]:.2f} m (mask 0) -> {med[i25]:.2f} m (0.25); "
          f"mean {mean[i0]:.2f} -> {mean[i25]:.2f} m")
    for k, t in enumerate(thr):
        print(f"thr {t:.2f}: |dλ|>1m {frac_dl[k]:.1f}%  max {max_dl[k]:.1f} m | "
              f"|dθ|>1° {frac_dt[k]:.1f}%  max {max_dt[k]:.1f}°")
    m = u["p1_masked_energy_frac"]
    print(f"energy masked at 0.25 (usable boxes): mean {100*m.mean():.2f}%, "
          f"median {100*m.median():.2f}%, 95th pct {100*m.quantile(.95):.2f}%, "
          f"max {100*m.max():.1f}%, boxes >1%: {100*(m>0.01).mean():.1f}%")


if __name__ == "__main__":
    main()