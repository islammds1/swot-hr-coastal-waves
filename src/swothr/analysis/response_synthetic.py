#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic sensitivity of the spectral centroid to the instrument power response.

Physical question behind RESPONSE_MIN_MASK / RESPONSE_WARN: where the power
response R(k) is small and steep, the product S(k) R(k) is tilted towards the
better-transmitted (longer, or more oblique) wavenumbers, so the energy-weighted
centroid is biased even with no noise. This script quantifies that bias, with
the SAME response model as the v4.3 pipelines:

    R(k) = [ H3(k.a, d_al) H3(k.r, d_ac) sinc(kx res/2) sinc(ky res/2) ]^2
    H3(k, d) = (1 + 2 cos(k d)) / 3

and also the sensitivity of R itself to a +/-10 % error in the across-track
posting d_ac (the response-model uncertainty).

Directional Gaussian swell spectra are placed on a fine (kx, ky) grid, one lobe
only, with along-track = +y and look direction = +x (so across-track = x).

Outputs
    response_sensitivity_synthetic.csv   one row per case and mask value
    response_sensitivity_synthetic.png   centroid bias against R at the peak

This isolates the LINEAR filter effect only. Velocity bunching, tilt and
layover are not modelled; the real-data counterpart is the error-vs-response
table of swothr.analysis.compare_ww3.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from swothr.config import path, apply_overrides
from swothr.utils.response import (ALONG_TRACK_POSTING, h3, sinc_half,  # noqa: F401
                                   choose_res)
from swothr.utils.response import response_power as _response_power

LAMBDA_MIN, LAMBDA_MAX = 130.0, 400.0
# ---- case grid ------------------------------------------------------------------
D_AC = (15.0, 20.0, 25.0)                 # across-track posting, rho >= 25 km
LAMBDA0 = (150.0, 175.0, 200.0, 250.0, 300.0, 350.0)
PHI_DEG = (0.0, 30.0, 45.0, 60.0, 90.0)   # wave angle from ALONG-track axis
SIGMA_K_REL = (0.05, 0.10, 0.15)          # radial width / k0
SIGMA_TH_DEG = (10.0, 20.0, 30.0)
MASKS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
D_AC_ERR = 0.10                           # response-model uncertainty test
DK = 2.0e-4                               # rad/m grid step (fine)
OUT_CSV = f"{path('root')}/Outputs/synthetic/response_sensitivity_synthetic.csv"
OUT_PNG = f"{path('root')}/Outputs/synthetic/response_sensitivity_synthetic.png"


apply_overrides(globals(), "synthetic", final=True)


def response_power(KX, KY, d_ac, res):
    # along-track = +y, across-track (look) = +x
    return _response_power(KX, KY, d_ac, res)


def centroid(E, KX, KY, sel):
    w = np.where(sel, E, 0.0); ws = w.sum()
    if ws <= 0:
        return np.nan, np.nan
    kx, ky = (w * KX).sum() / ws, (w * KY).sum() / ws
    k = np.hypot(kx, ky)
    lam = 2 * np.pi / k if k > 0 else np.nan
    phi = np.degrees(np.arctan2(kx, ky))   # angle from along-track (+y)
    return lam, phi


def main():
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    kmax = 2 * np.pi / 100.0
    k1 = np.arange(-kmax, kmax + DK, DK)
    KX, KY = np.meshgrid(k1, k1)
    K = np.hypot(KX, KY)
    TH = np.arctan2(KX, KY)                 # from +y
    band = (K >= 2 * np.pi / LAMBDA_MAX) & (K <= 2 * np.pi / LAMBDA_MIN)

    Rcache = {}
    for d_ac in D_AC:
        res = choose_res(d_ac)
        Rcache[d_ac] = (response_power(KX, KY, d_ac, res),
                        response_power(KX, KY, d_ac * (1 - D_AC_ERR), res),
                        response_power(KX, KY, d_ac * (1 + D_AC_ERR), res))
    rows = []
    for d_ac, lam0, phi, sk, sth in itertools.product(D_AC, LAMBDA0, PHI_DEG,
                                                      SIGMA_K_REL, SIGMA_TH_DEG):
        res = choose_res(d_ac)
        k0 = 2 * np.pi / lam0
        dth = np.angle(np.exp(1j * (TH - np.deg2rad(phi))))
        S = np.exp(-0.5 * ((K - k0) / (sk * k0)) ** 2) * np.exp(-0.5 * (dth / np.deg2rad(sth)) ** 2)
        S = np.where(np.abs(dth) < np.pi / 2, S, 0.0)    # one lobe
        R, Rlo, Rhi = Rcache[d_ac]
        E = S * R
        lam_true, phi_true = centroid(S, KX, KY, band)
        for mval in MASKS:
            sel = band & (R >= mval)
            lam_obs, phi_obs = centroid(E, KX, KY, sel)
            w = np.where(sel, E, 0.0)
            r_peak = float((w * R).sum() / w.sum()) if w.sum() > 0 else np.nan
            dR = float((w * np.maximum(np.abs(Rlo - R), np.abs(Rhi - R))).sum() / w.sum()) \
                if w.sum() > 0 else np.nan
            lost = float(1 - (S * R)[sel].sum() / (S * R)[band].sum()) if (S * R)[band].sum() > 0 else np.nan
            rows.append(dict(d_ac=d_ac, res=res, lambda0=lam0, phi_from_along=phi,
                             sigma_k_rel=sk, sigma_th=sth, mask=mval,
                             resp_pow_at_peak=r_peak,
                             resp_model_err_abs=dR,
                             resp_model_err_rel=dR / r_peak if r_peak > 0 else np.nan,
                             lambda_true=lam_true, lambda_obs=lam_obs,
                             dlambda_pct=100 * (lam_obs - lam_true) / lam_true,
                             dphi_deg=float(np.angle(np.exp(1j * np.deg2rad(phi_obs - phi_true)), deg=True)),
                             energy_lost_to_mask=lost))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV, len(df), "rows")

    # summary: bias by R_peak bin, no mask and production mask
    edges = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for mval in (0.0, 0.25):
        d = df[df["mask"] == mval].copy()
        d["_bin"] = pd.cut(d["resp_pow_at_peak"], edges, include_lowest=True)
        t = d.groupby("_bin", observed=True).agg(
            n=("dlambda_pct", "size"),
            dlam_med_pct=("dlambda_pct", "median"),
            dlam_p95_abs_pct=("dlambda_pct", lambda v: np.percentile(np.abs(v), 95)),
            dphi_p95_abs_deg=("dphi_deg", lambda v: np.percentile(np.abs(v), 95)),
            resp_err_rel_med=("resp_model_err_rel", "median"))
        print(f"\n=== mask = {mval:.2f}: centroid bias by response at the peak ===")
        print(t.round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    d0 = df[df["mask"] == 0.25]
    sc = ax[0].scatter(d0["resp_pow_at_peak"], d0["dlambda_pct"], c=d0["sigma_k_rel"],
                       s=8, cmap="viridis")
    ax[0].set_xlabel("power response at peak"); ax[0].set_ylabel(r"$\Delta\lambda$ centroid (%)")
    fig.colorbar(sc, ax=ax[0], label=r"$\sigma_k/k_0$")
    sc = ax[1].scatter(d0["resp_pow_at_peak"], d0["dphi_deg"], c=d0["phi_from_along"],
                       s=8, cmap="twilight")
    ax[1].set_xlabel("power response at peak"); ax[1].set_ylabel(r"$\Delta\phi$ centroid (deg)")
    fig.colorbar(sc, ax=ax[1], label="wave angle from along-track (deg)")
    ax[2].scatter(d0["resp_pow_at_peak"], 100 * d0["resp_model_err_rel"], s=8, c="k")
    ax[2].set_xlabel("power response at peak")
    ax[2].set_ylabel(fr"$|\Delta R|/R$ for $d_{{ac}}\pm${100*D_AC_ERR:.0f}% (%)")
    for a in ax:
        a.axvline(0.5, color="r", ls=":", lw=1); a.grid(alpha=0.3)
    fig.suptitle("Linear-filter centroid bias (mask 0.25), same response model as the pipelines",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()
