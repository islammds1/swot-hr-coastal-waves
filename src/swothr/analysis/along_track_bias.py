#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Origin of the SWOT HR wavelength bias against MARC/WW3.

Panels
  (a) dλ/λ vs |cross-track|, FFT and Radon, median + 95 % bootstrap CI,
      shaded linear-filter bound from the synthetic test.
  (b) same for FFT inside depth classes (depth-confound control).
  (c) |k|SWOT/|k|WW3 per component (along / across track) vs |cross-track|.
  (d) dλ/λ vs along-track alignment A = |k_al|/|k| of the model wave,
      with the distribution of A.

Bins with fewer than MIN_N boxes are drawn hollow and are excluded from the
printed summary. Sample sizes are written under each FFT point.

Printed / saved:
  - binned statistics (CSV) for every panel;
  - consistency check: bias predicted from the component ratios,
    sqrt(A² r_al² + (1 − A²) r_ac²), against the observed bias per A bin;
  - correlation of A with cross-track distance and depth (confound check);
  - multiple regression of dλ/λ on standardised cross-track distance, depth,
    alignment and (if available) model steepness.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swothr.config import path, apply_overrides
from swothr.utils import apply_paper_style, boot_median as _boot_median, binned as _binned

apply_paper_style()


# =============================================================================
# SETTINGS
# =============================================================================
ROOT = path("root")             # config/paths.yaml
PERBOX = os.path.join(ROOT, "Outputs", "comparison_ww3", "new", "fft_radon_ww3_perbox.csv")
DIR = {"F": os.path.join(ROOT, "Results", "FFT", "new"),
       "R": os.path.join(ROOT, "Results", "radon", "new")}
TEMPLATE = "swot_5km_{est}_{stem}_{est}_v4.3.csv"
EST = {"F": "FFT", "R": "RADON"}
OUT_DIR = os.path.join(ROOT, "Outputs", "comparison_ww3", "new")
OUT_PNG = os.path.join(OUT_DIR, "fig_wavelength_bias_origin.png")
OUT_CSV = os.path.join(OUT_DIR, "fig_wavelength_bias_origin_bins.csv")

KEY = ["Date", "Cycle", "Pass", "grid_ix", "grid_iy"]
GRADES = ("A", "B")
CT_EDGES = [10, 20, 30, 40, 50, 60]            # km
DEPTH_EDGES = [0, 30, 50, 200]                   # m
A_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MIN_N = 30                                       # bins below this are hollow / not summarised
MIN_COMPONENT = 0.25                             # component ratio only if |k_c| >= this * |k|
LINEAR_FILTER_BOUND = 3.2                        # % (synthetic 95th percentile)
N_BOOT = 1000
DPI = 300
GEOM = ["along_track_degN", "look_dir_degN", "k_along_obs_radm",
        "k_across_obs_radm", "ww3_Hs_m"]

apply_overrides(globals(), "bias", final=True)

COL = {"F": "tab:blue", "R": "#c1670c"}
LAB = {"F": "FFT", "R": "Radon"}
MRK = {"F": "o", "R": "^"}
DEPTH_COL = ("tab:purple", "tab:green", "tab:gray")


# =============================================================================
# HELPERS
# =============================================================================
def boot_median(x, n=N_BOOT, seed=0):
    return _boot_median(x, n=n, seed=seed)


def binned(x, y, edges):
    """DataFrame: centre, median, lo, hi, n."""
    return _binned(x, y, edges, n_boot=N_BOOT)


def plot_binned(ax, b, color, marker, label, offset=0.0, ls="-", annotate=False,
                drop_small=False):
    """
    Filled markers for n >= MIN_N, hollow for smaller bins. The connecting line
    runs through the reliable bins only. drop_small=True hides small bins.
    """
    b = b[np.isfinite(b["median"])]
    if drop_small:
        b = b[b["n"] >= MIN_N]
    if b.empty:
        return
    x = b["centre"].to_numpy() + offset
    y = b["median"].to_numpy()
    err = [y - b["ci_lo"].to_numpy(), b["ci_hi"].to_numpy() - y]
    ok = b["n"].to_numpy() >= MIN_N
    ax.plot(x[ok], y[ok], ls=ls, color=color, lw=1.5, zorder=2)
    ax.errorbar(x[ok], y[ok], yerr=[err[0][ok], err[1][ok]], fmt=marker, ms=6,
                color=color, capsize=3, label=label, zorder=3)
    if (~ok).any():
        ax.errorbar(x[~ok], y[~ok], yerr=[err[0][~ok], err[1][~ok]], fmt=marker, ms=6,
                    mfc="white", color=color, capsize=3, alpha=0.7, zorder=3)
    if annotate:
        for xx, lo, n in zip(x, b["ci_lo"].to_numpy(), b["n"].to_numpy()):
            ax.annotate(f"n={int(n)}", (xx, lo), xytext=(0, -11), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="0.35")


def unit(bearing_deg):
    t = np.deg2rad(np.asarray(bearing_deg, float))
    return np.sin(t), np.cos(t)                  # (east, north)


def main():
    # =============================================================================
    # DATA
    # =============================================================================
    pb = pd.read_csv(PERBOX)
    for c in KEY[:3]:
        pb[c] = pb[c].astype(str)

    geo = {"F": [], "R": []}
    for s in ("F", "R"):
        for stem in pb["stem"].unique():
            p = os.path.join(DIR[s], TEMPLATE.format(est=EST[s], stem=stem))
            if not os.path.exists(p):
                print(f"[data] missing {os.path.basename(p)}")
                continue
            d = pd.read_csv(p, usecols=lambda c: c in KEY + GEOM)
            for c in KEY[:3]:
                d[c] = d[c].astype(str)
            geo[s].append(d.rename(columns={c: f"{c}_{s}" for c in GEOM}).assign(stem=stem))
    if not geo["F"] or not geo["R"]:
        raise SystemExit("pipeline CSVs not found for one of the estimators")
    m = (pb.merge(pd.concat(geo["F"]), on=KEY + ["stem"], how="left")
           .merge(pd.concat(geo["R"]), on=KEY + ["stem"], how="left"))

    good = (m["instrument_observability_F"].isin(GRADES) & m["instrument_observability_R"].isin(GRADES)
            & np.isfinite(m["ww3_lambda_m"]) & np.isfinite(m["ww3_dir_degN"]))
    m = m[good].copy()
    m["ct_km"] = np.abs(m["cross_track_m"]) / 1000.0
    print(f"[data] {len(m)} common grade A/B boxes with model collocation")

    # model wavevector in each box's own along-track / look frame
    k_m = 2 * np.pi / m["ww3_lambda_m"].to_numpy(float)
    th_to = (m["ww3_dir_degN"].to_numpy(float) + 180.0) % 360.0          # WW3 'from' -> towards
    ke, kn = k_m * np.sin(np.deg2rad(th_to)), k_m * np.cos(np.deg2rad(th_to))
    for s in ("F", "R"):
        ae, an = unit(m[f"along_track_degN_{s}"])
        re_, rn = unit(m[f"look_dir_degN_{s}"])
        kal = np.abs(ke * ae + kn * an)
        kac = np.abs(ke * re_ + kn * rn)
        m[f"kal_mod_{s}"], m[f"kac_mod_{s}"] = kal, kac
        r_al = np.abs(m[f"k_along_obs_radm_{s}"].to_numpy(float)) / kal
        r_ac = np.abs(m[f"k_across_obs_radm_{s}"].to_numpy(float)) / kac
        m[f"ratio_al_{s}"] = np.where(kal >= MIN_COMPONENT * k_m, r_al, np.nan)
        m[f"ratio_ac_{s}"] = np.where(kac >= MIN_COMPONENT * k_m, r_ac, np.nan)
        m[f"dlam_rel_{s}"] = 100 * (m[f"lambda_obs_m_{s}"] - m["ww3_lambda_m"]) / m["ww3_lambda_m"]
    m["A_model"] = m["kal_mod_F"] / k_m

    # =============================================================================
    # BINNED STATISTICS
    # =============================================================================
    tabs = []
    B = {}
    for s in ("F", "R"):
        B[("a", s)] = binned(m["ct_km"], m[f"dlam_rel_{s}"], CT_EDGES)
        B[("d", s)] = binned(m["A_model"], m[f"dlam_rel_{s}"], A_EDGES)
        for comp in ("al", "ac"):
            B[("c", s, comp)] = binned(m["ct_km"], m[f"ratio_{comp}_{s}"], CT_EDGES)
    d = m["Depth_m"].to_numpy(float)
    for lo, hi in zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:]):
        sel = (d >= lo) & (d < hi)
        B[("b", f"{lo}-{hi}")] = binned(m.loc[sel, "ct_km"], m.loc[sel, "dlam_rel_F"], CT_EDGES)
    for k, v in B.items():
        tabs.append(v.assign(panel=k[0], series="_".join(k[1:])))
    pd.concat(tabs, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")

    # =============================================================================
    # FIGURE
    # =============================================================================
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.2))

    a = ax[0, 0]
    a.axhspan(0, LINEAR_FILTER_BOUND, color="0.87", label="linear-filter bound (synthetic)", zorder=0)
    for s, off in (("F", -0.6), ("R", 0.6)):
        plot_binned(a, B[("a", s)], COL[s], MRK[s], LAB[s], offset=off, annotate=(s == "F"))
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("|cross-track distance| (km)")
    a.set_ylabel(r"median $\Delta\lambda/\lambda_{WW3}$ (%)")
    a.set_title("(a) wavelength bias across the swath")
    a.legend(fontsize=8, loc="upper right")

    a = ax[0, 1]
    for (lo, hi), c in zip(zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:]), DEPTH_COL):
        plot_binned(a, B[("b", f"{lo}-{hi}")], c, "s", f"depth {lo}–{hi} m")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("|cross-track distance| (km)")
    a.set_ylabel(r"median $\Delta\lambda/\lambda_{WW3}$ (%), FFT")
    a.set_title("(b) FFT, within depth classes")
    a.legend(fontsize=8, loc="lower left")

    a = ax[1, 0]
    for s, ls in (("F", "-"), ("R", "--")):
        plot_binned(a, B[("c", s, "al")], "tab:red", MRK[s], f"{LAB[s]} along-track", ls=ls,
                    offset=-0.4 if s == "F" else 0.4)
        plot_binned(a, B[("c", s, "ac")], "tab:blue", MRK[s], f"{LAB[s]} across-track", ls=ls,
                    offset=-0.4 if s == "F" else 0.4)
    a.axhline(1, color="k", lw=0.8)
    a.axhline(0.77, color="tab:red", lw=0.9, ls=":", label="Yu et al. (2026) along-track median")
    a.set_xlabel("|cross-track distance| (km)")
    a.set_ylabel(r"$|k_c|_{SWOT}\,/\,|k_c|_{WW3}$")
    a.set_title("(c) wavenumber components, SWOT / model")
    a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
             frameon=False)

    a = ax[1, 1]
    a2 = a.twinx()
    a2.hist(m["A_model"].dropna(), bins=A_EDGES, color="0.85", edgecolor="0.7", zorder=0)
    a2.set_ylabel("boxes", color="0.4")
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    for s, off in (("F", -0.01), ("R", 0.01)):
        plot_binned(a, B[("d", s)], COL[s], MRK[s], LAB[s], offset=off, drop_small=True)
    vals = pd.concat([B[("d", s)] for s in ("F", "R")])
    vals = vals[vals["n"] >= MIN_N]
    if not vals.empty:
        top = np.nanmax(vals["ci_hi"]); bot = np.nanmin(vals["ci_lo"])
        a.set_ylim(min(-2, bot - 2), top + 3)
    a.axhline(0, color="k", lw=0.8)
    a.set_xlim(0, 1)
    a.set_xlabel(r"along-track alignment of the model wave  $A=|k_{al}|/|k|$")
    a.set_ylabel(r"median $\Delta\lambda/\lambda_{WW3}$ (%)")
    a.set_title("(d) bias against along-track alignment")
    a.legend(fontsize=8, loc="upper right")

    fig.suptitle("Origin of the SWOT wavelength bias against MARC/WW3", fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96], h_pad=2.5)
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[save] {OUT_PNG}")

    # =============================================================================
    # NUMBERS FOR THE TEXT
    # =============================================================================
    def ok(b):
        return b[b["n"] >= MIN_N]

    print("\n=== (a) bias across the swath (bins with n >= MIN_N) ===")
    for s in ("F", "R"):
        b = ok(B[("a", s)])
        if len(b) >= 2:
            inner, outer = b["median"].iloc[0], b[b["centre"] >= 40]["median"].mean()
            print(f"{LAB[s]:5s}: {inner:.1f} % at {b['centre'].iloc[0]:.0f} km, "
                  f"{outer:.1f} % beyond 40 km, ratio {inner/outer:.2f}")

    print("\n=== (c) component ratios ===")
    for s in ("F", "R"):
        for comp, name in (("al", "along"), ("ac", "across")):
            b = ok(B[("c", s, comp)])
            print(f"{LAB[s]:5s} {name:6s}: " +
                  "  ".join(f"{c:.0f} km {v:.2f}" for c, v in zip(b["centre"], b["median"])))

    A = m["A_model"].dropna()
    print(f"\n=== (d) alignment: median A = {A.median():.2f}; "
          f"A < 0.4 in {100*np.mean(A < 0.4):.0f} %; A < 0.5 in {100*np.mean(A < 0.5):.0f} % ===")
    for s in ("F", "R"):
        b = ok(B[("d", s)])
        print(f"{LAB[s]:5s}: " + "  ".join(f"A={c:.1f} {v:.1f} %" for c, v in zip(b["centre"], b["median"])))

    print("\n=== consistency: bias predicted from the component ratios vs observed ===")
    for s in ("F", "R"):
        r_al = np.nanmedian(m[f"ratio_al_{s}"]); r_ac = np.nanmedian(m[f"ratio_ac_{s}"])
        rows = []
        for _, r in ok(B[("d", s)]).iterrows():
            A_c = r["centre"]
            k_ratio = np.sqrt(A_c**2 * r_al**2 + (1 - A_c**2) * r_ac**2)
            rows.append(f"A={A_c:.1f}: pred {100*(1/k_ratio - 1):.1f} % / obs {r['median']:.1f} %")
        print(f"{LAB[s]:5s} (r_al={r_al:.2f}, r_ac={r_ac:.2f}): " + "; ".join(rows))

    print("\n=== confound check: Spearman correlation of A with cross-track and depth ===")
    print(m[["A_model", "ct_km", "Depth_m"]].corr(method="spearman").round(2).to_string())

    try:
        import statsmodels.formula.api as smf
        for s in ("F", "R"):
            cov = ["ct_km", "Depth_m", "A_model"]
            dd = m[[f"dlam_rel_{s}", "ww3_Hs_m_F", "ww3_lambda_m"] + cov].copy()
            if dd["ww3_Hs_m_F"].notna().sum() > 20:
                dd["steep"] = dd["ww3_Hs_m_F"] / dd["ww3_lambda_m"]; cov.append("steep")
            dd = dd[[f"dlam_rel_{s}"] + cov].dropna()
            cov = [c for c in cov if dd[c].std() > 0]
            if len(dd) < 10 or not cov:
                print(f"{LAB[s]}: too few boxes for the regression"); continue
            for c in cov:
                dd[c + "_z"] = (dd[c] - dd[c].mean()) / dd[c].std()
            f = smf.ols(f"dlam_rel_{s} ~ " + " + ".join(c + "_z" for c in cov), dd).fit()
            print(f"\n=== {LAB[s]}: dλ/λ (%) on standardised covariates "
                  f"(n={len(dd)}, R² = {f.rsquared:.2f}) ===")
            print(f.summary().tables[1])
    except ImportError:
        print("\n[reg] statsmodels not installed: pip install statsmodels --break-system-packages")


if __name__ == "__main__":
    main()
