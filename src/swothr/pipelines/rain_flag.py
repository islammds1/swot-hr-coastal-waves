# -*- coding: utf-8 -*-


import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj

from swothr.config import path, apply_overrides

ROOT = path("root")             # config/paths.yaml


# ============================================================
# USER SETTINGS
# ============================================================

HR_FILES = rf"{ROOT}/Data/SWOT HR/Nelson_42/SWOT_L2_HR_PIXC_*.nc"
HR_GROUP = "pixel_cloud"

# Concurrent L2_LR_SSH Expert granule for the SAME cycle and pass.
# Set to None to skip validation and only write the HR predictors.
LR_FILE = rf"{ROOT}/Data/SWOT_LR/SWOT_L2_LR_SSH_Expert_013_042_*.nc"

OUT_CSV = Path(f"{ROOT}/Outputs/rain_screen_Nelson_42.csv")

UTM_ZONE = 30

# Must match FFT_Methodology_Final.py so the box keys line up.
BOX_SIZE = 5000.0
BOX_STRIDE = 5000.0
RES = 30.0
MIN_RAW_PIXELS_BOX = 350

CROSS_TRACK_MIN = 10_000.0
CROSS_TRACK_MAX = 60_000.0

GEOLOC_QUAL_MAX = 3          # pixels at or above this count as failed

CT_CLIM_BIN = 5_000.0        # cross-track bin width for the scene climatology
SHORT_SMOOTH_CELLS = 5       # boxcar width in RES cells for the high-pass
BLOCK_M = 1000.0             # block size for the ~1 km variability statistic

# Predictors entering hr_score, with the sign that makes "high = contaminated".
SCORE_TERMS = {
    "hr_density_ratio": -1.0,
    "hr_cellfill": -1.0,
    "hr_qual_frac": +1.0,
    "hr_sig0_std_1km": +1.0,
    "hr_sig0_std_short": +1.0,
    "hr_ssh_std_short": +1.0,
}


# ============================================================
# HELPERS
# ============================================================

apply_overrides(globals(), ("common", "rain_flag"), final=True)
OUT_CSV = Path(OUT_CSV)


def robust_z(v):
    """Median/MAD z-score, NaN-safe."""
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(v)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(v)
    return (v - med) / scale


def ct_climatology_ratio(values, ct_abs, log_space=False):
    """
    Normalise a per-box quantity against its own cross-track climatology.

    Returns value / median(value in that cross-track bin), or a difference in
    dB if log_space is True. Bins with fewer than 5 boxes fall back to the
    global median.
    """
    values = np.asarray(values, dtype=float)
    ct_abs = np.asarray(ct_abs, dtype=float)

    out = np.full(values.shape, np.nan)
    gl = np.nanmedian(values)

    edges = np.arange(0.0, np.nanmax(ct_abs) + CT_CLIM_BIN, CT_CLIM_BIN)
    idx = np.digitize(ct_abs, edges)

    for b in np.unique(idx[np.isfinite(ct_abs)]):
        sel = (idx == b) & np.isfinite(values)
        if sel.sum() >= 5:
            ref = np.nanmedian(values[sel])
        else:
            ref = gl
        if not np.isfinite(ref):
            continue
        if log_space:
            out[idx == b] = values[idx == b] - ref
        elif ref > 0:
            out[idx == b] = values[idx == b] / ref

    return out


def boxcar2d(Z, n):
    """Separable NaN-aware boxcar mean of width n cells."""
    valid = np.isfinite(Z).astype(float)
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    k = np.ones(n) / n

    def smooth_axis(A, axis):
        return np.apply_along_axis(
            lambda m: np.convolve(m, k, mode="same"), axis, A)

    num = smooth_axis(smooth_axis(Zf, 0), 1)
    den = smooth_axis(smooth_axis(valid, 0), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den

    out[den < 0.25] = np.nan
    return out


def plane_residual_std(Z):
    """Std of Z after removing a fitted plane. NaN if too few valid cells."""
    ok = np.isfinite(Z)
    if ok.sum() < 20:
        return np.nan
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.c_[xx[ok].ravel(), yy[ok].ravel(), np.ones(ok.sum())]
    beta, *_ = np.linalg.lstsq(A, Z[ok].ravel(), rcond=None)
    fit = beta[0] * xx + beta[1] * yy + beta[2]
    return float(np.nanstd(Z - fit))


def auc_score(scores, labels):
    """
    Rank-based AUC (Mann-Whitney). Returns (auc, n_pos, n_neg).
    Higher score is assumed to predict label == True.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]

    n1 = int(labels.sum())
    n0 = int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, n1, n0

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1

    auc = (ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc), n1, n0


def best_youden(scores, labels):
    """Threshold maximising POD - FAR_rate. Returns (thr, pod, far_rate)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]

    if labels.sum() == 0 or (~labels).sum() == 0:
        return np.nan, np.nan, np.nan

    best = (-np.inf, np.nan, np.nan, np.nan)
    for thr in np.unique(scores):
        pred = scores >= thr
        pod = pred[labels].mean()
        fpr = pred[~labels].mean()
        j = pod - fpr
        if j > best[0]:
            best = (j, thr, pod, fpr)

    return float(best[1]), float(best[2]), float(best[3])


# ============================================================
# LOAD HR SCENE
# ============================================================

def _preprocess_pixc(ds):
    if "num_pixc_lines" in ds.dims:
        ds = ds.drop_dims("num_pixc_lines", errors="ignore")
    keep = ["latitude", "longitude", "height", "sig0",
            "geolocation_qual", "cross_track"]
    return ds[[v for v in keep if v in ds.variables]]


def load_hr(pattern, group, proj):
    files = sorted(glob.glob(pattern)) if isinstance(pattern, str) else list(pattern)
    if not files:
        raise FileNotFoundError(f"no HR granules match {pattern}")

    ds = xr.open_mfdataset(files, group=group, combine="nested",
                           concat_dim="points", preprocess=_preprocess_pixc,
                           mask_and_scale=True)

    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    ssh = np.asarray(ds["height"].values, dtype=float)
    sig = np.asarray(ds["sig0"].values, dtype=float)
    qual = np.asarray(ds["geolocation_qual"].values, dtype=float)
    ct = np.asarray(ds["cross_track"].values, dtype=float)

    ds.close()

    # NOTE: only geometry-invalid pixels are dropped here. Quality-failed and
    # sig0-failed pixels are KEPT so that dropout can be measured; that is the
    # whole point of the screen.
    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ct)
    keep &= (np.abs(ct) >= CROSS_TRACK_MIN) & (np.abs(ct) <= CROSS_TRACK_MAX)

    lat, lon, ssh, sig, qual, ct = (a[keep] for a in
                                    (lat, lon, ssh, sig, qual, ct))

    x, y = proj(lon, lat)

    return {"x": np.asarray(x), "y": np.asarray(y), "lat": lat, "lon": lon,
            "ssh": ssh, "sig0": sig, "qual": qual, "cross_track": ct}


# ============================================================
# PER-BOX HR PREDICTORS
# ============================================================

def hr_box_predictors(data):
    x, y = data["x"], data["y"]

    x0 = np.floor(np.nanmin(x) / BOX_SIZE) * BOX_SIZE
    x1 = np.ceil(np.nanmax(x) / BOX_SIZE) * BOX_SIZE
    y0 = np.floor(np.nanmin(y) / BOX_SIZE) * BOX_SIZE
    y1 = np.ceil(np.nanmax(y) / BOX_SIZE) * BOX_SIZE

    nb = int(round(BLOCK_M / RES))
    rows = []

    for xx in np.arange(x0, x1, BOX_STRIDE):
        for yy in np.arange(y0, y1, BOX_STRIDE):

            inside = ((x >= xx) & (x <= xx + BOX_SIZE) &
                      (y >= yy) & (y <= yy + BOX_SIZE))

            n_raw = int(inside.sum())
            if n_raw < MIN_RAW_PIXELS_BOX:
                continue

            xb, yb = x[inside], y[inside]
            sshb = data["ssh"][inside]
            sigb = data["sig0"][inside]
            qb = data["qual"][inside]
            ctb = data["cross_track"][inside]

            xe = np.arange(xx, xx + BOX_SIZE + RES, RES)
            ye = np.arange(yy, yy + BOX_SIZE + RES, RES)

            cnt, _, _ = np.histogram2d(yb, xb, bins=[ye, xe])

            fin_s = np.isfinite(sigb)
            fin_h = np.isfinite(sshb)

            cs, _, _ = np.histogram2d(yb[fin_s], xb[fin_s], bins=[ye, xe])
            sg, _, _ = np.histogram2d(yb[fin_s], xb[fin_s], bins=[ye, xe],
                                      weights=sigb[fin_s])
            ch, _, _ = np.histogram2d(yb[fin_h], xb[fin_h], bins=[ye, xe])
            sh, _, _ = np.histogram2d(yb[fin_h], xb[fin_h], bins=[ye, xe],
                                      weights=sshb[fin_h])

            sig_g = np.full_like(sg, np.nan)
            ssh_g = np.full_like(sh, np.nan)
            sig_g[cs > 0] = sg[cs > 0] / cs[cs > 0]
            ssh_g[ch > 0] = sh[ch > 0] / ch[ch > 0]

            with np.errstate(invalid="ignore", divide="ignore"):
                sig_db = 10.0 * np.log10(np.where(sig_g > 0, sig_g, np.nan))

            # ---- dropout ----
            area_km2 = (BOX_SIZE / 1000.0) ** 2
            density = n_raw / area_km2
            cellfill = float(np.mean(cnt >= 1))
            qual_frac = float(np.mean(qb >= GEOLOC_QUAL_MAX)) if n_raw else np.nan
            sig0_fill = float(np.mean(~fin_s))

            # ---- variance: ~1 km blocks, plane removed ----
            stds = []
            ny, nx = sig_db.shape
            for i in range(0, ny - nb + 1, nb):
                for j in range(0, nx - nb + 1, nb):
                    s = plane_residual_std(sig_db[i:i + nb, j:j + nb])
                    if np.isfinite(s):
                        stds.append(s)
            sig0_std_1km = float(np.median(stds)) if stds else np.nan

            # ---- variance: sub-wave-band (high-pass, ~150 m) ----
            sig_hp = sig_db - boxcar2d(sig_db, SHORT_SMOOTH_CELLS)
            ssh_hp = ssh_g - boxcar2d(ssh_g, SHORT_SMOOTH_CELLS)
            sig0_std_short = float(np.nanstd(sig_hp))
            ssh_std_short = float(np.nanstd(ssh_hp))

            rows.append({
                "grid_ix": int(round(xx / BOX_SIZE)),
                "grid_iy": int(round(yy / BOX_SIZE)),
                "x_min": float(xx), "y_min": float(yy),
                "x_c": float(xx + BOX_SIZE / 2.0),
                "y_c": float(yy + BOX_SIZE / 2.0),
                "n_raw_pixels": n_raw,
                "cross_track_mean": float(np.nanmean(ctb)),
                "_density": density,
                "hr_cellfill": cellfill,
                "hr_qual_frac": qual_frac,
                "hr_sig0_fill": sig0_fill,
                "_sig0_std_1km": sig0_std_1km,
                "_sig0_std_short": sig0_std_short,
                "_ssh_std_short": ssh_std_short,
                "_sig0_db": float(np.nanmedian(sig_db)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ct_abs = np.abs(df["cross_track_mean"].values)

    df["hr_density_ratio"] = ct_climatology_ratio(df["_density"], ct_abs)
    df["hr_sig0_std_1km"] = ct_climatology_ratio(df["_sig0_std_1km"], ct_abs)
    df["hr_sig0_std_short"] = ct_climatology_ratio(df["_sig0_std_short"], ct_abs)
    df["hr_ssh_std_short"] = ct_climatology_ratio(df["_ssh_std_short"], ct_abs)
    df["hr_sig0_anom_db"] = ct_climatology_ratio(df["_sig0_db"], ct_abs,
                                                 log_space=True)

    z = np.zeros(len(df))
    n = 0
    for col, sgn in SCORE_TERMS.items():
        if col in df and np.isfinite(df[col].values).sum() > 5:
            z = z + sgn * robust_z(df[col].values)
            n += 1
    df["hr_score"] = z / max(n, 1)

    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


# ============================================================
# LR COLLOCATION
# ============================================================

def _bit(ds_raw, var, name):
    """Boolean array for a named bit of a LR quality bitfield."""
    if var not in ds_raw.variables:
        return None
    attrs = ds_raw[var].attrs
    names = attrs.get("flag_meanings", "").split()
    masks = attrs.get("flag_masks", None)
    if masks is None or name not in names:
        return None
    m = np.uint64(int(masks[names.index(name)]))
    return (ds_raw[var].values.astype(np.uint64) & m) > 0


def load_lr_labels(lr_file, proj):
    files = sorted(glob.glob(lr_file)) if isinstance(lr_file, str) else [lr_file]
    if not files:
        raise FileNotFoundError(f"no LR granule matches {lr_file}")
    path = files[0]

    ds = xr.open_dataset(path)
    raw = xr.open_dataset(path, mask_and_scale=False)

    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    rain_flag = np.asarray(ds["rain_flag"].values, dtype=float)
    rain_rate = np.asarray(ds["rain_rate"].values, dtype=float)
    sclass = np.asarray(ds["ancillary_surface_classification_flag"].values,
                        dtype=float)

    b_rain = _bit(raw, "swh_karin_qual", "suspect_rain_likely")
    b_wstd = _bit(raw, "sig0_karin_qual", "suspect_large_nrcs_window_std")
    b_delt = _bit(raw, "sig0_karin_qual", "suspect_large_nrcs_delta")

    zeros = np.zeros_like(rain_flag, dtype=bool)
    b_rain = zeros if b_rain is None else b_rain
    b_wstd = zeros if b_wstd is None else b_wstd
    b_delt = zeros if b_delt is None else b_delt

    ok = np.isfinite(lat) & np.isfinite(lon) & (sclass == 0)

    x, y = proj(lon[ok], lat[ok])

    ds.close()
    raw.close()

    return {
        "x": np.asarray(x), "y": np.asarray(y),
        "rain_flag": rain_flag[ok],
        "rain_rate": rain_rate[ok],
        "b_rain_likely": b_rain[ok],
        "b_nrcs_wstd": b_wstd[ok],
        "b_nrcs_delta": b_delt[ok],
        "path": path,
    }


def attach_lr(df, lr):
    """Aggregate LR pixels falling inside each box."""
    cols = {k: np.full(len(df), np.nan) for k in
            ["lr_n", "lr_rain_frac", "lr_rain_rate_max",
             "lr_rain_likely_frac", "lr_nrcs_wstd_frac", "lr_nrcs_delta_frac"]}

    for i, r in df.reset_index(drop=True).iterrows():
        sel = ((lr["x"] >= r["x_min"]) & (lr["x"] <= r["x_min"] + BOX_SIZE) &
               (lr["y"] >= r["y_min"]) & (lr["y"] <= r["y_min"] + BOX_SIZE))
        n = int(sel.sum())
        cols["lr_n"][i] = n
        if n == 0:
            continue
        cols["lr_rain_frac"][i] = np.nanmean(lr["rain_flag"][sel] >= 1)
        cols["lr_rain_rate_max"][i] = np.nanmax(lr["rain_rate"][sel])
        cols["lr_rain_likely_frac"][i] = np.mean(lr["b_rain_likely"][sel])
        cols["lr_nrcs_wstd_frac"][i] = np.mean(lr["b_nrcs_wstd"][sel])
        cols["lr_nrcs_delta_frac"][i] = np.mean(lr["b_nrcs_delta"][sel])

    for k, v in cols.items():
        df[k] = v

    df["lr_label_model"] = df["lr_rain_frac"] > 0.0
    df["lr_label_instr"] = ((df["lr_rain_likely_frac"] > 0.5) |
                            (df["lr_nrcs_wstd_frac"] > 0.5))
    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate(df):
    preds = ["hr_score"] + list(SCORE_TERMS) + ["hr_sig0_fill",
                                               "hr_sig0_anom_db"]

    print("\n" + "=" * 66)
    print("HR-INTERNAL SCREEN vs LR FLAGS")
    print("=" * 66)
    print(f"boxes with LR coverage : {int((df['lr_n'] > 0).sum())} / {len(df)}")

    for label, title in [("lr_label_instr", "LR INSTRUMENT label (primary)"),
                         ("lr_label_model", "LR MODEL label (ECMWF, coarse)")]:

        y = df[label].values.astype(bool) & (df["lr_n"].values > 0)
        n_pos = int(y.sum())
        n_neg = int(((~df[label].values.astype(bool)) &
                     (df["lr_n"].values > 0)).sum())

        print(f"\n--- {title} ---")
        print(f"positives {n_pos}, negatives {n_neg}, "
              f"base rate {n_pos / max(n_pos + n_neg, 1):.3f}")

        if n_pos < 20:
            print("  UNDERPOWERED: fewer than 20 positive boxes. "
                  "Pool several scenes before deciding.")

        print(f"  {'predictor':<22} {'AUC':>6} {'thr':>8} {'POD':>6} {'FPR':>6}")
        for p in preds:
            if p not in df:
                continue
            s = df[p].values.astype(float)
            s = np.where(df["lr_n"].values > 0, s, np.nan)
            auc, _, _ = auc_score(s, y)
            thr, pod, fpr = best_youden(s, y)
            if np.isfinite(auc):
                # a predictor with AUC < 0.5 is informative with sign flipped
                print(f"  {p:<22} {auc:6.3f} {thr:8.3f} {pod:6.3f} {fpr:6.3f}")

    y = df["lr_label_instr"].values.astype(bool) & (df["lr_n"].values > 0)
    auc, n1, n0 = auc_score(df["hr_score"].values, y)

    print("\n" + "-" * 66)
    if not np.isfinite(auc) or n1 < 20:
        verdict = ("INCONCLUSIVE: not enough flagged boxes in this scene. "
                   "Pool all four events, then re-run.")
    elif auc >= 0.80:
        verdict = ("KEEP THE HR SCREEN. It reproduces the LR instrument flags "
                   "at higher resolution; LR becomes optional.")
    elif auc >= 0.65:
        verdict = ("PARTIAL SKILL. Use the union of the HR score and the LR "
                   "flag; neither alone is sufficient.")
    else:
        verdict = ("FALL BACK TO THE LR FLAG. The HR internals carry no useful "
                   "rain signal in this scene.")
    print(f"hr_score AUC vs LR instrument label = {auc:.3f}")
    print(verdict)
    print("-" * 66 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    proj = Proj(proj="utm", zone=UTM_ZONE, ellps="WGS84")

    print("loading HR scene ...")
    hr = load_hr(HR_FILES, HR_GROUP, proj)
    print(f"  {hr['x'].size} pixels after cross-track filter")

    print("computing per-box predictors ...")
    df = hr_box_predictors(hr)
    print(f"  {len(df)} boxes")

    if df.empty:
        print("no boxes; check HR_FILES and MIN_RAW_PIXELS_BOX")
        return

    if LR_FILE:
        print("loading LR granule ...")
        lr = load_lr_labels(LR_FILE, proj)
        print(f"  {lr['path']}")
        print(f"  {lr['x'].size} ocean pixels")
        df = attach_lr(df, lr)
        evaluate(df)
    else:
        print("LR_FILE is None; skipping validation")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"written: {OUT_CSV}")
    print("merge onto your FFT results on ['grid_ix', 'grid_iy']")


if __name__ == "__main__":
    main()
