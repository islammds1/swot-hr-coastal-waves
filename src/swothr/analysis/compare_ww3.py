#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT vs Radon vs MARC/WW3 on a matched 5 km box set, all storms.

Same construction as the two-way FFT/Radon comparison, with WW3 added as a
third leg. Because both estimators now emit a row for every box that passes the
coverage gates, the three fields live on one identical box grid and every
statistic below is a paired per-box difference with no selection step.

Pairs, written as (y minus x):
    FFT   - WW3      observation against model
    Radon - WW3      observation against model

The estimator-against-estimator leg (Radon - FFT) lives in its own script and
is off by default here; set INCLUDE_RADON_FFT = True to add it back.

What the model side is
----------------------
WW3 peak wavelength is NOT taken from a mean-wavelength field. It is obtained
from fp through the dispersion relation, using the same depth and current the
SWOT pipeline used. WW3 integrates its spectrum on the INTRINSIC frequency
grid, so by default fp is intrinsic and

    k_p :  2 pi fp = sqrt(g k tanh(k h))                 (WW3_FP_IS_ABSOLUTE=False)
    k_p :  2 pi fp = sqrt(g k tanh(k h)) + k U_along     (WW3_FP_IS_ABSOLUTE=True)

The model PERIOD is Tp = 1/fp taken directly from the model, with NO Doppler
correction and no recomputation. Each SWOT period is compared with it:

    T_int (SWOT, no current)          vs   Tp_WW3 = 1/fp
    T_abs (SWOT, with k.U Doppler)    vs   Tp_WW3 = 1/fp

for FFT and for Radon. WW3 fp is computed on the intrinsic frequency grid, so
T_int vs Tp_WW3 is the like-for-like pair; T_abs vs Tp_WW3 shows how much the
Doppler term moves SWOT away from (or towards) the model.

Direction
---------
theta_obs_degN is written for every box. lobe_resolved is True only where
SWOT's own velocity-bunching vote chose the lobe; lobe_source == "model" means
the 180 deg choice was taken from this same model. By default
(DIR_POPULATION = "all_axial") every grade A/B box enters the direction
comparison as an AXIS: the difference is folded into [-90, 90], which does not
depend on the lobe and so is a fair test for resolved, model-chosen and
unresolved boxes alike. Full 360 deg statistics are also tabulated, on SWOT-
resolved lobes (dir_*, independent) and on all boxes (all_*, not independent).

Outputs
-------
    fft_radon_ww3_stats.csv      agreement table, per storm and per pair
    fft_radon_ww3_perbox.csv     paired per-box values and differences
    fft_radon_ww3_scatter.png    scatter grid: lambda, T_int vs Tp, T_abs vs Tp, theta
    fft_radon_ww3_hist.png       difference histograms, both estimators vs WW3
    fft_radon_ww3_sens_*.csv     response-threshold sensitivity tables (v4.3)
    fft_radon_ww3_sens_response.png  error against peak filter response
    fft_radon_ww3_doppler.csv    SWOT Doppler size; T_int vs T_abs closeness to Tp
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
import xarray as xr
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================
from swothr.config import path, apply_overrides

ROOT = path("root")             # config/paths.yaml
DIR_FFT   = os.path.join(ROOT, "Results", "FFT", "new")
DIR_RADON = os.path.join(ROOT, "Results", "radon", "new")
MODEL_DIR = os.path.join(ROOT, "Data", "Wave Validation")
OUT_DIR   = os.path.join(ROOT, "Outputs", "comparison_ww3", "new")

TEMPLATE = "swot_5km_{est}_{stem}_{est}_v4.3.csv"

# One entry per granule. The model is matched to the SWOT ACQUISITION time:
#   1. the CSV column acq_time_utc (written by the v4.3 pipelines) if present;
#   2. otherwise "time" below, which must then be the SWOT overpass time
#      (start time in the PIXC file name), NOT the storm peak.
# A model field more than MODEL_MAX_DT_H from that time is not used.
# Set model=None to compare the estimators only for that granule.
GRANULES = [
    dict(storm="Ciaran", stem="Ciaran",
         model="MARC_WW3-NORGAS-2M_ciaran.nc",
         time="2023-11-04T01:00:00"),
    dict(storm="Debi", stem="Debi",
         model="MARC_WW3-NORGAS-2M_Debi.nc",
         time="2023-11-14T23:00:00"),
    dict(storm="Mathis", stem="Mathis",
         model="MARC_WW3-NORGAS-2M_Mathis.nc",
         time="2023-04-05T10:00:00"),
    dict(storm="Nelson", stem="Nelson_42",
         model="MARC_WW3-NORGAS-2M_Nelson_42.nc",
         time="2024-03-29T02:00:00"),
    dict(storm="Nelson", stem="Nelson_70",
         model="MARC_WW3-NORGAS-2M_Nelson_70.nc",
         time="2024-03-30T02:00:00"),
]
MISSING_OK = True          # skip granules whose files are absent
MODEL_MAX_DT_H = 1.0       # model hour farther than this from SWOT -> no model

BOX_CRS = "EPSG:32630"
BOX_SIZE_M = 5000.0
G = 9.81

KEY = ["Date", "Cycle", "Pass", "grid_ix", "grid_iy"]
DIR_COL    = "theta_obs_degN"
LAMBDA_COL = "lambda_obs_m"
TI_COL = "T_intrinsic_from_lambda_obs_s"          # comparable with intrinsic fp
TA_COL = "T_absolute_from_lambda_obs_current_s"    # needs a resolved direction
PERIOD_COL = TI_COL                                 # legacy name used below
SWOT_DIR_CONVENTION = "from"                        # DIR_CONVENTION of the runs
DEPTH_COL, UE_COL, UN_COL = "Depth_m", "U_e_mps", "U_n_mps"

# Quality definition, identical to the map and two-way scripts.
#   "usable" | "R_window" | "grade"
QUALITY_MODE = "grade"
R_MIN        = 0.10
GRADES_GOOD  = ("A", "B")
REPORT_MODES = ("usable", "grade")

# "common": statistics on boxes good for BOTH estimators, so the three legs
#           describe the same population (recommended for the paper).
# "own"   : each estimator judged on its own good boxes.
SUBSET_MODE = "common"

# The Radon-FFT leg is a separate study (swot_fft_vs_radon.py), so it is off
# here. Set True to add it back as a third pair in the table and the figures.
INCLUDE_RADON_FFT = False

SHOW_REJECTED = False      # figures: draw the rejected boxes in grey as well
FLIP_DEG = 150.0

# ---- WW3 variables ----------------------------------------------------------
WW3_DIR_CANDIDATES = ["dp", "VPED", "dir", "th1p", "mean_direction"]
WW3_FP_CANDIDATES  = ["fp", "peak_frequency"]
WW3_LM_CANDIDATES  = ["lp", "wlp", "lm", "lambda", "wavelength"]
# Where the model wavelength comes from:
#   "fp"       : peak wavelength from fp through the dispersion relation
#                (same k as the spectral peak SWOT retrieves)            [default]
#   "variable" : the model's own wavelength variable (first found in
#                WW3_LM_CANDIDATES), used directly, no dispersion inversion.
#                NOTE: WW3 'lm' is the MEAN wavelength, not the peak one;
#                it is systematically shorter than the peak wavelength.
WW3_LAMBDA_SOURCE = "fp"
# WW3 computes fp from the spectrum on its intrinsic (relative) frequency grid,
# so it is intrinsic even when the run is current-forced. Confirm for MARC.
WW3_FP_IS_ABSOLUTE = False
WW3_UNSTRUCT_NEAREST_M = 5000.0   # boxes with no node inside: nearest node within

# Offset added to the WW3 direction to bring it into the SWOT convention.
# "metadata": from the variable's standard_name/long_name (the pipelines use
#             the same rule for the model-assisted lobe choice);
# "auto"    : tests 0 and 180 on SWOT-resolved boxes (printed in every mode as
#             a check);  "fixed": DIR_OFFSET_FIXED.
DIR_OFFSET_MODE  = "fixed"      # confirmed: MARC dp is "from", SWOT is "from"
DIR_OFFSET_FIXED = 0.0

# ---- v4.3 sensitivity ----------------------------------------------------------
RESP_EDGES = [0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
RESPONSE_WARN_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70)
SENS_MASKS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
N_BOOT_CI = 500
# Direction statistics that include model-chosen lobes are NOT independent of
# the model; they are written only as a labelled supplementary row.
REPORT_MODEL_LOBE_DIRECTION = True

# ---- which boxes enter the DIRECTION comparison (figures, histograms, sens) --
#   "all_axial"     : every good (grade A/B) box, compared as an AXIS
#                     (error folded into [-90, 90]); independent of the 180 deg
#                     choice, so unresolved and model-chosen lobes are valid.
#   "swot_resolved" : full 360 deg direction, SWOT velocity-bunching lobes only.
#   "all"           : full direction, every good box. NOT independent: model-
#                     chosen lobes agree with the model by construction and
#                     unresolved lobes are arbitrary. Diagnostic only.
# The stats table always reports all three (ax_*, dir_*, all_*).
DIR_POPULATION = "all_axial"

# ---- scatter axes --------------------------------------------------------------
# Each quantity gets the SAME axis range in every row (FFT, Radon), taken from
# the AXIS_PCT percentiles of all plotted points of both rows. Points outside
# are counted in the panel ("k outside"); statistics always use every point.
# Pin a range explicitly with e.g. AXIS_LIMITS = {"theta": (150, 330)}.
AXIS_PCT = (0.5, 99.5)

# Model period that SWOT T_int and SWOT T_abs are compared with:
#   "absolute" : WW3 T_abs for BOTH  (lambda_p from fp, + k.U with the model
#                direction and the same current as that estimator)     [default]
#   "intrinsic": WW3 T_int for BOTH  (lambda_p from fp, no current)
#   "matched"  : WW3 T_int vs SWOT T_int, and WW3 T_abs vs SWOT T_abs
WW3_PERIOD_REFERENCE = "absolute"


apply_overrides(globals(), "compare", final=True)


def model_T_col(side, quantity):
    """WW3 period column compared with SWOT quantity 'Ti' or 'Ta' of one side."""
    ref = WW3_PERIOD_REFERENCE
    if ref == "matched":
        ref = "intrinsic" if quantity == "Ti" else "absolute"
    return "ww3_Tp_s" if ref == "intrinsic" else "ww3_T_absolute_s_" + side


def model_T_label(quantity):
    ref = WW3_PERIOD_REFERENCE
    if ref == "matched":
        ref = "intrinsic" if quantity == "Ti" else "absolute"
    return (r"WW3  $T_{int}$ from $\lambda_p$ (no Doppler)  (s)" if ref == "intrinsic"
            else r"WW3  $T_p$ from $\lambda_p$")

# Separate scatter/histogram figures per group, in addition to the all-storm one.
#   "storm": Ciaran, Debi, Mathis, Nelson (both Nelson passes together)
#   "stem" : one per granule (Nelson_42 and Nelson_70 separately)
#   None   : all-storm figures only
PER_GROUP_FIGURES = "storm"
AXIS_LIMITS = {}

DPI = 300


# =============================================================================
# SMALL HELPERS
# =============================================================================
def wrap180(d):
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def wrap90(d):
    """Axial difference: fold an angle difference into [-90, 90)."""
    return (np.asarray(d, dtype=float) + 90.0) % 180.0 - 90.0


def dtheta_col(tag, population=None):
    pop = DIR_POPULATION if population is None else population
    return ("dtheta_ax_" if pop == "all_axial" else "dtheta_") + tag


def dir_label(population=None):
    pop = DIR_POPULATION if population is None else population
    return {"all_axial": "all A/B boxes, axial (mod 180)",
            "swot_resolved": "SWOT-resolved lobes",
            "all": "all A/B boxes, full dir. (not independent)"}[pop]


def circ_mean_deg(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    a = np.deg2rad(v)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360.0)


def circ_stats(d):
    """Statistics of an already-differenced angle series (deg)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0, bias=np.nan, sd=np.nan, rmse=np.nan, mad=np.nan,
                    p10=np.nan, p20=np.nan, p30=np.nan, n_flip=0)
    a = np.deg2rad(d)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    R = np.hypot(C, S)
    core = d[np.abs(d) <= FLIP_DEG]
    return dict(n=int(d.size),
                bias=float(np.rad2deg(np.arctan2(S, C))),
                sd=float(np.rad2deg(np.sqrt(max(-2.0*np.log(max(R, 1e-12)), 0.0)))),
                rmse=float(np.sqrt(np.mean(d**2))),
                mad=float(np.median(np.abs(core - np.median(core)))) if core.size else np.nan,
                p10=float(100*np.mean(np.abs(d) <= 10)),
                p20=float(100*np.mean(np.abs(d) <= 20)),
                p30=float(100*np.mean(np.abs(d) <= 30)),
                n_flip=int(np.sum(np.abs(d) > FLIP_DEG)))


def lin_stats(x, y):
    """Agreement of y against reference x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return dict(n=int(x.size), bias=np.nan, med=np.nan, rmsd=np.nan,
                    si=np.nan, r=np.nan, slope=np.nan, intercept=np.nan)
    d = y - x
    if np.std(x) == 0 or np.std(y) == 0:
        slope = intercept = np.nan
        rmsd = float(np.sqrt(np.mean(d**2)))
        return dict(n=int(x.size), bias=float(d.mean()), med=float(np.median(d)),
                    rmsd=rmsd, si=float(100*rmsd/np.mean(x)), r=np.nan,
                    slope=slope, intercept=intercept)
    slope, intercept = np.polyfit(x, y, 1)
    rmsd = float(np.sqrt(np.mean(d**2)))
    return dict(n=int(x.size), bias=float(d.mean()), med=float(np.median(d)),
                rmsd=rmsd, si=float(100*rmsd/np.mean(x)),
                r=float(np.corrcoef(x, y)[0, 1]),
                slope=float(slope), intercept=float(intercept))


def good_mask(m, side, mode=None):
    """'good' mask for one estimator side ('F' or 'R')."""
    mode = QUALITY_MODE if mode is None else mode
    if mode == "usable":
        col = "usable_" + side
        g = m[col].fillna(False).astype(bool).to_numpy() if col in m else np.zeros(len(m), bool)
    elif mode == "grade":
        col = "instrument_observability_" + side
        g = m[col].isin(GRADES_GOOD).to_numpy() if col in m else np.zeros(len(m), bool)
    elif mode == "R_window":
        col = "R_window_" + side
        g = (m[col].to_numpy(float) > R_MIN) if col in m else np.zeros(len(m), bool)
    else:
        raise ValueError(f"unknown quality mode {mode!r}")
    return g & m["has_both"].to_numpy()


# =============================================================================
# WW3
# =============================================================================
def find_name(ds, candidates, label, required=True):
    available = set(ds.variables) | set(ds.coords)
    for name in candidates:
        if name in available:
            return name
    if not required:
        return None
    raise KeyError(f"could not identify {label}; tried {candidates}")


def select_time(ds, target):
    """Nearest model time to the SWOT time; returns (dataset, offset_hours)."""
    for name in ("time", "time_counter", "valid_time", "forecast_time"):
        if name in ds.coords or name in ds.variables:
            try:
                tv = pd.to_datetime(np.asarray(ds[name].values).ravel())
                sel = ds.sel({name: np.datetime64(pd.Timestamp(target))}, method="nearest")
                actual = pd.to_datetime(np.asarray(sel[name].values).item())
                off_h = abs(pd.Timestamp(actual) - pd.Timestamp(target)) / pd.Timedelta("1h")
                print(f"  [ww3] SWOT {pd.Timestamp(target)} -> model {actual} "
                      f"(offset {off_h:.2f} h; file covers {tv.min()} .. {tv.max()})")
                return sel, float(off_h)
            except (KeyError, TypeError, ValueError):
                continue
    print("  [ww3] no time axis found; field used as is")
    return ds, 0.0


def as_2d(da, shape, name):
    arr = np.asarray(da.squeeze(drop=True).values, dtype=float)
    if arr.shape == shape:
        return arr
    if arr.ndim == 2 and arr.T.shape == shape:
        return arr.T
    raise ValueError(f"'{name}' has shape {arr.shape}, grid is {shape}")


def ww3_convention(da):
    """'from' | 'towards' | None from the direction variable's metadata."""
    txt = " ".join(str(da.attrs.get(a, "")) for a in
                   ("standard_name", "long_name", "convention", "comment",
                    "direction_reference", "description")).lower()
    if "to_direction" in txt or "going to" in txt or "toward" in txt:
        return "towards"
    if ("from_direction" in txt or "coming from" in txt or "nautical" in txt
            or "direction from" in txt):
        return "from"
    return None


def collocate_ww3(boxes, model_path, model_time):
    """Box-mean WW3 fp, direction and lm on the SWOT box grid.

    Handles regular lon/lat grids and unstructured meshes (lon/lat on one node
    dimension). Boxes without a model node inside take the nearest node within
    WW3_UNSTRUCT_NEAREST_M of the box centre.
    """
    ds = xr.open_dataset(model_path)
    ds, off_h = select_time(ds, model_time)
    if off_h > MODEL_MAX_DT_H:
        ds.close()
        print(f"  [ww3] SKIPPED: nearest model time is {off_h:.2f} h from SWOT "
              f"(> MODEL_MAX_DT_H = {MODEL_MAX_DT_H} h). The model file does not "
              "cover the overpass; get the matching hour.")
        return None

    lon_name = find_name(ds, ["longitude", "lon", "nav_lon", "x"], "longitude")
    lat_name = find_name(ds, ["latitude", "lat", "nav_lat", "y"], "latitude")
    dir_name = find_name(ds, WW3_DIR_CANDIDATES, "WW3 direction")
    fp_name  = find_name(ds, WW3_FP_CANDIDATES, "WW3 peak frequency")
    lm_name  = find_name(ds, WW3_LM_CANDIDATES, "WW3 wavelength", required=False)
    conv = ww3_convention(ds[dir_name])

    lon_da = ds[lon_name].squeeze(drop=True); lat_da = ds[lat_name].squeeze(drop=True)
    unstructured = (lon_da.ndim == 1 and lat_da.ndim == 1 and lon_da.dims == lat_da.dims)
    if unstructured:
        space = lon_da.dims
        lon2d = np.asarray(lon_da.values, float); lat2d = np.asarray(lat_da.values, float)
    elif lon_da.ndim == 1:
        space = (lat_da.dims[0], lon_da.dims[0])
        lon2d, lat2d = np.meshgrid(np.asarray(lon_da.values, float), np.asarray(lat_da.values, float))
    else:
        space = lon_da.dims
        lon2d = np.asarray(lon_da.values, float); lat2d = np.asarray(lat_da.values, float)

    def field(name, positive=False):
        da = ds[name].squeeze(drop=True)
        if positive:
            da = da.where(da > 0)
        if set(space).issubset(da.dims):
            da = da.transpose(*space)
        arr = np.asarray(da.values, float)
        if arr.size != lon2d.size:
            raise ValueError(f"'{name}' has shape {arr.shape}, grid is {lon2d.shape}")
        return arr.ravel()

    f_fp = field(fp_name, positive=True)
    f_dp = field(dir_name) % 360.0
    f_lm = field(lm_name, positive=True) if lm_name else np.full(lon2d.size, np.nan)
    print(f"  [ww3] {'unstructured' if unstructured else 'regular'} grid; "
          f"dir={dir_name} std_name={ds[dir_name].attrs.get('standard_name','')!r} "
          f"long_name={ds[dir_name].attrs.get('long_name','')!r} -> convention {conv}"
          f"  fp={fp_name}" + (f"  lm={lm_name}" if lm_name else ""))
    if lm_name:
        print(f"  [ww3] wavelength variable '{lm_name}': "
              f"long_name={ds[lm_name].attrs.get('long_name', '')!r} "
              f"std_name={ds[lm_name].attrs.get('standard_name', '')!r} "
              f"units={ds[lm_name].attrs.get('units', '')!r}")

    tr = Transformer.from_crs("EPSG:4326", BOX_CRS, always_xy=True)
    mx, my = tr.transform(lon2d.ravel(), lat2d.ravel())
    mx, my = np.asarray(mx), np.asarray(my)
    ds.close()

    ox = np.median(boxes["x_min"].to_numpy() - boxes["grid_ix"].to_numpy()*BOX_SIZE_M)
    oy = np.median(boxes["y_min"].to_numpy() - boxes["grid_iy"].to_numpy()*BOX_SIZE_M)
    res_x = np.abs(boxes["x_min"].to_numpy() - boxes["grid_ix"].to_numpy()*BOX_SIZE_M - ox)
    res_y = np.abs(boxes["y_min"].to_numpy() - boxes["grid_iy"].to_numpy()*BOX_SIZE_M - oy)
    if np.nanmax(res_x) > 1.0 or np.nanmax(res_y) > 1.0:
        raise RuntimeError("box grid is not regular; cannot bin the model field")

    good = np.isfinite(mx) & np.isfinite(my) & np.isfinite(f_fp) & np.isfinite(f_dp)
    keep = (good & (mx >= boxes["x_min"].min()) & (mx <= boxes["x_max"].max()) &
            (my >= boxes["y_min"].min()) & (my <= boxes["y_max"].max()))
    pts = pd.DataFrame({
        "grid_ix": np.floor((mx[keep] - ox)/BOX_SIZE_M).astype(int),
        "grid_iy": np.floor((my[keep] - oy)/BOX_SIZE_M).astype(int),
        "fp": f_fp[keep], "lm": f_lm[keep],
        "s": np.sin(np.deg2rad(f_dp[keep])), "c": np.cos(np.deg2rad(f_dp[keep]))})
    agg = pts.groupby(["grid_ix", "grid_iy"], as_index=False).agg(
        ww3_fp_hz=("fp", "mean"), ww3_lm_m=("lm", "mean"),
        _s=("s", "mean"), _c=("c", "mean"), n_model_points=("fp", "size"))

    # boxes with no node inside (coarse or unstructured mesh): nearest node
    have = set(zip(agg["grid_ix"], agg["grid_iy"]))
    need = boxes[["grid_ix", "grid_iy", "x_min", "x_max", "y_min", "y_max"]].drop_duplicates(["grid_ix", "grid_iy"])
    need = need[[(a, b) not in have for a, b in zip(need["grid_ix"], need["grid_iy"])]]
    if len(need) and good.any():
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[mx[good], my[good]])
        xc = 0.5*(need["x_min"] + need["x_max"]).to_numpy(); yc = 0.5*(need["y_min"] + need["y_max"]).to_numpy()
        dist, j = tree.query(np.c_[xc, yc])
        ok = dist <= WW3_UNSTRUCT_NEAREST_M
        if ok.any():
            gi = np.flatnonzero(good)[j[ok]]
            extra = pd.DataFrame({"grid_ix": need["grid_ix"].to_numpy()[ok],
                                  "grid_iy": need["grid_iy"].to_numpy()[ok],
                                  "ww3_fp_hz": f_fp[gi], "ww3_lm_m": f_lm[gi],
                                  "_s": np.sin(np.deg2rad(f_dp[gi])), "_c": np.cos(np.deg2rad(f_dp[gi])),
                                  "n_model_points": 0})
            agg = pd.concat([agg, extra], ignore_index=True)
            print(f"  [ww3] {int(ok.sum())} box(es) filled from the nearest node")
    agg["ww3_dir_raw_degN"] = np.rad2deg(np.arctan2(agg["_s"], agg["_c"])) % 360.0
    agg["ww3_Tp_model_s"] = 1.0/agg["ww3_fp_hz"]
    agg["ww3_dir_convention_meta"] = conv if conv else ""
    return agg.drop(columns=["_s", "_c"])


def k_from_frequency(f_abs, depth, u_along, absolute=True):
    """Invert omega = sqrt(g k tanh(k h)) + k u for k, by bisection."""
    f_abs = np.asarray(f_abs, float); depth = np.asarray(depth, float)
    u = np.asarray(u_along, float) if absolute else np.zeros_like(f_abs)
    out = np.full(f_abs.shape, np.nan)
    ok = np.isfinite(f_abs) & (f_abs > 0) & np.isfinite(depth) & (depth > 0)
    if not ok.any():
        return out
    omega = 2.0*np.pi*f_abs[ok]
    h = depth[ok]
    uu = np.where(np.isfinite(u[ok]), u[ok], 0.0)
    lo = np.full(omega.shape, 1e-5); hi = np.full(omega.shape, 1.0)
    for _ in range(80):
        mid = 0.5*(lo + hi)
        val = np.sqrt(G*mid*np.tanh(mid*h)) + mid*uu
        small = val < omega
        lo = np.where(small, mid, lo)
        hi = np.where(small, hi, mid)
    out[ok] = 0.5*(lo + hi)
    return out


def to_towards(theta_deg):
    """SWOT-convention bearing -> propagation (towards) bearing."""
    t = np.asarray(theta_deg, float)
    return (t + 180.0) % 360.0 if SWOT_DIR_CONVENTION == "from" else t % 360.0


# =============================================================================
# BUILD THE MATCHED TABLE
# =============================================================================
def _no_model(m):
    for c in ("ww3_fp_hz", "ww3_dir_raw_degN", "ww3_lm_m",
              "ww3_Tp_model_s", "n_model_points"):
        m[c] = np.nan
    m["ww3_dir_convention_meta"] = ""
    return m


def load_granule(g):
    paths = {s: os.path.join(d, TEMPLATE.format(est=e, stem=g["stem"]))
             for s, d, e in (("F", DIR_FFT, "FFT"), ("R", DIR_RADON, "RADON"))}
    for s, p in paths.items():
        if not os.path.exists(p):
            if not MISSING_OK:
                raise FileNotFoundError(p)
            print(f"[data] {g['stem']}: missing {os.path.basename(p)} (skipped)")
            return None
    fft = pd.read_csv(paths["F"]); rad = pd.read_csv(paths["R"])
    kf = set(map(tuple, fft[KEY[:3]].astype(str).drop_duplicates().to_numpy()))
    kr = set(map(tuple, rad[KEY[:3]].astype(str).drop_duplicates().to_numpy()))
    if kf != kr:
        print(f"[data] {g['stem']}: WARNING Date/Cycle/Pass differ between files:\n"
              f"         FFT   {sorted(kf)}\n         Radon {sorted(kr)}\n"
              "         -> the two CSVs are not the same granule/config; fix the "
              "pipeline DATE/CYCLE/PASS and rerun one of them")

    m = fft.merge(rad, on=KEY, suffixes=("_F", "_R"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        n = int((m["_merge"] != "both").sum())
        print(f"[data] {g['stem']}: WARNING {n} box(es) on one side only")
    m = m[m["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    if m.empty:
        print(f"[data] {g['stem']}: no common box -> granule skipped")
        return None
    m["storm"] = g["storm"]; m["stem"] = g["stem"]

    for s in ("F", "R"):
        for c in ("usable_", "lobe_resolved_", "direction_resolved_",
                  "spectral_internal_quality_"):
            if c + s in m:
                m[c + s] = m[c + s].fillna(False).astype(bool)
    m["has_both"] = (np.isfinite(m[LAMBDA_COL + "_F"]) &
                     np.isfinite(m[LAMBDA_COL + "_R"]))

    # v4.2 CSVs have no lobe_source: every SWOT-resolved lobe was VB.
    for s in ("F", "R"):
        if "lobe_source_" + s not in m:
            m["lobe_source_" + s] = np.where(m["lobe_resolved_" + s], "vb", "unresolved")
        if "direction_resolved_" + s not in m:
            m["direction_resolved_" + s] = m["lobe_resolved_" + s]
    # The effective (Kirby-Chen) current depends on each estimator's own k, so
    # it differs slightly between the two CSVs. Each side keeps its own current;
    # the model Doppler is computed separately with each side's value.
    du = np.hypot(m[UE_COL + "_F"] - m[UE_COL + "_R"], m[UN_COL + "_F"] - m[UN_COL + "_R"])
    if np.nanmax(du.to_numpy(float), initial=0.0) > 0.10:
        print(f"[data] {g['stem']}: WARNING FFT/Radon currents differ by up to "
              f"{np.nanmax(du):.2f} m/s (median {np.nanmedian(du):.3f}); check that "
              "both runs used the same current file")

    # SWOT acquisition time for the model match
    t_swot = None
    for s in ("F", "R"):
        c = "acq_time_utc_" + s
        if c in m:
            tt = pd.to_datetime(m[c].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna(),
                                errors="coerce").dropna()
            if len(tt):
                t_swot = tt.min() + (tt.max() - tt.min()) / 2
                break
    if t_swot is None:
        t_swot = pd.Timestamp(g["time"])
        print(f"[data] {g['stem']}: no acq_time_utc in the CSVs -> using GRANULES time "
              f"{t_swot} (must be the SWOT overpass time)")
    else:
        print(f"[data] {g['stem']}: SWOT acquisition time {t_swot} (from CSV)")

    # geometry and forcing are identical on both sides; keep one copy
    for c in (DEPTH_COL, "cross_track_m", "lon_center",
              "lat_center", "x_min", "x_max", "y_min", "y_max"):
        if c + "_F" in m:
            m[c] = m[c + "_F"]

    if not g.get("model"):
        print(f"[data] {g['stem']}: no model file configured, estimators only")
        return _no_model(m)

    mp = g["model"] if os.path.isabs(g["model"]) else os.path.join(MODEL_DIR, g["model"])
    if not os.path.exists(mp):
        if not MISSING_OK:
            raise FileNotFoundError(mp)
        print(f"[data] {g['stem']}: missing model {os.path.basename(mp)} (estimators only)")
        return _no_model(m)

    agg = collocate_ww3(m, mp, t_swot)
    if agg is None:
        return _no_model(m)
    m = m.merge(agg, on=["grid_ix", "grid_iy"], how="left")
    print(f"  [ww3] {int(m['ww3_fp_hz'].notna().sum())}/{len(m)} boxes collocated")
    return m


def resolve_direction_offset(m):
    """Offset bringing WW3 into the SWOT convention; always prints the 0/180 test."""
    sel = (good_mask(m, "F") & good_mask(m, "R")
           & m["lobe_resolved_F"].to_numpy() & m["lobe_resolved_R"].to_numpy()
           & np.isfinite(m["ww3_dir_raw_degN"].to_numpy(float)))
    out = {}
    if sel.sum() >= 10:
        for off in (0.0, 180.0):
            d = wrap180(0.5*(m.loc[sel, DIR_COL + "_F"] + m.loc[sel, DIR_COL + "_R"])
                        - (m.loc[sel, "ww3_dir_raw_degN"] + off))
            s = circ_stats(d)
            out[off] = s
            print(f"[dir ] offset {off:5.0f}: bias {s['bias']:+7.1f}  RMSE {s['rmse']:6.1f}"
                  f"  within 30 deg {s['p30']:5.1f}%  (n={s['n']}, SWOT-resolved lobes)")
    else:
        print(f"[dir ] only {int(sel.sum())} SWOT-resolved boxes; 0/180 test skipped")

    meta = set(m.get("ww3_dir_convention_meta", pd.Series(dtype=str)).dropna()) - {""}
    meta_off = None
    if len(meta) == 1:
        conv = meta.pop()
        meta_off = 0.0 if conv == SWOT_DIR_CONVENTION else 180.0
        print(f"[dir ] metadata says WW3 direction is '{conv}' -> offset {meta_off:.0f}")
    elif len(meta) > 1:
        print(f"[dir ] WARNING inconsistent WW3 conventions across files: {meta}")

    if out:
        best = min(out, key=lambda k: out[k]["rmse"] if np.isfinite(out[k]["rmse"]) else np.inf)
        worst = max(out[0.0]["rmse"], out[180.0]["rmse"])
        bestv = max(min(out[0.0]["rmse"], out[180.0]["rmse"]), 1e-9)
        if worst/bestv < 1.3:
            print("[dir ] WARNING 0 and 180 fit almost equally well on SWOT-resolved boxes")
        if meta_off is not None and best != meta_off:
            print("[dir ] WARNING the data prefer the OTHER convention than the metadata. "
                  "Either the metadata or the SWOT VB sign is wrong; model-chosen lobes "
                  "in the pipelines inherit the metadata choice.")

    if DIR_OFFSET_MODE == "fixed":
        return float(DIR_OFFSET_FIXED)
    if DIR_OFFSET_MODE == "metadata" and meta_off is not None:
        return float(meta_off)
    if out:
        best = min(out, key=lambda k: out[k]["rmse"] if np.isfinite(out[k]["rmse"]) else np.inf)
        print(f"[dir ] using data-selected offset {best:.0f}")
        return float(best)
    print(f"[dir ] falling back to DIR_OFFSET_FIXED={DIR_OFFSET_FIXED:.0f}")
    return float(DIR_OFFSET_FIXED)


def add_derived(m, dir_offset):
    """
    WW3 direction in the SWOT convention, WW3 wavelength (from fp or the model
    variable), WW3 period Tp = 1/fp (no Doppler), and all differences.
    """
    m["ww3_dir_degN"] = (m["ww3_dir_raw_degN"] + dir_offset) % 360.0
    th_to_m = to_towards(m["ww3_dir_degN"].to_numpy(float))
    h = m[DEPTH_COL].to_numpy(float)

    def u_along_side(s):
        ue = m[UE_COL + "_" + s].to_numpy(float); un = m[UN_COL + "_" + s].to_numpy(float)
        return ue, un, ue*np.sin(np.deg2rad(th_to_m)) + un*np.cos(np.deg2rad(th_to_m))

    # lambda_p from fp (intrinsic: no current needed; absolute: FFT current)
    _, _, ua_F = u_along_side("F")
    k = k_from_frequency(m["ww3_fp_hz"], h, ua_F, absolute=WW3_FP_IS_ABSOLUTE)
    m["ww3_lambda_from_fp_m"] = np.where(np.isfinite(k) & (k > 0), 2.0*np.pi/k, np.nan)
    if "ww3_lm_m" in m and m["ww3_lm_m"].notna().any():
        r = (m["ww3_lm_m"] / m["ww3_lambda_from_fp_m"]).replace([np.inf, -np.inf], np.nan).dropna()
        if r.size:
            print(f"[ww3 ] model wavelength variable / lambda(fp): median {r.median():.3f} "
                  f"(5-95%: {r.quantile(0.05):.3f}-{r.quantile(0.95):.3f}, n={r.size})")
    if WW3_LAMBDA_SOURCE == "variable":
        if "ww3_lm_m" not in m or m["ww3_lm_m"].isna().all():
            raise RuntimeError("WW3_LAMBDA_SOURCE='variable' but no wavelength variable "
                               f"found among {WW3_LM_CANDIDATES}")
        m["ww3_lambda_m"] = m["ww3_lm_m"]
        print("[ww3 ] model wavelength taken directly from the model variable")
    else:
        m["ww3_lambda_m"] = m["ww3_lambda_from_fp_m"]
    m["ww3_k_radm"] = 2.0*np.pi/m["ww3_lambda_m"]

    # Model period: fp -> lambda_p (above) -> T through the SAME dispersion
    # relation as SWOT, with NO Doppler term:  T = 2 pi / sqrt(g k tanh(k h)).
    # Both SWOT periods (intrinsic and absolute) are compared with this value.
    h_all = m[DEPTH_COL].to_numpy(float)
    lam_all = m["ww3_lambda_m"].to_numpy(float)
    ok = np.isfinite(lam_all) & (lam_all > 0) & np.isfinite(h_all) & (h_all > 0)
    kk = np.where(ok, 2.0*np.pi/np.where(ok, lam_all, 1.0), np.nan)
    m["ww3_Tp_s"] = np.where(ok, 2.0*np.pi/np.sqrt(G*kk*np.tanh(kk*np.where(ok, h_all, 1.0))), np.nan)
    chk = (m["ww3_Tp_s"] - m["ww3_Tp_model_s"]).abs().dropna()
    if chk.size:
        print(f"[ww3 ] Tp from lambda_p via dispersion vs 1/fp: max |diff| = "
              f"{chk.max():.2e} s (n={chk.size})")
    for s in ("F", "R"):
        ue, un, ua = u_along_side(s)
        m["ww3_U_along_mps_" + s] = ua
        # Model ABSOLUTE period: same lambda_p and depth, plus the Doppler term
        # k.U with the MODEL propagation direction and this side's current:
        #   T_abs = 2 pi / (sqrt(g k tanh(k h)) + k U_along)
        kk_ = 2.0*np.pi/m["ww3_lambda_m"].to_numpy(float)
        om_ = np.sqrt(G*kk_*np.tanh(kk_*h_all)) + kk_*np.where(np.isfinite(ua), ua, np.nan)
        m["ww3_T_absolute_s_" + s] = np.where(om_ > 0, 2.0*np.pi/om_, np.nan)

    for s in ("F", "R"):
        m[f"dlambda_{s}W"] = m[LAMBDA_COL + "_" + s] - m["ww3_lambda_m"]
        m[f"dlambda_rel_{s}W"] = 100*m[f"dlambda_{s}W"]/m["ww3_lambda_m"]
        m[f"dTi_{s}W"] = m[TI_COL + "_" + s] - m[model_T_col(s, "Ti")]
        m[f"dTa_{s}W"] = m[TA_COL + "_" + s] - m[model_T_col(s, "Ta")]
        m[f"dT_{s}W"] = m[f"dTi_{s}W"]                         # legacy name
        m[f"dtheta_{s}W"] = wrap180(m[DIR_COL + "_" + s] - m["ww3_dir_degN"])
        m[f"dtheta_ax_{s}W"] = wrap90(m[f"dtheta_{s}W"])
        # observed axis put on the lobe nearest the model (for plotting)
        # no %360: keeps the point next to the 1:1 line across the 0/360 seam
        m[DIR_COL + "_ax_" + s] = m["ww3_dir_degN"] + m[f"dtheta_ax_{s}W"]
    if INCLUDE_RADON_FFT:
        m["dlambda_RF"] = m[LAMBDA_COL + "_R"] - m[LAMBDA_COL + "_F"]
        m["dTi_RF"] = m[TI_COL + "_R"] - m[TI_COL + "_F"]
        m["dTa_RF"] = m[TA_COL + "_R"] - m[TA_COL + "_F"]
        m["dT_RF"] = m["dTi_RF"]
        m["dtheta_RF"] = wrap180(m[DIR_COL + "_R"] - m[DIR_COL + "_F"])
        m["dtheta_ax_RF"] = wrap90(m["dtheta_RF"])
    return m.copy()          # defragment after the block of inserts


# =============================================================================
# PAIRS AND STATISTICS
# =============================================================================
# label -> (x column suffix, y column suffix, difference tag)
PAIRS = {
    "FFT-WW3":   ("ww3", "F", "FW"),
    "Radon-WW3": ("ww3", "R", "RW"),
}
if INCLUDE_RADON_FFT:
    PAIRS["Radon-FFT"] = ("F", "R", "RF")

QCOL = {"lambda": (LAMBDA_COL, "ww3_lambda_m", "dlambda"),
        "Ti": (TI_COL, "ww3_Tp_s", "dTi"),
        "Ta": (TA_COL, "ww3_T_absolute_s", "dTa"),   # side suffix added below
        "theta": (DIR_COL, "ww3_dir_degN", "dtheta")}


def pair_columns(pair, quantity):
    """(x_col, y_col) for one pair and one of lambda / Ti / Ta / theta."""
    xs, ys, _ = PAIRS[pair]
    base, ww3, _ = QCOL[quantity]
    if quantity in ("Ti", "Ta"):
        ww3 = model_T_col(ys if ys != "ww3" else xs, quantity)
    if quantity == "theta" and DIR_POPULATION == "all_axial" and "ww3" in (xs, ys):
        base = DIR_COL + "_ax"
    x = ww3 if xs == "ww3" else f"{base}_{xs}"
    y = ww3 if ys == "ww3" else f"{base}_{ys}"
    return x, y


def _flag(m, col):
    return m[col].fillna(False).astype(bool).to_numpy() if col in m else np.zeros(len(m), bool)


def pair_mask(m, pair, mode=None, for_direction=False, quantity=None,
              include_model_lobes=False, dir_population=None):
    """Boxes entering one pair's statistics.

    Direction: SWOT-resolved lobes only (lobe_resolved), unless
    include_model_lobes (supplementary, not an independent validation).
    Ta: direction must be resolved by SWOT or the model (direction_resolved).
    """
    if quantity == "theta":
        for_direction = True
    xs, ys, _ = PAIRS[pair]
    ok = m["has_both"].to_numpy().copy()
    sides = [s for s in (xs, ys) if s != "ww3"]
    if SUBSET_MODE == "common":
        sides = ["F", "R"]
    for s in sides:
        ok &= good_mask(m, s, mode)
    pop = DIR_POPULATION if dir_population is None else dir_population
    if for_direction and pop == "swot_resolved":
        col = "direction_resolved_" if include_model_lobes else "lobe_resolved_"
        for s in sides:
            ok &= _flag(m, col + s)
    if quantity == "Ta":
        for s in sides:
            ok &= _flag(m, "direction_resolved_" + s)
    if "ww3" in (xs, ys):
        ok &= np.isfinite(m["ww3_lambda_m"].to_numpy(float))
        if for_direction:
            ok &= np.isfinite(m["ww3_dir_degN"].to_numpy(float))
    return ok


def stats_table(m, modes=None):
    modes = (QUALITY_MODE,) if modes is None else modes
    groups = [("ALL", m)] + ([(s, d) for s, d in m.groupby("storm")]
                             if m["storm"].nunique() > 1 else [])
    rows = []
    for mode in modes:
        for gname, g in groups:
            for pair, (_, _, tag) in PAIRS.items():
                row = dict(quality_mode=mode, storm=gname, pair=pair)
                for pop, pre in (("all_axial", "ax"), ("swot_resolved", "dir"), ("all", "all")):
                    if pop == "all" and not REPORT_MODEL_LOBE_DIRECTION:
                        continue
                    sel_d = pair_mask(g, pair, mode, quantity="theta", dir_population=pop)
                    c = circ_stats(g.loc[sel_d, dtheta_col(tag, pop)])
                    row.update({f"{pre}_n": c["n"], f"{pre}_bias": c["bias"],
                                f"{pre}_rmse": c["rmse"], f"{pre}_sd": c["sd"],
                                f"{pre}_within10_pct": c["p10"],
                                f"{pre}_within30_pct": c["p30"]})
                    if pre == "dir":
                        row["dir_flips"] = c["n_flip"]
                for q, tag2 in (("lambda", "lam"), ("Ti", "Ti"), ("Ta", "Ta")):
                    xc, yc = pair_columns(pair, q)
                    sel = pair_mask(g, pair, mode, quantity=q)
                    s = lin_stats(g.loc[sel, xc], g.loc[sel, yc])
                    row.update({f"{tag2}_n": s["n"], f"{tag2}_bias": s["bias"],
                                f"{tag2}_med": s["med"], f"{tag2}_rmsd": s["rmsd"],
                                f"{tag2}_si_pct": s["si"], f"{tag2}_r": s["r"],
                                f"{tag2}_slope": s["slope"]})
                rows.append(row)
    return pd.DataFrame(rows)





def stratify(m, col, edges, label):
    """Median differences against WW3 in bins of a covariate, both estimators."""
    sel = pair_mask(m, "FFT-WW3", for_direction=False)
    s = m[sel].copy()
    s["_bin"] = pd.cut(s[col], edges)
    out = s.groupby("_bin", observed=True).agg(
        n=("dlambda_FW", "size"),
        dlam_F=("dlambda_FW", "median"), dlam_R=("dlambda_RW", "median"),
        dTi_F=("dTi_FW", "median"), dTi_R=("dTi_RW", "median"),
        dTa_F=("dTa_FW", "median"), dTa_R=("dTa_RW", "median"))
    out.index.name = label
    return out.round(2)


# =============================================================================
# FIGURES
# =============================================================================
QUANTS = [("lambda", r"$\lambda$  (m)", "lam"),
          ("Ti", r"$T_{int}$  (s)", "Ti"),
          ("Ta", r"$T_{abs}$  (s)", "Ta"),
          ("theta", r"$\theta$  ($^\circ$N)", "dir")]
PAIR_ORDER = list(PAIRS)


def scatter_grid(m, path, group=None):
    """One row per pair, one column per quantity (lambda, T_int, T_abs, theta)."""
    nrow, ncol = len(PAIR_ORDER), len(QUANTS)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.8*ncol, 4.75*nrow + 1.0), squeeze=False)

    # one axis range per quantity, shared by all rows
    lims = {}
    for q, _, _ in QUANTS:
        if q in AXIS_LIMITS:
            lims[q] = tuple(map(float, AXIS_LIMITS[q])); continue
        vals = []
        for pair in PAIR_ORDER:
            xc, yc = pair_columns(pair, q)
            fg = pair_mask(m, pair, quantity=q)
            vals += [m.loc[fg, xc].to_numpy(float), m.loc[fg, yc].to_numpy(float)]
        v = np.concatenate(vals) if vals else np.array([])
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = np.percentile(v, AXIS_PCT)
            pad = 0.04*max(hi - lo, 1e-6)
            lims[q] = (float(lo - pad), float(hi + pad))
        else:
            lims[q] = (0.0, 1.0)
    # T_int and T_abs are compared with the same model Tp: give them one range
    if "Ti" in lims and "Ta" in lims and "Ti" not in AXIS_LIMITS and "Ta" not in AXIS_LIMITS:
        lims["Ti"] = lims["Ta"] = (min(lims["Ti"][0], lims["Ta"][0]),
                                   max(lims["Ti"][1], lims["Ta"][1]))

    for i, pair in enumerate(PAIR_ORDER):
        for j, (q, lab, _) in enumerate(QUANTS):
            a = ax[i, j]
            xc, yc = pair_columns(pair, q)
            fg = pair_mask(m, pair, quantity=q)
            bg = m["has_both"].to_numpy() & ~fg
            x, y = m[xc].to_numpy(float), m[yc].to_numpy(float)
            if SHOW_REJECTED:
                a.scatter(x[bg], y[bg], s=5, c="blue", edgecolors="black", alpha=0.4)
            a.scatter(x[fg], y[fg], s=10, c="blue", alpha=0.4, edgecolors="none")

            lo, hi = lims[q]
            a.plot([lo, hi], [lo, hi], "k--", lw=0.9)
            a.set_xlim(lo, hi); a.set_ylim(lo, hi)
            n_out = int(np.sum(fg & np.isfinite(x) & np.isfinite(y) &
                               ((x < lo) | (x > hi) | (y < lo) | (y > hi))))
            if n_out:
                a.text(0.97, 0.03, f"{n_out} outside axes", transform=a.transAxes,
                       ha="right", va="bottom", fontsize=7.5, color="0.35")
            a.set_aspect("equal", adjustable="box")
            a.grid(alpha=0.25, lw=0.4)
            xs, ys, _ = PAIRS[pair]
            name = lambda t: "WW3" if t == "ww3" else ("FFT" if t == "F" else "Radon")
            a.set_xlabel(model_T_label(q) if (xs == "ww3" and q in ("Ti", "Ta"))
                         else f"{name(xs)}  {lab}")
            a.set_ylabel(f"{name(ys)}  {lab}")

            if q == "theta":
                st = circ_stats(m.loc[fg, dtheta_col(PAIRS[pair][2])])
                txt = (f"n = {st['n']}\nbias = {st['bias']:+.2f}$^\\circ$\n"
                       f"RMSE = {st['rmse']:.2f}$^\\circ$\n"
                       f"|$\\Delta$|$\\leq$10$^\\circ$: {st['p10']:.0f}%\n({dir_label()})")
            else:
                st = lin_stats(m.loc[fg, xc], m.loc[fg, yc])
                txt = (f"n = {st['n']}\nbias = {st['bias']:+.2f}\n"
                       f"RMSD = {st['rmsd']:.2f}\nSI = {st['si']:.1f}%\n"
                       f"r = {st['r']:.3f}\nslope = {st['slope']:.3f}")
            a.text(0.03, 0.97, txt, transform=a.transAxes, va="top", ha="left",
                   fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                           ec="0.5", lw=0.6, alpha=0.9))
        ax[i, 1].set_title(pair, fontsize=13, fontweight="bold", pad=12)
    fig.suptitle("SWOT HR 5 km boxes against MARC/WW3, identical box set",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("saved:", path)


def diff_hist(m, path, group=None):
    """Both estimators' departure from WW3, overlaid."""
    specs = [("dlambda", r"$\Delta\lambda$  (m)", "lambda"),
             ("dTi", r"$\Delta T_{int}$  (s)", "Ti"),
             ("dTa", r"$\Delta T_{abs}$  (s)", "Ta"),
             ("dtheta", r"$\Delta\theta$  ($^\circ$)", "theta")]
    fig, ax = plt.subplots(1, len(specs), figsize=(4.6*len(specs), 4.4))
    for a, (stem, lab, q) in zip(ax, specs):
        selF = pair_mask(m, "FFT-WW3", quantity=q)
        selR = pair_mask(m, "Radon-WW3", quantity=q)
        if q == "theta":
            dF = m.loc[selF, dtheta_col("FW")].dropna()
            dR = m.loc[selR, dtheta_col("RW")].dropna()
            lab = r"$\Delta\theta$  ($^\circ$), " + dir_label()
        else:
            dF = m.loc[selF, stem + "_FW"].dropna()
            dR = m.loc[selR, stem + "_RW"].dropna()
        both = np.r_[dF.values, dR.values]
        if both.size == 0:
            continue
        rng = np.nanpercentile(both, [1, 99])
        if rng[0] == rng[1]:
            rng = (rng[0]-1, rng[1]+1)
        bins = np.linspace(rng[0], rng[1], 41)
        a.hist(dF, bins=bins, histtype="stepfilled", alpha=0.55,
               color="blue", label=f"FFT $-$ WW3  (med {np.median(dF):+.2f})")
        a.hist(dR, bins=bins, histtype="stepfilled", alpha=0.55,
               color="#c1670c", label=f"Radon $-$ WW3  (med {np.median(dR):+.2f})")
        a.axvline(0, color="k", lw=0.9, ls="--")
        a.set_xlabel(lab); a.set_ylabel("boxes")
        a.grid(alpha=0.25, lw=0.4); a.legend(fontsize=8)
    fig.suptitle("Departure from MARC/WW3, both estimators, identical boxes"
                 + (f"   |   {group}" if group else ""),
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("saved:", path)


# =============================================================================
# v4.3  RESPONSE-THRESHOLD SENSITIVITY AND DOPPLER CHECK
# =============================================================================
def _boot_ci(x, fun, n=N_BOOT_CI, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = [fun(rng.choice(x, x.size)) for _ in range(n)]
    return tuple(np.percentile(v, [2.5, 97.5]))


def _dir_diffs(m, sel, side):
    """Direction differences for the sensitivity tables, per DIR_POPULATION."""
    if DIR_POPULATION == "swot_resolved":
        sel = sel & _flag(m, "lobe_resolved_" + side)
    return m.loc[sel, dtheta_col(side + "W")]


def _population(m, side):
    """Internally consistent, model-collocated boxes, response NOT pre-filtered."""
    ok = m["has_both"].to_numpy().copy() & np.isfinite(m["ww3_lambda_m"].to_numpy(float))
    sides = ["F", "R"] if SUBSET_MODE == "common" else [side]
    for s in sides:
        ok &= _flag(m, "spectral_internal_quality_" + s)
    return ok


def error_vs_response(m):
    """Error against WW3 binned by the estimator's own peak filter response."""
    rows = []
    for side, name in (("F", "FFT"), ("R", "Radon")):
        rc = "resp_pow_at_peak_" + side
        if rc not in m:
            continue
        base = _population(m, side)
        cats = pd.cut(m[rc], RESP_EDGES, include_lowest=True)
        for b in cats.cat.categories:
            sel = base & (cats == b).to_numpy()
            dl = m.loc[sel, f"dlambda_rel_{side}W"]
            dti = m.loc[sel, f"dTi_{side}W"]
            dth = _dir_diffs(m, sel, side)
            rmsd = lambda v: float(np.sqrt(np.mean(np.square(v))))
            lo, hi = _boot_ci(dl, rmsd)
            rows.append(dict(estimator=name, resp_bin=str(b), n=int(sel.sum()),
                             dlam_rel_med_pct=float(dl.median()) if len(dl) else np.nan,
                             dlam_rel_rmsd_pct=rmsd(dl.dropna()) if dl.notna().any() else np.nan,
                             dlam_rel_rmsd_ci_lo=lo, dlam_rel_rmsd_ci_hi=hi,
                             dTi_med_s=float(dti.median()) if len(dti) else np.nan,
                             dTi_rmsd_s=rmsd(dti.dropna()) if dti.notna().any() else np.nan,
                             n_dir=int(dth.notna().sum()),
                             dtheta_rmse_deg=circ_stats(dth)["rmse"]))
    return pd.DataFrame(rows)


def response_warn_sweep(m):
    """Statistics if the observability flag threshold were each sweep value."""
    rows = []
    for thr in RESPONSE_WARN_SWEEP:
        for side, name in (("F", "FFT"), ("R", "Radon")):
            rc = "resp_pow_at_peak_" + side
            if rc not in m:
                continue
            sel = _population(m, side)
            sides = ["F", "R"] if SUBSET_MODE == "common" else [side]
            for s in sides:
                sel &= m["resp_pow_at_peak_" + s].to_numpy(float) >= thr
            sl = lin_stats(m.loc[sel, "ww3_lambda_m"], m.loc[sel, LAMBDA_COL + "_" + side])
            st = lin_stats(m.loc[sel, model_T_col(side, "Ti")], m.loc[sel, TI_COL + "_" + side])
            c = circ_stats(_dir_diffs(m, sel, side))
            rows.append(dict(response_warn=thr, estimator=name, n=sl["n"],
                             lam_bias=sl["bias"], lam_rmsd=sl["rmsd"], lam_si_pct=sl["si"],
                             Ti_bias=st["bias"], Ti_rmsd=st["rmsd"],
                             n_dir=c["n"], dir_rmse=c["rmse"]))
    return pd.DataFrame(rows)


def mask_sweep(m):
    """Primary partition re-extracted with each mask value (sens_mXX_* columns)."""
    rows = []
    for side, name in (("F", "FFT"), ("R", "Radon")):
        base = _population(m, side)
        for mv in SENS_MASKS:
            tag = f"sens_m{int(round(100*mv)):02d}_"
            lc = tag + "lambda_m_" + side; tc = tag + "theta_degN_" + side
            if lc not in m:
                continue
            lam = m[lc].to_numpy(float)
            sel = base & np.isfinite(lam)
            s = lin_stats(m.loc[sel, "ww3_lambda_m"], m.loc[sel, lc])
            shift = np.abs(lam[sel] - m.loc[sel, LAMBDA_COL + "_" + side].to_numpy(float))
            dth_base = np.abs(wrap180(m.loc[sel, tc].to_numpy(float) -
                                      m.loc[sel, DIR_COL + "_" + side].to_numpy(float)))
            seld = sel & (_flag(m, "lobe_resolved_" + side) if DIR_POPULATION == "swot_resolved"
                          else np.ones(len(m), bool))
            dd = wrap180(m.loc[seld, tc] - m.loc[seld, "ww3_dir_degN"])
            c = circ_stats(wrap90(dd) if DIR_POPULATION == "all_axial" else dd)
            rows.append(dict(estimator=name, mask=mv, n=s["n"],
                             n_lost_vs_all=int(base.sum() - sel.sum()),
                             lam_bias=s["bias"], lam_rmsd=s["rmsd"], lam_si_pct=s["si"],
                             med_abs_shift_vs_production_m=float(np.median(shift)) if shift.size else np.nan,
                             p95_abs_shift_vs_production_m=float(np.percentile(shift, 95)) if shift.size else np.nan,
                             med_abs_dtheta_vs_production_deg=float(np.median(dth_base)) if dth_base.size else np.nan,
                             dir_rmse=c["rmse"], n_dir=c["n"]))
        mf = "p1_masked_energy_frac_" + side
        if mf in m:
            v = m.loc[base, mf].dropna()
            if v.size:
                print(f"[sens] {name}: production mask removes >10% of the partition "
                      f"energy in {100*np.mean(v > 0.10):.1f}% of {v.size} boxes "
                      f"(median {100*v.median():.2f}%)")
    return pd.DataFrame(rows)


def plot_error_vs_response(tab, path):
    if tab.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for name, col, mk in (("FFT", "blue", "o"), ("Radon", "#c1670c", "s")):
        t = tab[tab.estimator == name]
        if t.empty:
            continue
        x = np.arange(len(t))
        ax[0].errorbar(x, t["dlam_rel_rmsd_pct"],
                       yerr=[t["dlam_rel_rmsd_pct"] - t["dlam_rel_rmsd_ci_lo"],
                             t["dlam_rel_rmsd_ci_hi"] - t["dlam_rel_rmsd_pct"]],
                       color=col, marker=mk, capsize=3, label=name)
        ax[1].plot(x, t["dTi_rmsd_s"], color=col, marker=mk, label=name)
        ax[2].plot(x, t["dtheta_rmse_deg"], color=col, marker=mk, label=name)
        for a in ax:
            a.set_xticks(x); a.set_xticklabels(t["resp_bin"], rotation=35, fontsize=8)
    ax[0].set_ylabel(r"RMSD $\Delta\lambda/\lambda_{WW3}$ (%)")
    ax[1].set_ylabel(r"RMSD $\Delta T_{int}$ (s)")
    ax[2].set_ylabel(r"RMSE $\Delta\theta$ ($^\circ$)" + "\n" + dir_label())
    for a in ax:
        a.set_xlabel("peak power response bin"); a.grid(alpha=0.3); a.legend(fontsize=8)
        a.axvline(np.searchsorted(RESP_EDGES, 0.5) - 1 - 0.5, color="k", ls=":", lw=0.9)
    fig.suptitle("Retrieval error against MARC/WW3 by instrument response at the peak",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig); print("saved:", path)


def doppler_check(m):
    """
    Paired comparison on the SAME boxes (direction resolved):
        T_int(SWOT) - Tp(WW3)          both intrinsic, no current
        T_abs(SWOT) - T_abs(WW3)       both with k.U (own direction, same current)
    and the Doppler terms themselves, SWOT vs model.
    """
    rows = []
    for side, name in (("F", "FFT"), ("R", "Radon")):
        sel = (_flag(m, "direction_resolved_" + side) & good_mask(m, side)
               & np.isfinite(m[model_T_col(side, "Ti")].to_numpy(float)))
        ti = m.loc[sel, TI_COL + "_" + side].to_numpy(float)
        ta = m.loc[sel, TA_COL + "_" + side].to_numpy(float)
        tp = m.loc[sel, model_T_col(side, "Ti")].to_numpy(float)
        tpa = m.loc[sel, model_T_col(side, "Ta")].to_numpy(float)
        src = (m.loc[sel, "lobe_source_" + side].to_numpy()
               if "lobe_source_" + side in m else np.array(["vb"]*int(sel.sum())))
        ok = np.isfinite(ti) & np.isfinite(ta) & np.isfinite(tp) & np.isfinite(tpa)
        ok &= np.isfinite(m.loc[sel, "ww3_T_absolute_s_" + side].to_numpy(float))
        ok &= np.isfinite(m.loc[sel, "ww3_Tp_s"].to_numpy(float))
        for lab in ("vb", "model", "all"):
            ss = ok & ((src == lab) if lab != "all" else True)
            if ss.sum() < 3:
                rows.append(dict(estimator=name, lobe_source=lab, n=int(ss.sum())))
                continue
            d_i, d_a = ti[ss] - tp[ss], ta[ss] - tpa[ss]
            # Doppler terms are always T_abs - T_int, independent of the reference
            dop_s = ta[ss] - ti[ss]
            dop_m = (m.loc[sel, "ww3_T_absolute_s_" + side].to_numpy(float)[ss]
                     - m.loc[sel, "ww3_Tp_s"].to_numpy(float)[ss])
            rows.append(dict(estimator=name, lobe_source=lab, n=int(ss.sum()),
                             dTi_bias_s=float(d_i.mean()),
                             dTi_rmsd_s=float(np.sqrt(np.mean(d_i**2))),
                             dTa_bias_s=float(d_a.mean()),
                             dTa_rmsd_s=float(np.sqrt(np.mean(d_a**2))),
                             doppler_swot_med_s=float(np.median(dop_s)),
                             doppler_model_med_s=float(np.median(dop_m)),
                             doppler_sign_agree_pct=float(100*np.mean(np.sign(dop_s) == np.sign(dop_m)))))
    return pd.DataFrame(rows)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    frames = []
    for g in GRANULES:
        print(f"\n[gran] {g['storm']} / {g['stem']}")
        d = load_granule(g)
        if d is not None:
            frames.append(d)
    if not frames:
        raise RuntimeError("no granule could be loaded")
    m = pd.concat(frames, ignore_index=True)
    print(f"\n[all ] {len(m)} boxes, {int(m.has_both.sum())} with both retrievals, "
          f"{int(m['ww3_fp_hz'].notna().sum())} with WW3")

    dir_offset = resolve_direction_offset(m)
    m = add_derived(m, dir_offset)

    os.makedirs(OUT_DIR, exist_ok=True)
    tab = stats_table(m, modes=REPORT_MODES)
    pd.set_option("display.width", 250)
    print(f"\n=== agreement statistics, y minus x, quality modes {list(REPORT_MODES)} ===")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_stats.csv"), index=False)

    print(f"\n=== departure from WW3 by depth [{QUALITY_MODE}, {SUBSET_MODE}] ===")
    print(stratify(m, DEPTH_COL, [0, 20, 40, 60, 100, 300], "depth (m)").to_string())
    m["_ct_km"] = np.abs(m["cross_track_m"])/1000.0
    print(f"\n=== departure from WW3 by |cross-track| ===")
    print(stratify(m, "_ct_km", [0, 10, 20, 30, 40, 60], "|cross-track| (km)").to_string())

    # ---- v4.3 sensitivity -----------------------------------------------------
    ev = error_vs_response(m)
    print("\n=== error against WW3 by peak response (internal-quality boxes, response not pre-filtered) ===")
    print(ev.round(3).to_string(index=False))
    ev.to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_sens_error_vs_response.csv"), index=False)
    plot_error_vs_response(ev, os.path.join(OUT_DIR, "fft_radon_ww3_sens_response.png"))

    ws = response_warn_sweep(m)
    print("\n=== RESPONSE_WARN sweep: statistics on boxes with peak response >= threshold ===")
    print(ws.round(3).to_string(index=False))
    ws.to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_sens_warn_sweep.csv"), index=False)

    ms = mask_sweep(m)
    if not ms.empty:
        print("\n=== RESPONSE_MIN_MASK sweep: primary partition re-extracted per mask ===")
        print(ms.round(3).to_string(index=False))
        ms.to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_sens_mask_sweep.csv"), index=False)
    else:
        print("\n[sens] no sens_mXX_* columns: rerun the v4.3 pipelines for the mask sweep")

    dc = doppler_check(m)
    print("\n=== Paired periods on direction-resolved boxes: SWOT T_int and T_abs vs model "
          + {"absolute": "T_abs (both)", "intrinsic": "T_int (both)",
             "matched": "T_int / T_abs"}[WW3_PERIOD_REFERENCE] + " ===")
    print(dc.round(3).to_string(index=False))
    dc.to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_doppler.csv"), index=False)
    for s_ in ("F", "R"):
        if "vb_agrees_model_" + s_ in m:
            v = m.loc[_flag(m, "lobe_resolved_" + s_), "vb_agrees_model_" + s_].dropna()
            if v.size:
                print(f"[lobe] {'FFT' if s_ == 'F' else 'Radon'}: SWOT VB lobe within 90 deg of "
                      f"the model in {100*v.mean():.1f}% of {v.size} VB-resolved boxes")

    keep = ([c for c in KEY] + ["storm", "stem", "lon_center", "lat_center",
            DEPTH_COL, "cross_track_m", UE_COL + "_F", UN_COL + "_F",
            UE_COL + "_R", UN_COL + "_R", "n_model_points", "acq_time_utc_F",
            "ww3_fp_hz", "ww3_Tp_model_s", "ww3_lambda_m", "ww3_lambda_from_fp_m",
            "ww3_dir_degN", "ww3_lm_m",
            "ww3_Tp_s", "ww3_U_along_mps_F", "ww3_U_along_mps_R",
            LAMBDA_COL + "_F", LAMBDA_COL + "_R",
            TI_COL + "_F", TI_COL + "_R", TA_COL + "_F", TA_COL + "_R",
            DIR_COL + "_F", DIR_COL + "_R",
            "dlambda_FW", "dlambda_RW", "dlambda_RF",
            "dTi_FW", "dTi_RW", "dTi_RF", "dTa_FW", "dTa_RW", "dTa_RF",
            "dtheta_FW", "dtheta_RW", "dtheta_RF",   # RF present only if enabled
            "dtheta_ax_FW", "dtheta_ax_RW", "dtheta_ax_RF",
            "resp_pow_at_peak_F", "resp_pow_at_peak_R",
            "p1_masked_energy_frac_F", "p1_masked_energy_frac_R",
            "usable_F", "usable_R", "lobe_resolved_F", "lobe_resolved_R",
            "lobe_source_F", "lobe_source_R", "direction_resolved_F", "direction_resolved_R",
            "U_along_wave_mps_F", "U_along_wave_mps_R", "doppler_rel_pct_F", "doppler_rel_pct_R",
            "instrument_observability_F", "instrument_observability_R",
            "observability_reason_F", "observability_reason_R"])
    keep = [c for c in keep if c in m.columns]
    m[keep].to_csv(os.path.join(OUT_DIR, "fft_radon_ww3_perbox.csv"), index=False)
    print("\nwrote:", os.path.join(OUT_DIR, "fft_radon_ww3_perbox.csv"))

    scatter_grid(m, os.path.join(OUT_DIR, "fft_radon_ww3_scatter.png"))
    diff_hist(m, os.path.join(OUT_DIR, "fft_radon_ww3_hist.png"))

    # ---- one figure set per storm (or granule) --------------------------------
    if PER_GROUP_FIGURES:
        for gname, g in m.groupby(PER_GROUP_FIGURES):
            if not np.isfinite(g["ww3_lambda_m"].to_numpy(float)).any():
                print(f"[fig ] {gname}: no model collocation, per-group figure skipped")
                continue
            tag = str(gname).replace(" ", "_")
            scatter_grid(g, os.path.join(OUT_DIR, f"fft_radon_ww3_scatter_{tag}.png"), group=str(gname))
            diff_hist(g, os.path.join(OUT_DIR, f"fft_radon_ww3_hist_{tag}.png"), group=str(gname))
        # per-group statistics table, same content as the storm rows of the main table
        by = PER_GROUP_FIGURES
        rows = []
        for gname, g in m.groupby(by):
            t = stats_table(g.assign(storm=str(gname)), modes=(QUALITY_MODE,))
            t = t[t["storm"] == "ALL"].copy()     # one group -> its rows are labelled ALL
            t["storm"] = str(gname)
            rows.append(t)
        if rows:
            tg = pd.concat(rows, ignore_index=True).rename(columns={"storm": by})
            cols = [by, "pair", "lam_n", "lam_bias", "lam_rmsd", "lam_si_pct", "lam_r",
                    "Ti_n", "Ti_bias", "Ti_rmsd", "Ti_si_pct", "Ti_r",
                    "Ta_n", "Ta_bias", "Ta_rmsd", "Ta_si_pct", "Ta_r",
                    "ax_n", "ax_bias", "ax_rmse", "ax_within10_pct"]
            print(f"\n=== per-{by} statistics [{QUALITY_MODE}, {SUBSET_MODE}] ===")
            print(tg[[c for c in cols if c in tg]].round(2).to_string(index=False))
            tg.to_csv(os.path.join(OUT_DIR, f"fft_radon_ww3_stats_per_{by}.csv"), index=False)

    print("\nNote: model lambda_p from fp by dispersion. WW3_PERIOD_REFERENCE = "
          f"'{WW3_PERIOD_REFERENCE}': SWOT T_int is compared with "
          + ("WW3 T_abs" if model_T_col("F", "Ti").startswith("ww3_T_abs") else "WW3 T_int")
          + ", SWOT T_abs with "
          + ("WW3 T_abs" if model_T_col("F", "Ta").startswith("ww3_T_abs") else "WW3 T_int")
          + " (WW3 T_abs = lambda_p + k.U, model direction, same current as that estimator)."
          + f" Model wavelength source: {WW3_LAMBDA_SOURCE}. Direction figures use: {dir_label()}.")