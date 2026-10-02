# -*- coding: utf-8 -*-
"""
SWOT HR v4 against MARC/WW3: per-box comparison and along-track transfer.

Updated for the v4 column names and, more importantly, for two things the
previous version got wrong scientifically.

WHAT CHANGED AND WHY
====================

1. Direction is compared ONLY on lobe-resolved boxes
-----------------------------------------------------
theta_obs_degN is written for every box, but where lobe_resolved is False the
180 deg choice was never made and the value is close to a coin flip. Including
those boxes mixes a random sign into the comparison and makes the direction
statistics meaningless. In the Ciaran scene that is 183 of 363 boxes.

lambda_obs_m and Tp_obs_intrinsic_s are NOT affected by the lobe choice, so
they are compared over the full quality selection.
Tp_obs_current_corr_s IS affected, because the Doppler term k.U reverses sign
with the lobe, so it is also restricted to resolved boxes.

2. The direction convention is tested, not assumed
---------------------------------------------------
DIR_CONVENTION_OFF was hardwired to 180. The script now evaluates the circular
bias and RMSE under both 0 and 180 on the resolved boxes and reports which one
fits, alongside the WW3 variable metadata. Set DIR_OFFSET_MODE = "auto" to let
it choose, or pin it once you have confirmed the convention from the metadata.

3. "Percent inside the SWOT band" is demoted to a secondary metric
-------------------------------------------------------------------
theta_CI_half_deg is the bootstrap spread over Welch windows. It is the
STATISTICAL uncertainty only and its median is about 2.5 deg. It does not
contain the instrumental distortion, which is the dominant error. A stably
distorted peak has a tight CI and is still wrong. Expect a low coverage
percentage; that is the expected behaviour of an honest statistical CI, not a
failure of the retrieval. Bias and RMSE are reported as the primary metrics.

4. WW3 wavelength is derived from fp through the dispersion relation
---------------------------------------------------------------------
Comparing a SWOT peak wavelength against a WW3 mean wavelength (lm) is a
mismatch of definitions. The script instead inverts

    omega = sqrt(g k tanh(k h)) + k . U

for k at each box, using the same depth and current the pipeline used, so the
model and the observation refer to the same quantity. If lm is present it is
carried as a secondary column but is not the primary comparison.

5. The along-track transfer characterisation is included
----------------------------------------------------------
This is the point of the exercise. Yu et al. find the along-track frequency
centroid ratio KaRIn/buoy has a median of 0.77 while the across-track ratio is
about 1.0, and state that the distortion cannot be represented by a linear
transfer function in general. That 0.77 is a dataset statistic from open-ocean
Cal/Val cases, computed in radar geometry at SWH 2-4 m.

This script MEASURES the equivalent ratio for your regime: ground geometry,
coastal, storm sea states, from your own WW3 collocation. It regresses
k_along_obs against k_along_ww3 and k_across_obs against k_across_ww3 through
the origin, and reports the slopes with their scatter, binned by cross-track
distance. The result is a characterisation with a validity map, not a
correction applied upstream.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

from swothr.config import path, apply_overrides

ROOT = path("root")             # config/paths.yaml


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = Path(
    f"{ROOT}/Outputs/swot_5km_FFT_Ciaran_v4.csv"
)

MODEL_FILE = Path(
    f"{ROOT}/Data/Wave Validation/MARC_WW3-NORGAS-2M_20231104T01Z.nc"
)

OUTPUT_DIR = CSV_FILE.parent / "model_validation_Ciaran_v4"
MODEL_TIME = np.datetime64("2023-11-04T01:00:00")
apply_overrides(globals(), "validate")
CSV_FILE, MODEL_FILE, OUTPUT_DIR = Path(CSV_FILE), Path(MODEL_FILE), Path(OUTPUT_DIR)
MODEL_TIME = np.datetime64(MODEL_TIME) if isinstance(MODEL_TIME, str) else MODEL_TIME
BOX_CRS = "EPSG:32630"

g = 9.81

# ---- quality selection ------------------------------------------------------
# "usable"      : spectral_internal_quality AND observability in A/B
# "observability": grade in OBS_GRADES_KEEP
# "all"         : every row
SELECTION = "usable"
OBS_GRADES_KEEP = ("A", "B")

# Direction and the current-corrected period are only meaningful where the
# 180 deg ambiguity was resolved. Leave this True.
DIRECTION_REQUIRES_RESOLVED_LOBE = True

# ---- direction convention ---------------------------------------------------
# "auto" tests 0 and 180 and picks the lower circular RMSE.
# Pin to 0 or 180 once the WW3 metadata has been confirmed.
DIR_OFFSET_MODE = "auto"
DIR_OFFSET_FIXED = 0.0

# ---- model variables --------------------------------------------------------
WW3_DIR_CANDIDATES = ["dp", "dir", "th1p", "mean_direction"]
WW3_FP_CANDIDATES = ["fp", "peak_frequency"]
WW3_LM_CANDIDATES = ["lm", "lambda", "wavelength"]

# Is WW3 fp the absolute (Eulerian) frequency? True when the model was run with
# current forcing. If False, fp is intrinsic and the current term is skipped.
WW3_FP_IS_ABSOLUTE = True

# ---- period to compare ------------------------------------------------------
# "current_corr" -> Tp_obs_current_corr_s   (needs a resolved lobe)
# "intrinsic"    -> Tp_obs_intrinsic_s      (lobe-independent)
TP_KIND = "current_corr"

# ---- profile plots ----------------------------------------------------------
# "box" keeps the pipeline ordering, which is x-outer and therefore a sawtooth
# in space. "latitude" or "along_track" give a spatially meaningful transect.
SORT_BY = "latitude"
MAX_X_TICKS = 25
SHOW_ALL_BOX_TICK_LABELS = False
DPI = 300


# ============================================================
# COLUMN MAP (v4)
# ============================================================

COL = {
    "lambda": "lambda_obs_m",
    "lambda_lo": "lambda_CI_low_m",
    "lambda_hi": "lambda_CI_high_m",
    "theta": "theta_obs_degN",
    "theta_half": "theta_CI_half_deg",
    "tp_intrinsic": "Tp_obs_intrinsic_s",
    "tp_intrinsic_lo": "T_intrinsic_from_lambda_obs_CI_low_s",
    "tp_intrinsic_hi": "T_intrinsic_from_lambda_obs_CI_high_s",
    "tp_current": "Tp_obs_current_corr_s",
    "tp_current_lo": "T_absolute_from_lambda_obs_current_CI_low_s",
    "tp_current_hi": "T_absolute_from_lambda_obs_current_CI_high_s",
    "k_along": "k_along_obs_radm",
    "k_across": "k_across_obs_radm",
    "kx": "kx_obs_radm",
    "ky": "ky_obs_radm",
    "along_track": "along_track_degN",
    "depth": "Depth_m",
    "u_e": "U_e_mps",
    "u_n": "U_n_mps",
    "cross_track": "cross_track_m",
    "resolved": "lobe_resolved",
}


# ============================================================
# HELPERS
# ============================================================

apply_overrides(globals(), "validate", final=True)


def find_name(ds, candidates, label, required=True):
    available = set(ds.variables) | set(ds.coords)
    for name in candidates:
        if name in available:
            return name
    if not required:
        return None
    raise KeyError(f"Could not identify {label}. Tried {candidates}. "
                   f"Available: {sorted(available)}")


def select_time(ds, target_time):
    for name in ["time", "time_counter", "valid_time", "forecast_time"]:
        if name in ds.coords or name in ds.variables:
            try:
                sel = ds.sel({name: target_time}, method="nearest")
                actual = pd.to_datetime(np.asarray(sel[name].values).item())
                offset = abs(pd.Timestamp(actual) - pd.Timestamp(target_time))
                print(f"[model] time {actual}  (offset {offset})")
                if offset > pd.Timedelta("1h"):
                    print("[model] WARNING: nearest model time is more than "
                          "1 h from the requested time")
                return sel
            except (KeyError, TypeError, ValueError):
                continue
    print("[model] no selectable time coordinate; using as supplied")
    return ds


def as_2d(da, shape, name):
    arr = np.asarray(da.squeeze(drop=True).values, dtype=float)
    if arr.shape == shape:
        return arr
    if arr.ndim == 2 and arr.T.shape == shape:
        return arr.T
    raise ValueError(f"'{name}' has shape {arr.shape}, grid is {shape}")


def circ_mean_deg(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    a = np.deg2rad(v)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(a)),
                                       np.mean(np.cos(a)))) % 360.0)


def circ_diff_deg(a, b):
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            + 180.0) % 360.0 - 180.0


def circ_stats(model_dir, swot_dir):
    """Circular bias, RMSE and MAE of model minus SWOT."""
    d = circ_diff_deg(model_dir, swot_dir)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, mae=np.nan, within_30=np.nan)
    return dict(n=int(d.size),
                bias=float(circ_mean_deg(d) if circ_mean_deg(d) <= 180
                           else circ_mean_deg(d) - 360),
                rmse=float(np.sqrt(np.mean(d ** 2))),
                mae=float(np.mean(np.abs(d))),
                within_30=float(np.mean(np.abs(d) <= 30.0)))


def wavenumber_from_frequency(f_abs, depth, u_along_dir, absolute=True):
    """
    Invert omega = sqrt(g k tanh(k h)) + k * u_along_dir for k.

    u_along_dir is the current component along the wave propagation direction.
    Uses bisection; the right-hand side is monotonic in k over the bracket for
    any physically sensible current.
    """
    f_abs = np.asarray(f_abs, dtype=float)
    depth = np.asarray(depth, dtype=float)
    u = np.asarray(u_along_dir, dtype=float) if absolute else np.zeros_like(f_abs)

    out = np.full(f_abs.shape, np.nan)
    ok = np.isfinite(f_abs) & (f_abs > 0) & np.isfinite(depth) & (depth > 0)
    if not ok.any():
        return out

    omega = 2.0 * np.pi * f_abs[ok]
    h = depth[ok]
    uu = np.where(np.isfinite(u[ok]), u[ok], 0.0)

    lo = np.full(omega.shape, 1e-5)
    hi = np.full(omega.shape, 1.0)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = np.sqrt(g * mid * np.tanh(mid * h)) + mid * uu
        too_small = val < omega
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)

    out[ok] = 0.5 * (lo + hi)
    return out


def box_sort_number(name):
    m = re.search(r"\d+", str(name))
    return int(m.group()) if m else 10 ** 12


def apply_selection(df):
    if SELECTION == "all":
        return np.ones(len(df), dtype=bool)
    if SELECTION == "observability":
        return df["instrument_observability"].isin(OBS_GRADES_KEEP).values
    if "usable" not in df:
        raise KeyError("CSV has no 'usable' column; set SELECTION")
    return df["usable"].astype(bool).values


# ============================================================
# LOAD SWOT
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
swot_all = pd.read_csv(CSV_FILE)

missing = [c for c in COL.values() if c not in swot_all.columns]
if missing:
    raise KeyError(f"CSV is missing expected v4 columns: {missing}")

keep = apply_selection(swot_all)
swot = swot_all.loc[keep].copy()

print(f"[swot ] {len(swot_all)} boxes, {len(swot)} after SELECTION='{SELECTION}'")
print(f"[swot ] lobe resolved within the selection: "
      f"{int(swot[COL['resolved']].sum())}/{len(swot)}")

if SORT_BY == "latitude":
    swot = swot.sort_values("lat_center")
elif SORT_BY == "along_track":
    a = np.deg2rad(swot[COL["along_track"]].median())
    swot = swot.assign(
        _s=swot["x_center"] * np.sin(a) + swot["y_center"] * np.cos(a)
    ).sort_values("_s")
else:
    swot = swot.assign(_n=swot["Box"].map(box_sort_number)).sort_values("_n")

swot = swot.reset_index(drop=True)

if TP_KIND == "current_corr":
    tp_col, tp_lo, tp_hi = COL["tp_current"], COL["tp_current_lo"], COL["tp_current_hi"]
    tp_desc, tp_needs_lobe = "current-corrected", True
else:
    tp_col, tp_lo, tp_hi = COL["tp_intrinsic"], COL["tp_intrinsic_lo"], COL["tp_intrinsic_hi"]
    tp_desc, tp_needs_lobe = "intrinsic", False


# ============================================================
# LOAD WW3
# ============================================================

model = xr.open_dataset(MODEL_FILE)
model = select_time(model, MODEL_TIME)

lon_name = find_name(model, ["longitude", "lon", "nav_lon", "x"], "longitude")
lat_name = find_name(model, ["latitude", "lat", "nav_lat", "y"], "latitude")
dir_name = find_name(model, WW3_DIR_CANDIDATES, "WW3 direction")
fp_name = find_name(model, WW3_FP_CANDIDATES, "WW3 peak frequency")
lm_name = find_name(model, WW3_LM_CANDIDATES, "WW3 wavelength", required=False)

lon = np.asarray(model[lon_name].squeeze(drop=True).values, dtype=float)
lat = np.asarray(model[lat_name].squeeze(drop=True).values, dtype=float)
lon2d, lat2d = (np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat))
shape = lon2d.shape

fp = as_2d(model[fp_name].where(model[fp_name] > 0), shape, fp_name)
dp = as_2d(model[dir_name], shape, dir_name) % 360.0
lm = as_2d(model[lm_name].where(model[lm_name] > 0), shape, lm_name) \
     if lm_name else np.full(shape, np.nan)

print(f"\n[model] direction variable : {dir_name}")
print(f"[model] metadata           : {model[dir_name].attrs}")
print("[model] READ THE long_name/comment ABOVE. 'from' matches the SWOT "
      "convention and needs offset 0; 'to'/'towards' needs 180.")
print(f"[model] peak frequency     : {fp_name}")
if lm_name:
    print(f"[model] wavelength (2nd)   : {lm_name} {model[lm_name].attrs.get('long_name','')}")

transformer = Transformer.from_crs("EPSG:4326", BOX_CRS, always_xy=True)
mx, my = transformer.transform(lon2d, lat2d)
fx, fy = np.asarray(mx).ravel(), np.asarray(my).ravel()
f_fp, f_dp, f_lm = fp.ravel(), dp.ravel(), lm.ravel()


# ============================================================
# COLLOCATE
# ============================================================

rows = []
for _, r in swot.iterrows():
    ins = (np.isfinite(fx) & np.isfinite(fy) &
           (fx >= r["x_min"]) & (fx <= r["x_max"]) &
           (fy >= r["y_min"]) & (fy <= r["y_max"]))

    v_fp = f_fp[ins][np.isfinite(f_fp[ins])]
    v_dp = f_dp[ins][np.isfinite(f_dp[ins])]
    v_lm = f_lm[ins][np.isfinite(f_lm[ins])]

    rows.append({
        **r.to_dict(),
        "ww3_fp_hz": float(np.mean(v_fp)) if v_fp.size else np.nan,
        "ww3_Tp_model_s": float(1.0 / np.mean(v_fp)) if v_fp.size else np.nan,
        "ww3_dir_raw_degN": circ_mean_deg(v_dp) if v_dp.size else np.nan,
        "ww3_lm_m": float(np.mean(v_lm)) if v_lm.size else np.nan,
        "n_model_points": int(ins.sum()),
        "n_model_valid_fp": int(v_fp.size),
        "n_model_valid_dir": int(v_dp.size),
    })

val = pd.DataFrame(rows)
print(f"\n[collo] boxes with finite WW3 fp  : "
      f"{int(val['ww3_fp_hz'].notna().sum())}/{len(val)}")
print(f"[collo] boxes with finite WW3 dir : "
      f"{int(val['ww3_dir_raw_degN'].notna().sum())}/{len(val)}")


# ============================================================
# DIRECTION CONVENTION TEST
# ============================================================

res = val[COL["resolved"]].astype(bool).values
if not DIRECTION_REQUIRES_RESOLVED_LOBE:
    res = np.ones(len(val), dtype=bool)

print(f"\n[dir  ] testing the convention on {int(res.sum())} lobe-resolved boxes")
cand = {}
for off in (0.0, 180.0):
    s = circ_stats((val["ww3_dir_raw_degN"].values[res] + off) % 360.0,
                   val[COL["theta"]].values[res])
    cand[off] = s
    print(f"        offset {off:5.0f} deg -> bias {s['bias']:+7.1f}  "
          f"RMSE {s['rmse']:6.1f}  MAE {s['mae']:6.1f}  "
          f"within 30 deg {100*s['within_30']:5.1f} %  (n={s['n']})")

if DIR_OFFSET_MODE == "auto":
    DIR_OFFSET = min(cand, key=lambda k: cand[k]["rmse"]
                     if np.isfinite(cand[k]["rmse"]) else np.inf)
    print(f"[dir  ] auto-selected offset = {DIR_OFFSET:.0f} deg")
    if np.isfinite(cand[0.0]["rmse"]) and np.isfinite(cand[180.0]["rmse"]):
        ratio = max(cand[0.0]["rmse"], cand[180.0]["rmse"]) / \
                max(min(cand[0.0]["rmse"], cand[180.0]["rmse"]), 1e-9)
        if ratio < 1.3:
            print("[dir  ] WARNING: the two offsets fit almost equally well. "
                  "The comparison cannot settle the convention; confirm it "
                  "from the metadata above.")
else:
    DIR_OFFSET = float(DIR_OFFSET_FIXED)
    print(f"[dir  ] using pinned offset = {DIR_OFFSET:.0f} deg")

val["ww3_dir_degN"] = (val["ww3_dir_raw_degN"] + DIR_OFFSET) % 360.0


# ============================================================
# WW3 WAVELENGTH FROM DISPERSION
# ============================================================

# current component along the WW3 propagation direction
th_prop = np.deg2rad((val["ww3_dir_degN"].values + 180.0) % 360.0)  # "from" -> "to"
u_along = (val[COL["u_e"]].values * np.sin(th_prop) +
           val[COL["u_n"]].values * np.cos(th_prop))

k_ww3 = wavenumber_from_frequency(val["ww3_fp_hz"].values,
                                  val[COL["depth"]].values,
                                  u_along,
                                  absolute=WW3_FP_IS_ABSOLUTE)

val["ww3_k_radm"] = k_ww3
val["ww3_lambda_m"] = np.where(np.isfinite(k_ww3) & (k_ww3 > 0),
                               2.0 * np.pi / k_ww3, np.nan)

print(f"\n[disp ] WW3 lambda from {fp_name}: median "
      f"{np.nanmedian(val['ww3_lambda_m']):.0f} m")
if lm_name:
    print(f"[disp ] WW3 {lm_name} (mean wavelength, secondary): median "
          f"{np.nanmedian(val['ww3_lm_m']):.0f} m")
print(f"[disp ] SWOT lambda_obs: median "
      f"{np.nanmedian(val[COL['lambda']]):.0f} m")


# ============================================================
# ALONG-TRACK TRANSFER CHARACTERISATION
# ============================================================

a_deg = val[COL["along_track"]].values
a_e, a_n = np.sin(np.deg2rad(a_deg)), np.cos(np.deg2rad(a_deg))
r_e, r_n = a_n, -a_e                      # look direction, 90 deg from track

kx_w = val["ww3_k_radm"].values * np.sin(th_prop)
ky_w = val["ww3_k_radm"].values * np.cos(th_prop)

val["ww3_k_along_radm"] = kx_w * a_e + ky_w * a_n
val["ww3_k_across_radm"] = np.abs(kx_w * r_e + ky_w * r_n)
val["swot_k_across_abs_radm"] = np.abs(val[COL["k_across"]].values)


def slope_through_origin(x, y):
    """Least-squares slope with no intercept, plus the ratio distribution."""
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-6)
    if ok.sum() < 5:
        return dict(n=int(ok.sum()), slope=np.nan, r=np.nan,
                    ratio_med=np.nan, ratio_iqr=(np.nan, np.nan))
    xs, ys = x[ok], y[ok]
    slope = float(np.sum(xs * ys) / np.sum(xs * xs))
    r = float(np.corrcoef(xs, ys)[0, 1])
    ratio = ys / xs
    return dict(n=int(ok.sum()), slope=slope, r=r,
                ratio_med=float(np.median(ratio)),
                ratio_iqr=tuple(np.percentile(ratio, [25, 75])))


print("\n" + "=" * 68)
print("ALONG-TRACK TRANSFER CHARACTERISATION  (SWOT / WW3)")
print("=" * 68)
print("Yu et al. report a median along-track ratio of 0.77 and an across-track")
print("ratio near 1.0, in radar geometry, open ocean, SWH 2-4 m. The numbers")
print("below are the equivalent for THIS regime. They are a result, not a")
print("correction: nothing downstream is rescaled by them.\n")

al = slope_through_origin(val["ww3_k_along_radm"].values[res],
                          val[COL["k_along"]].values[res])
ac = slope_through_origin(val["ww3_k_across_radm"].values[res],
                          val["swot_k_across_abs_radm"].values[res])

for lab, s in [("along-track k", al), ("across-track k", ac)]:
    print(f"  {lab:16s} n={s['n']:4d}  slope={s['slope']:.3f}  r={s['r']:.3f}  "
          f"median ratio={s['ratio_med']:.3f}  "
          f"IQR {s['ratio_iqr'][0]:.3f}-{s['ratio_iqr'][1]:.3f}")

if COL["cross_track"] in val:
    print("\n  binned by |cross-track| distance:")
    rho = np.abs(val[COL["cross_track"]].values) / 1e3
    for lo, hi in [(25, 35), (35, 45), (45, 60)]:
        m = res & (rho >= lo) & (rho < hi)
        if m.sum() >= 5:
            s = slope_through_origin(val["ww3_k_along_radm"].values[m],
                                     val[COL["k_along"]].values[m])
            print(f"    {lo}-{hi} km : n={s['n']:4d}  "
                  f"along-track slope={s['slope']:.3f}  r={s['r']:.3f}")


# ============================================================
# PLOT HELPERS
# ============================================================

def axis_setup(data):
    x = np.arange(len(data), dtype=float)
    labels = data["Box"].astype(str).tolist()
    if SHOW_ALL_BOX_TICK_LABELS or len(data) <= MAX_X_TICKS:
        ticks = np.arange(len(data), dtype=int)
    else:
        ticks = np.unique(np.linspace(0, len(data) - 1, MAX_X_TICKS, dtype=int))
    return x, labels, ticks


def format_axis(ax, x, labels, ticks):
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_xticks(x[ticks])
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel(f"Box (sorted by {SORT_BY})")
    ax.grid(True, alpha=0.25)


def stats_box(ax, text):
    ax.text(0.01, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})


def linear_profile(data, sc, lo, hi, mc, ylabel, title, slab, mlab, out):
    x, labels, ticks = axis_setup(data)
    s = data[sc].to_numpy(float)
    m = data[mc].to_numpy(float)
    l_, h_ = data[lo].to_numpy(float), data[hi].to_numpy(float)

    ok = np.isfinite(s) & np.isfinite(m)
    d = m[ok] - s[ok]
    inside = np.mean((m[ok] >= np.minimum(l_[ok], h_[ok])) &
                     (m[ok] <= np.maximum(l_[ok], h_[ok]))) if ok.any() else np.nan

    fig, ax = plt.subplots(figsize=(18, 6.2))
    line = ax.plot(x, s, marker="o", ms=3.2, lw=1.25, label=slab, zorder=3)[0]
    ax.fill_between(x, l_, h_, color=line.get_color(), alpha=0.22,
                    label="SWOT statistical 95% CI", zorder=1)
    ax.scatter(x[np.isfinite(m)], m[np.isfinite(m)], marker="x", s=45,
               linewidths=1.5, label=mlab, zorder=4, color="crimson")

    format_axis(ax, x, labels, ticks)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    stats_box(ax,
              f"n = {int(ok.sum())} of {len(data)}\n"
              f"bias (model - SWOT) = {np.mean(d):+.2f}\n"
              f"RMSE = {np.sqrt(np.mean(d**2)):.2f}\n"
              f"model inside statistical CI = {100*inside:.1f} %")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return dict(n=int(ok.sum()), bias=float(np.mean(d)),
                rmse=float(np.sqrt(np.mean(d ** 2))), inside=float(inside))


def direction_profile(data, out):
    x, labels, ticks = axis_setup(data)
    s = data[COL["theta"]].to_numpy(float) % 360.0
    m = data["ww3_dir_degN"].to_numpy(float) % 360.0
    hw = data[COL["theta_half"]].to_numpy(float)

    su = np.rad2deg(np.unwrap(np.deg2rad(s)))
    ok = np.isfinite(m) & np.isfinite(s)
    mu = np.full(m.shape, np.nan)
    mu[ok] = su[ok] + circ_diff_deg(m[ok], s[ok])

    st = circ_stats(m, s)
    inside = np.mean(np.abs(circ_diff_deg(m[ok], s[ok])) <= hw[ok]) if ok.any() else np.nan

    fig, ax = plt.subplots(figsize=(18, 6.2))
    line = ax.plot(x, su, marker="o", ms=3.2, lw=1.25,
                   label="SWOT direction (lobe-resolved)", zorder=3)[0]
    ax.fill_between(x, su - hw, su + hw, color=line.get_color(), alpha=0.22,
                    label="SWOT statistical 95% CI", zorder=1)
    ax.scatter(x[ok], mu[ok], marker="x", s=45, linewidths=1.5,
               label=f"MARC/WW3 {dir_name} (+{DIR_OFFSET:.0f} deg)",
               zorder=4, color="crimson")

    format_axis(ax, x, labels, ticks)
    ax.set_ylabel("Peak direction (deg N)")
    ax.set_title("Peak direction by box: SWOT against MARC/WW3")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v % 360:.0f}"))
    ax.legend(loc="best")
    stats_box(ax,
              f"n = {st['n']} lobe-resolved of {len(data)}\n"
              f"circular bias = {st['bias']:+.1f} deg\n"
              f"circular RMSE = {st['rmse']:.1f} deg\n"
              f"within 30 deg = {100*st['within_30']:.1f} %\n"
              f"model inside statistical CI = {100*inside:.1f} %")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return st


def transfer_figure(data, mask, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, (xc, yc, lab, s) in zip(axes, [
        ("ww3_k_along_radm", COL["k_along"], "along-track", al),
        ("ww3_k_across_radm", "swot_k_across_abs_radm", "across-track", ac),
    ]):
        xv = data[xc].values[mask]
        yv = data[yc].values[mask]
        ax.scatter(xv, yv, s=16, alpha=0.6, c="#3b7dd8", linewidths=0)
        lim = np.nanpercentile(np.abs(np.r_[xv, yv]), 99)
        gl = np.linspace(-lim, lim, 5) if lab == "along-track" else np.linspace(0, lim, 5)
        ax.plot(gl, gl, "k--", lw=1, label="1:1")
        if np.isfinite(s["slope"]):
            ax.plot(gl, s["slope"] * gl, "r-", lw=1.4,
                    label=f"slope {s['slope']:.3f}")
        ax.set_xlabel(f"WW3 {lab} k  [rad m$^{{-1}}$]")
        ax.set_ylabel(f"SWOT {lab} k  [rad m$^{{-1}}$]")
        ax.set_title(f"{lab}   n={s['n']}   r={s['r']:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Along- and across-track wavenumber transfer, SWOT against WW3")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RUN PLOTS
# ============================================================

summary = []

s = linear_profile(val, COL["lambda"], COL["lambda_lo"], COL["lambda_hi"],
                   "ww3_lambda_m",
                   "Peak wavelength, lambda (m)",
                   f"Peak wavelength by box: SWOT against WW3 (from {fp_name} "
                   f"via dispersion)",
                   "SWOT observed wavelength", f"WW3 lambda from {fp_name}",
                   OUTPUT_DIR / "profile_lambda.png")
summary.append({"Metric": "Peak wavelength", **s})

tp_mask = val[COL["resolved"]].astype(bool) if tp_needs_lobe \
    else pd.Series(True, index=val.index)
s = linear_profile(val[tp_mask].reset_index(drop=True),
                   tp_col, tp_lo, tp_hi, "ww3_Tp_model_s",
                   "Peak period, Tp (s)",
                   f"Peak period by box: SWOT {tp_desc} against WW3 1/{fp_name}",
                   f"SWOT {tp_desc} period", f"WW3 Tp = 1/{fp_name}",
                   OUTPUT_DIR / f"profile_Tp_{TP_KIND}.png")
summary.append({"Metric": f"Peak period ({tp_desc})", **s})

dstat = direction_profile(val[res].reset_index(drop=True),
                          OUTPUT_DIR / "profile_direction.png")
summary.append({"Metric": "Peak direction", "n": dstat["n"],
                "bias": dstat["bias"], "rmse": dstat["rmse"],
                "inside": np.nan})

transfer_figure(val, res, OUTPUT_DIR / "along_track_transfer.png")


# ============================================================
# SAVE
# ============================================================

val["model_lambda_inside_CI"] = (
    val["ww3_lambda_m"].between(val[COL["lambda_lo"]], val[COL["lambda_hi"]]))
val["model_Tp_inside_CI"] = val["ww3_Tp_model_s"].between(val[tp_lo], val[tp_hi])
val["model_dir_within_CI"] = (
    np.abs(circ_diff_deg(val["ww3_dir_degN"], val[COL["theta"]]))
    <= val[COL["theta_half"]])
val["dir_comparison_valid"] = val[COL["resolved"]].astype(bool)

val.to_csv(OUTPUT_DIR / "SWOT_WW3_collocated_v4.csv", index=False)
pd.DataFrame(summary).to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
pd.DataFrame([
    {"quantity": "k_along", **al},
    {"quantity": "k_across", **ac},
]).to_csv(OUTPUT_DIR / "along_track_transfer_stats.csv", index=False)

print("\n" + pd.DataFrame(summary).to_string(index=False))
print(f"\n[out  ] {OUTPUT_DIR}")
print("\nNote: 'model inside statistical CI' will be low. theta_CI_half_deg and")
print("the lambda CI are bootstrap spreads over Welch windows and contain no")
print("instrumental bias term. Bias and RMSE are the metrics to quote.")

model.close()