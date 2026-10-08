#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import textwrap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# USER SETTINGS
# =============================================================================
from swothr.config import path, apply_overrides

BASE = path("root")             # config/paths.yaml
FFT_CSV = f"{BASE}/Results/FFT/swot_5km_FFT_ciaran_FFT_v4.csv"
BATHY_FILE = path("bathymetry")
OUT_DIR = f"{BASE}/Results/bathy_relations"

# bathymetry file: None = auto-detect
BATHY_VAR = None                # e.g. "elevation"
BATHY_LON, BATHY_LAT = None, None
BATHY_POSITIVE_DOWN = None      # True if the variable is depth (>0 at sea); None = auto
SMOOTH_KM = 2.5                 # Gaussian sigma for the gradient (≈ half a 5 km box)
SENS_SMOOTH_KM = [1.5, 2.5, 5.0]  # sensitivity of the ray prediction to smoothing
SENS_T0 = True                  # also vary T0 by ± its CI half-width
ISOBATHS = [10, 20, 30, 40, 50, 60, 80]

# transects (ordered deep -> shallow is not required, sorted by depth internally)
ZONES_SR = {                    # shoaling + refraction
    "Z1": ["B0590", "B0591", "B0592", "B0593"],
    "Z3": ["B0355", "B0354", "B0353", "B0352", "B0351"],
}
# All four Z5 boxes are downwave of the islands for the ambient swell.
Z5_UNSHADOWED = ["B0328", "B0369"]     # 49.49°N row, outside the geometric shadow of Sark
Z5_SHADOW = ["B0327", "B0368"]         # 49.45°N row, inside the geometric shadow of Sark
# Energy reference: median over these boxes (open sea N and S of the shadow).
# B0286 alone is not recommended: its SSH std (1.68) is ~10x its neighbours.
Z5_ENERGY_REF = ["B0329", "B0370", "B0325", "B0366"]
Z5_MAP_EXTENT = (-2.47, -2.09, 49.39, 49.555)   # lon0, lon1, lat0, lat1

# Lobe assignment from physical interpretation, used where the Re C lobe is
# unresolved: the direction (from, °N) is set to whichever of θ or θ+180 is
# closer to the value given. Drawn with a dashed arrow.
LOBE_OVERRIDE = {"B0327": 237.0, "B0368": 248.0}

T0_MODE = "deepest"             # "deepest": T0 from λ,h of the deepest box; "median"
AMBIENT_FROM_DEG = 279.0        # open-sea swell direction north of the Sark shadow
RAY_MAX_KM = 45.0               # length of the up-wave ray searched for land
OBS_BACKRAY_KM = 30.0           # length of the back-ray along the observed direction
LABEL_KM = 2.6                  # box label offset from the box centre (km)
RAY_STEP_M = 100.0              # ray-tracing step
RAY_LEN_KM = 40.0               # length of the Z1/Z3 rays from the deepest box
FAN_START_KM = 35.0             # Z5 fan launched this far up-wave of the boxes
FAN_HALF_KM = 25.0              # half-width of the fan
FAN_SPACING_KM = 0.25           # ray spacing
BOX_KM = 5.0                    # analysis box size (for the undisturbed ray count)
H_MIN_RAY = 1.0                 # rays stop in water shallower than this (m)

# boxes shown with open symbols (flags from the pipeline)
FLAG_REASONS = ("gap_leakage", "inner_swath")

g = 9.81
ZONE_COL = {"Z1": "tab:blue", "Z3": "tab:red", "Z5": "tab:purple"}
DPI = 250
SHOW_CAPTIONS = False           # text under the figures (moved to the paper captions)
from swothr.utils import apply_paper_style
apply_paper_style()


# =============================================================================
# LINEAR WAVE THEORY
# =============================================================================
apply_overrides(globals(), "bathy", final=True)


def k_from_T(T, h):
    """Wavenumber from period and depth (vectorised Newton)."""
    T, h = np.broadcast_arrays(np.asarray(T, float), np.asarray(h, float))
    w2 = (2 * np.pi / T) ** 2
    k = w2 / g / np.sqrt(np.tanh(w2 * h / g))            # Eckart start
    for _ in range(30):
        th = np.tanh(k * h)
        f = g * k * th - w2
        df = g * th + g * k * h * (1 - th ** 2)
        k = k - f / df
    return k


def T_from_lambda(lam, h):
    k = 2 * np.pi / lam
    return 2 * np.pi / np.sqrt(g * k * np.tanh(k * h))


def phase_speed(T, h):
    return (2 * np.pi / T) / k_from_T(T, h)


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# =============================================================================
# DATA
# =============================================================================
def box_id(x):
    s = str(x).strip().upper()
    return s if s.startswith("B") else f"B{int(s):04d}"


def load_boxes():
    df = pd.read_csv(FFT_CSV)
    df["Box"] = df["Box"].astype(str).str.strip()
    df["h"] = df["Depth_median_m"].abs()
    df["theta_from"] = df["theta_obs_degN"].astype(float)
    df["assigned"] = False
    for b, want in LOBE_OVERRIDE.items():
        m = df["Box"] == box_id(b)
        if m.any():
            th = float(df.loc[m, "theta_from"].iloc[0])
            alt = (th + 180.0) % 360.0
            df.loc[m, "theta_from"] = th if abs(wrap180(th - want)) <= abs(wrap180(alt - want)) else alt
            df.loc[m, "assigned"] = True
    df["theta_to"] = (df["theta_from"] + 180.0) % 360.0
    reason = df.get("observability_reason", pd.Series("", index=df.index)).astype(str)
    df["flagged"] = reason.apply(lambda r: any(f in r for f in FLAG_REASONS))
    return df


def pick(df, ids):
    ids = [box_id(i) for i in ids]
    sub = df[df["Box"].isin(ids)].copy()
    miss = sorted(set(ids) - set(sub["Box"]))
    if miss:
        print(f"  ! not in CSV: {miss}")
    return sub.sort_values("h", ascending=False).reset_index(drop=True)


def load_bathy(bbox):
    """Depth (m, >0 at sea) on a lon/lat grid inside bbox = (lon0, lon1, lat0, lat1)."""
    ds = xr.open_dataset(BATHY_FILE)
    lonn = BATHY_LON or next(n for n in ("lon", "longitude", "x", "LON") if n in ds.coords or n in ds.dims)
    latn = BATHY_LAT or next(n for n in ("lat", "latitude", "y", "LAT") if n in ds.coords or n in ds.dims)
    var = BATHY_VAR or next((v for v in ("elevation", "depth", "Band1", "z", "DEPTH", "bathymetry")
                             if v in ds.data_vars), list(ds.data_vars)[0])
    lo0, lo1, la0, la1 = bbox
    lat_vals = ds[latn].values
    lat_sl = slice(la0, la1) if lat_vals[0] < lat_vals[-1] else slice(la1, la0)
    da = ds[var].sel({lonn: slice(lo0, lo1), latn: lat_sl})
    z = np.asarray(da.values, float).squeeze()
    lon = np.asarray(da[lonn].values, float)
    lat = np.asarray(da[latn].values, float)
    if da.dims.index(latn) > da.dims.index(lonn):
        z = z.T
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    pos_down = BATHY_POSITIVE_DOWN
    if pos_down is None:
        pos_down = np.nanmedian(z) > 0            # sea is the majority of the tile
    depth = z if pos_down else -z
    land = ~np.isfinite(depth) | (depth <= 0)
    print(f"  bathymetry: '{var}' {depth.shape}, "
          f"{'depth' if pos_down else 'elevation'} convention, land cells {land.mean():.1%}")
    return lon, lat, depth, land


class Bathy:
    """Smoothed depth, gradient and land mask with point interpolation."""
    def __init__(self, lon, lat, depth, land, smooth_km=SMOOTH_KM):
        self.smooth_km = smooth_km
        self.lon, self.lat, self.land = lon, lat, land
        self.depth = np.where(land, np.nan, depth)
        lat0 = np.deg2rad(np.mean(lat))
        dy = np.mean(np.diff(lat)) * 110570.0
        dx = np.mean(np.diff(lon)) * 111320.0 * np.cos(lat0)
        D = np.where(land, 0.0, depth)                       # coast = 0 m
        Ds = gaussian_filter(D, sigma=(smooth_km * 1e3 / dy, smooth_km * 1e3 / dx))
        gy, gx = np.gradient(Ds, dy, dx)                      # d(depth)/d(north, east)
        self.Ds = Ds
        self._dx, self._dy = dx, dy
        kw = dict(bounds_error=False, fill_value=np.nan)
        self._gx = RegularGridInterpolator((lat, lon), gx, **kw)
        self._gy = RegularGridInterpolator((lat, lon), gy, **kw)
        self._land = RegularGridInterpolator((lat, lon), land.astype(float),
                                             method="nearest", **kw)
        dland = distance_transform_edt(~land, sampling=(dy, dx)) / 1e3   # km to land
        self._dland = RegularGridInterpolator((lat, lon), dland, **kw)

    def set_period(self, T0):
        """Phase-speed field c(x, y) for period T0 and its gradient."""
        if getattr(self, "_T0", None) == T0:
            return
        hs = np.maximum(self.Ds, H_MIN_RAY)
        C = phase_speed(T0, hs)
        cy, cx = np.gradient(C, self._dy, self._dx)
        kw = dict(bounds_error=False, fill_value=np.nan)
        self._c = RegularGridInterpolator((self.lat, self.lon), C, **kw)
        self._cx = RegularGridInterpolator((self.lat, self.lon), cx, **kw)
        self._cy = RegularGridInterpolator((self.lat, self.lon), cy, **kw)
        self._hs = RegularGridInterpolator((self.lat, self.lon), self.Ds, **kw)
        self._T0 = T0

    def trace(self, lon0, lat0, theta0, T0, max_km, step=RAY_STEP_M):
        """Refraction rays (vectorised over rays). theta = propagation bearing °N.
        Returns lon, lat, theta arrays (n_steps+1, n_rays), NaN after a ray stops."""
        self.set_period(T0)
        lon = np.atleast_1d(np.asarray(lon0, float)).copy()
        lat = np.atleast_1d(np.asarray(lat0, float)).copy()
        th = np.deg2rad(np.atleast_1d(np.asarray(theta0, float))).copy()
        n = int(max_km * 1e3 / step)
        L, A, T = [lon.copy()], [lat.copy()], [np.degrees(th) % 360.0]
        alive = np.ones(lon.size, bool)
        mlat = 1.0 / 110570.0

        def rhs(lo, la, t):
            p = np.c_[la, lo]
            c, cx, cy = self._c(p), self._cx(p), self._cy(p)
            dth = -(cx * np.cos(t) - cy * np.sin(t)) / c
            mlon = 1.0 / (111320.0 * np.cos(np.deg2rad(la)))
            return np.sin(t) * mlon, np.cos(t) * mlat, dth

        for _ in range(n):
            a1, b1, c1 = rhs(lon, lat, th)
            lo2, la2, t2 = lon + 0.5 * step * a1, lat + 0.5 * step * b1, th + 0.5 * step * c1
            a2, b2, c2 = rhs(lo2, la2, t2)
            lon = lon + step * a2
            lat = lat + step * b2
            th = th + step * c2
            h = self._hs(np.c_[lat, lon])
            alive &= np.isfinite(h) & (h > H_MIN_RAY) & np.isfinite(th)
            L.append(np.where(alive, lon, np.nan))
            A.append(np.where(alive, lat, np.nan))
            T.append(np.where(alive, np.degrees(th) % 360.0, np.nan))
            if not alive.any():
                break
        return np.array(L), np.array(A), np.array(T)

    def upslope(self, lon, lat):
        """Bearing (°N) of -∇h (towards shallower water) and |∇h|."""
        p = np.c_[np.atleast_1d(lat), np.atleast_1d(lon)]
        gx, gy = self._gx(p), self._gy(p)
        return np.degrees(np.arctan2(-gx, -gy)) % 360.0, np.hypot(gx, gy)

    def back_ray(self, lon, lat, bearing, max_km=RAY_MAX_KM, step_m=200.0):
        """Straight line from a box along `bearing` (the 'from' direction).
        Returns first-land distance (km, nan if none), closest approach to land
        (km) and where along the line it occurs (km), and the path."""
        x, lo, la = self.first_land(lon, lat, bearing, max_km, step_m)
        n = len(lo) if not np.isfinite(x) else max(int(round(x * 1e3 / step_m)), 1)
        lo, la = lo[:n], la[:n]
        d = self._dland(np.c_[la, lo])
        i = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
        return dict(x_hit=x, d_min=float(d[i]) if d.size else np.nan,
                    s_min=(i + 1) * step_m / 1e3, lon=lo, lat=la)

    def first_land(self, lon, lat, bearing, max_km=RAY_MAX_KM, step_m=200.0):
        """Distance (km) to the first land cell along a bearing, or nan."""
        t = np.deg2rad(bearing)
        s = np.arange(step_m, max_km * 1e3, step_m)
        la = lat + s * np.cos(t) / 110570.0
        lo = lon + s * np.sin(t) / (111320.0 * np.cos(np.deg2rad(lat)))
        hit = self._land(np.c_[la, lo]) > 0.5
        return (s[np.argmax(hit)] / 1e3, lo, la) if hit.any() else (np.nan, lo, la)


# =============================================================================
# ZONE CALCULATIONS
# =============================================================================
def transect_T0(z):
    if T0_MODE == "median":
        T0 = float(np.nanmedian(z["T_intrinsic_from_lambda_obs_s"]))
    else:
        d = z.iloc[0]                                          # deepest (sorted)
        T0 = float(T_from_lambda(d["lambda_obs_m"], d["h"]))
    lo = z.iloc[0].get("T_intrinsic_from_lambda_obs_CI_low_s", np.nan)
    hi = z.iloc[0].get("T_intrinsic_from_lambda_obs_CI_high_s", np.nan)
    dT = 0.5 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else 0.0
    return T0, dT


def launch_fan(B, lon_c, lat_c, from_deg, T0, start_km=FAN_START_KM,
               half_km=FAN_HALF_KM, spacing_km=FAN_SPACING_KM, extra_km=15.0):
    """Parallel rays launched on a line start_km up-wave of (lon_c, lat_c),
    all with propagation direction from_deg + 180."""
    kx = 1.0 / (111.32 * np.cos(np.deg2rad(lat_c)))
    ky = 1.0 / 110.57
    ta = np.deg2rad(from_deg)
    c_lon = lon_c + start_km * np.sin(ta) * kx
    c_lat = lat_c + start_km * np.cos(ta) * ky
    off = np.arange(-half_km, half_km + 1e-9, spacing_km)
    f_lon = c_lon + off * np.sin(ta + np.pi / 2) * kx
    f_lat = c_lat + off * np.cos(ta + np.pi / 2) * ky
    return B.trace(f_lon, f_lat, np.full(off.size, (from_deg + 180.0) % 360.0),
                   T0, start_km + extra_km)


def rays_in_box(fan, lon, lat, half_km=2.5):
    """Number of rays crossing a 5 km box and their mean direction inside it."""
    FL, FA, FT = fan
    kx = 1.0 / (111.32 * np.cos(np.deg2rad(lat)))
    ky = 1.0 / 110.57
    inx = (np.abs((FL - lon) / kx) <= half_km) & (np.abs((FA - lat) / ky) <= half_km)
    hit = inx.any(axis=0)
    if not hit.any():
        return 0, np.nan
    th = np.deg2rad(FT[inx])
    return int(hit.sum()), float(np.degrees(np.arctan2(np.nanmean(np.sin(th)),
                                                       np.nanmean(np.cos(th)))) % 360.0)


def rays_undisturbed(to_deg, box_km=BOX_KM, spacing_km=FAN_SPACING_KM):
    """Number of straight parallel rays (spacing_km apart, propagation to_deg)
    crossing a square box of side box_km: projected box width / ray spacing.
    Reference for the refraction-only ray density with no bathymetry or land."""
    t = np.deg2rad(to_deg)
    return box_km * (abs(np.sin(t)) + abs(np.cos(t))) / spacing_km


def add_ray(z, B, T0, verbose=True):
    """Refraction prediction for a transect.
    A fan of parallel rays is launched FAN_START_KM up-wave; its offshore direction
    is calibrated so that the rays crossing the deepest (reference) box arrive with
    the observed direction there. The prediction at every other box is the mean
    direction of the rays crossing it."""
    z = z.copy()
    up, grad = B.upslope(z["lon_center"].to_numpy(), z["lat_center"].to_numpy())
    z["upslope_deg"], z["grad_h"] = up, grad
    lon_c, lat_c = z["lon_center"].mean(), z["lat_center"].mean()
    r0 = z.iloc[0]
    obs0 = r0["theta_to"]

    def miss(from_deg):
        fan = launch_fan(B, lon_c, lat_c, from_deg, T0)
        n, th = rays_in_box(fan, r0["lon_center"], r0["lat_center"])
        return (wrap180(th - obs0) if n else np.nan), fan

    base = (obs0 + 180.0) % 360.0                       # 'from' at the reference box
    best = None
    for step, span in ((5.0, 30.0), (1.0, 5.0)):
        centre = base if best is None else best[0]
        for d in np.arange(-span, span + 1e-9, step):
            e, fan = miss((centre + d) % 360.0)
            if np.isfinite(e) and (best is None or abs(e) < abs(best[1])):
                best = ((centre + d) % 360.0, e, fan)
    if best is None:
        raise RuntimeError("no ray reaches the reference box: widen FAN_HALF_KM")
    from_deg, err, fan = best
    pred, nray = [], []
    for _, r in z.iterrows():
        n, th = rays_in_box(fan, r["lon_center"], r["lat_center"])
        pred.append(th); nray.append(n)
    z["theta_to_ray"], z["n_rays"] = pred, nray
    if verbose:
        print(f"    offshore launch direction {from_deg:.0f}° (from), "
              f"mismatch at the reference box {err:+.1f}°")
    return z, fan, from_deg


def refraction_stats(z):
    """rms of ray prediction vs a no-refraction null (direction held at the
    reference box), over the unflagged non-reference boxes."""
    t = z.iloc[1:]
    t = t[~t["flagged"] & np.isfinite(t["theta_to_ray"])]
    if t.empty:
        return dict(rms_ray=np.nan, rms_null=np.nan, skill=np.nan, n=0)
    ref = z["theta_to"].iloc[0]
    rms_ray = float(np.sqrt(np.mean(wrap180(t["theta_to"] - t["theta_to_ray"]) ** 2)))
    rms_null = float(np.sqrt(np.mean(wrap180(t["theta_to"] - ref) ** 2)))
    skill = 1.0 - rms_ray ** 2 / rms_null ** 2 if rms_null > 0 else np.nan
    return dict(rms_ray=rms_ray, rms_null=rms_null, skill=skill, n=len(t))


def rotation(z, col):
    zz = z[np.isfinite(z[col])]
    return float(wrap180(zz[col].iloc[-1] - zz[col].iloc[0])) if len(zz) > 1 else np.nan


def sensitivity(zones, raw_bathy):
    """Re-run the calibrated ray prediction for every smoothing in SENS_SMOOTH_KM
    and T0 ± CI. Returns {zone: DataFrame(sigma, T0, box..., rotation)}."""
    out = {}
    cache = {}
    for sig in SENS_SMOOTH_KM:
        cache[sig] = Bathy(*raw_bathy, smooth_km=sig)
    for name, (z, T0, dT, _, _) in zones.items():
        rows = []
        Ts = [T0 - dT, T0, T0 + dT] if (SENS_T0 and dT > 0) else [T0]
        for sig, Bs in cache.items():
            for T in Ts:
                try:
                    zs, _, _ = add_ray(z, Bs, T, verbose=False)
                except RuntimeError:
                    continue
                row = {"sigma": sig, "T0": T, "rotation": rotation(zs, "theta_to_ray")}
                row.update({b: th for b, th in zip(zs["Box"], zs["theta_to_ray"])})
                row.update({f"stat_{k}": v for k, v in refraction_stats(zs).items()})
                rows.append(row)
        out[name] = pd.DataFrame(rows)
        d = out[name]
        print(f"  {name} sensitivity: ray rotation {d['rotation'].min():+.0f}° to "
              f"{d['rotation'].max():+.0f}° (obs {rotation(z, 'theta_to'):+.0f}°), "
              f"skill {d['stat_skill'].min():.2f}–{d['stat_skill'].max():.2f}")
    return out


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def mfc(row, col):
    return "white" if row["flagged"] else col


def draw_map(ax, B, sub, col, pad_km=6.0, arrows_up=True, lam_scale_km=4.0, extent=None,
             ray=None):
    lat0 = np.deg2rad(sub["lat_center"].mean())
    px, py = pad_km / (111.32 * np.cos(lat0)), pad_km / 110.57
    x0, x1 = sub["lon_center"].min() - px, sub["lon_center"].max() + px
    y0, y1 = sub["lat_center"].min() - py, sub["lat_center"].max() + py
    if extent is not None:
        x0, x1 = min(x0, extent[0]), max(x1, extent[1])
        y0, y1 = min(y0, extent[2]), max(y1, extent[3])
    # make the window at least square in km so tall/wide transects keep context
    wkm = (x1 - x0) * 111.32 * np.cos(lat0)
    hkm = (y1 - y0) * 110.57
    if wkm < hkm:
        dx = 0.5 * (hkm - wkm) / (111.32 * np.cos(lat0))
        x0, x1 = x0 - dx, x1 + dx
    elif hkm < 0.6 * wkm:
        dy = 0.5 * (0.6 * wkm - hkm) / 110.57
        y0, y1 = y0 - dy, y1 + dy
    ix = (B.lon >= x0) & (B.lon <= x1)
    iy = (B.lat >= y0) & (B.lat <= y1)
    lon, lat = B.lon[ix], B.lat[iy]
    D = B.depth[np.ix_(iy, ix)]
    land = B.land[np.ix_(iy, ix)]
    ax.pcolormesh(lon, lat, np.where(land, np.nan, D), cmap="Blues", vmin=0,
                  vmax=max(ISOBATHS), shading="auto", alpha=0.55)
    ax.contourf(lon, lat, land.astype(float), levels=[0.5, 1.5], colors=["0.75"])
    cs = ax.contour(lon, lat, B.Ds[np.ix_(iy, ix)], levels=ISOBATHS, colors="0.35",
                    linewidths=0.6)
    ax.clabel(cs, fmt="%d m", fontsize=6)
    kx = 1.0 / (111.32 * np.cos(lat0))
    ky = 1.0 / 110.57
    for _, r in sub.iterrows():
        t = np.deg2rad(r["theta_to"])
        L = lam_scale_km
        ax.annotate("", xy=(r["lon_center"] + 0.5 * L * np.sin(t) * kx,
                            r["lat_center"] + 0.5 * L * np.cos(t) * ky),
                    xytext=(r["lon_center"] - 0.5 * L * np.sin(t) * kx,
                            r["lat_center"] - 0.5 * L * np.cos(t) * ky),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                    linestyle="--" if r["assigned"] else "-",
                                    mutation_scale=13))
        ax.plot(r["lon_center"], r["lat_center"], "s", ms=5, mfc=mfc(r, col), mec=col)
        tl = t - np.pi / 2                       # 90° to the left of the wave arrow
        ax.text(r["lon_center"] + LABEL_KM * np.sin(tl) * kx,
                r["lat_center"] + LABEL_KM * np.cos(tl) * ky,
                f"{r['Box'][1:]}\n{r['h']:.0f} m", fontsize=6, ha="center", va="center",
                bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.3))
        if arrows_up and "upslope_deg" in r and np.isfinite(r["upslope_deg"]):
            u = np.deg2rad(r["upslope_deg"])
            ax.annotate("", xy=(r["lon_center"] + 1.6 * np.sin(u) * kx,
                                r["lat_center"] + 1.6 * np.cos(u) * ky),
                        xytext=(r["lon_center"], r["lat_center"]),
                        arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.0,
                                        mutation_scale=8))
    if ray is not None:
        ax.plot(ray[0], ray[1], color="tab:cyan", lw=0.35, alpha=0.6, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(lat0))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=7)


# =============================================================================
# FIGURE 1 -- SHOALING
# =============================================================================
def fig_shoaling(zones):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    hh = np.linspace(5, 60, 200)
    kh = np.linspace(0.05, 3.2, 300)
    for name, (z, T0, dT, _, _) in zones.items():
        col = ZONE_COL[name]
        # (a) dimensional
        lam_th = 2 * np.pi / k_from_T(T0, hh)
        ax[0].plot(hh, lam_th, color=col, lw=1.4, label=f"{name}: dispersion, T0 = {T0:.1f} s")
        if dT > 0:
            ax[0].fill_between(hh, 2 * np.pi / k_from_T(T0 - dT, hh),
                               2 * np.pi / k_from_T(T0 + dT, hh), color=col, alpha=0.12)
        for _, r in z.iterrows():
            ax[0].errorbar(r["h"], r["lambda_obs_m"],
                           yerr=r.get("lambda_CI_half_m", np.nan),
                           xerr=r.get("Depth_std_m", np.nan), fmt="o", color=col,
                           mfc=mfc(r, col), ms=6, capsize=2, lw=0.8)
        # (b) normalised collapse
        L0 = g * T0 ** 2 / (2 * np.pi)
        for _, r in z.iterrows():
            ax[1].errorbar(r["h"] / L0, r["lambda_obs_m"] / L0,
                           yerr=r.get("lambda_CI_half_m", np.nan) / L0, fmt="o",
                           color=col, mfc=mfc(r, col), ms=6, capsize=2, lw=0.8)
        # (c) period conservation
        ax[2].axhline(T0, color=col, lw=1.0, ls=":")
        for _, r in z.iterrows():
            ax[2].plot(r["h"], r["T_intrinsic_from_lambda_obs_s"], "o", color=col,
                       mfc=mfc(r, col), ms=6)
            if np.isfinite(r.get("T_absolute_from_lambda_obs_current_s", np.nan)):
                ax[2].plot(r["h"], r["T_absolute_from_lambda_obs_current_s"], "^",
                           color=col, mfc="none", ms=6)
    # universal curve λ/L0 = tanh(kh) with h/L0 = kh tanh(kh) / (2π)
    x = kh * np.tanh(kh) / (2 * np.pi)
    ax[1].plot(x, np.tanh(kh), "k-", lw=1.4, label=r"$\lambda/L_0=\tanh(kh)$")
    xs_all, ys_all = [], []
    for name, (z, T0, dT, _, _) in zones.items():
        L0 = g * T0 ** 2 / (2 * np.pi)
        xs_all += list(z["h"] / L0)
        ys_all += list(z["lambda_obs_m"] / L0)
    ax[1].set_xlim(0, max(xs_all) * 1.4)
    ax[1].set_ylim(min(ys_all) * 0.85, min(1.02, max(ys_all) * 1.15))

    ax[0].set_xlabel("depth h (m)")
    ax[0].set_ylabel("wavelength λ (m)")
    ax[0].set_title("(a) λ vs depth, linear dispersion at T0")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$h/L_0$")
    ax[1].set_ylabel(r"$\lambda/L_0$")
    ax[1].set_title("(b) normalised: both transects on one curve")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[2].set_xlabel("depth h (m)")
    ax[2].set_ylabel("period (s)")
    ax[2].set_title("(c) period conservation")
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="T intrinsic"),
                          Line2D([], [], marker="^", ls="", color="k", mfc="none",
                                 label="T current-corrected"),
                          Line2D([], [], ls=":", color="k", label="T0")], fontsize=7)
    for a in ax:
        a.grid(lw=0.3, alpha=0.5)
    if SHOW_CAPTIONS:
        fig.text(0.5, -0.02, "Open symbols: boxes flagged by the pipeline "
             f"({', '.join(FLAG_REASONS)}). Shading: T0 ± CI half-width of the reference box.",
             ha="center", fontsize=7)
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 2 -- REFRACTION
# =============================================================================
def fig_refraction(zones, B, sens):
    names = list(zones)
    fig = plt.figure(figsize=(12, 10.5))
    gs = GridSpec(2, max(len(names), 2), figure=fig, height_ratios=[1.15, 1],
                  hspace=0.3, wspace=0.3)
    for j, name in enumerate(names):
        z, T0, _, fan, from_deg = zones[name]
        a = fig.add_subplot(gs[0, j])
        draw_map(a, B, z, ZONE_COL[name], ray=fan)
        a.set_title(f"({'ab'[j]}) {name}: waves, upslope −∇h (black), rays (cyan)")

    def envelope(name, box):
        d = sens.get(name)
        if d is None or box not in d:
            return np.nan, np.nan
        v = d[box].to_numpy(float)
        v = v[np.isfinite(v)]
        if not v.size:
            return np.nan, np.nan
        ref = v[0]
        v = ref + wrap180(v - ref)
        return v.min(), v.max()

    # (c) direction vs depth
    ax = fig.add_subplot(gs[1, 0])
    for name in names:
        z, T0, _, _, from_deg = zones[name]
        col = ZONE_COL[name]
        for _, r in z.iterrows():
            lo, hi = envelope(name, r["Box"])
            if np.isfinite(lo):
                ax.fill_between([r["h"] - 0.35, r["h"] + 0.35], lo, hi, color=col,
                                alpha=0.18, lw=0)
        ax.plot(z["h"], z["theta_to_ray"], "-", color=col, lw=1.4, alpha=0.8,
                label=f"{name}: rays (offshore {from_deg:.0f}°, T0 = {T0:.1f} s)")
        ax.plot(z["h"], z["theta_to_ray"], "D", color=col, mfc="w", ms=5)
        base = z["theta_to"].iloc[0]
        ax.plot([z["h"].max(), z["h"].min()], [base, base], ":", color=col, lw=1.0)
        for _, r in z.iterrows():
            ax.errorbar(r["h"], r["theta_to"], yerr=r["theta_CI_half_deg"], fmt="o",
                        color=col, mfc=mfc(r, col), ms=6, capsize=2, lw=0.8)
            ax.annotate(r["Box"][1:], (r["h"], r["theta_to"]), fontsize=6,
                        xytext=(4, 3), textcoords="offset points")
    ax.plot([], [], ":", color="k", label="no refraction (null)")
    ax.fill_between([], [], [], color="0.6", alpha=0.3, label="rays: σ and T0 range")
    ax.invert_xaxis()
    ax.set_xlabel("depth h (m)  →  shallower")
    ax.set_ylabel("propagation direction (°N)")
    ax.set_title("(c) direction vs depth: observed (●), rays (◇)")
    ax.legend(fontsize=6.5)
    ax.grid(lw=0.3, alpha=0.5)

    # (d) 1:1 with sensitivity range as horizontal bars
    ax = fig.add_subplot(gs[1, 1])
    allv = []
    for name in names:
        z = zones[name][0]
        col = ZONE_COL[name]
        zz = z[np.isfinite(z["theta_to_ray"])]
        for _, r in zz.iterrows():
            lo, hi = envelope(name, r["Box"])
            xerr = ([[max(r["theta_to_ray"] - lo, 0)], [max(hi - r["theta_to_ray"], 0)]]
                    if np.isfinite(lo) else None)
            ax.errorbar(r["theta_to_ray"], r["theta_to"], yerr=r["theta_CI_half_deg"],
                        xerr=xerr, fmt="o", color=col, mfc=mfc(r, col), ms=6, capsize=2, lw=0.8)
            ax.annotate(r["Box"][1:], (r["theta_to_ray"], r["theta_to"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        allv += list(zz["theta_to"]) + list(zz["theta_to_ray"])
        st = refraction_stats(z)
        d = sens.get(name)
        rr = (f" [{d['rotation'].min():+.0f}…{d['rotation'].max():+.0f}°]"
              if d is not None and len(d) else "")
        ax.plot([], [], "o", color=col,
                label=(f"{name}: rotation obs {rotation(z, 'theta_to'):+.0f}°, "
                       f"rays {rotation(z, 'theta_to_ray'):+.0f}°{rr}\n"
                       f"      rms rays {st['rms_ray']:.1f}° vs null {st['rms_null']:.1f}°, "
                       f"skill {st['skill']:.2f}"))
    lo, hi = min(allv) - 8, max(allv) + 8
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ray-predicted propagation direction (°N)")
    ax.set_ylabel("observed propagation direction (°N)")
    ax.set_title("(d) observed vs refraction theory")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(lw=0.3, alpha=0.5)
    cap = (f"Rays: linear-dispersion phase speed at T0 on the bathymetry smoothed with σ = "
           f"{SMOOTH_KM} km; parallel rays launched {FAN_START_KM:.0f} km up-wave, offshore "
           "direction calibrated on the deepest box only; prediction = mean direction of the rays "
           "crossing each 5 km box. Shading / horizontal bars: range over σ = "
           f"{', '.join(f'{v:g}' for v in SENS_SMOOTH_KM)} km"
           + (" and T0 ± CI" if SENS_T0 else "") + ". Null: direction held at the reference box. "
           "skill = 1 − rms²(rays)/rms²(null), unflagged non-reference boxes. Error bars: θ CI "
           "half-width; open symbols: flagged boxes.")
    if SHOW_CAPTIONS:
        fig.text(0.5, 0.0, "\n".join(textwrap.wrap(cap, 160)), ha="center", fontsize=7)
    return fig


def fig_refraction_sensitivity(zones, sens):
    names = [n for n in zones if n in sens and len(sens[n])]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        z, T0, dT, _, _ = zones[name]
        d = sens[name]
        col = ZONE_COL[name]
        obs = rotation(z, "theta_to")
        ci = np.hypot(z["theta_CI_half_deg"].iloc[0], z["theta_CI_half_deg"].iloc[-1])
        ax.axhspan(obs - ci, obs + ci, color=col, alpha=0.15)
        ax.axhline(obs, color=col, lw=1.5, label=f"observed {obs:+.0f}° ± {ci:.0f}°")
        ax.axhline(0, color="0.4", ls=":", lw=1.0, label="no refraction")
        for T, g_ in d.groupby("T0"):
            ls = "-" if abs(T - T0) < 1e-6 else "--"
            ax.plot(g_["sigma"], g_["rotation"], ls, marker="o", color="k",
                    alpha=1.0 if ls == "-" else 0.55, label=f"rays, T0 = {T:.1f} s")
        ax.set_xlabel("bathymetry smoothing σ (km)")
        ax.set_ylabel(f"rotation {z['Box'].iloc[0][1:]} → {z['Box'].iloc[-1][1:]} (deg)")
        ax.set_title(f"{name}: sensitivity of the refraction prediction")
        ax.legend(fontsize=7)
        ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 3 -- DIFFRACTION (Z5)
# =============================================================================
def fig_diffraction(df, B):
    uns_ids = [box_id(b) for b in Z5_UNSHADOWED]
    sh_ids = [box_id(b) for b in Z5_SHADOW]
    z = pick(df, uns_ids + sh_ids)
    z["role"] = np.where(z["Box"].isin(sh_ids), "shadow", "unshadowed")
    order = {b: i for i, b in enumerate(uns_ids + sh_ids)}
    z = z.assign(_o=z["Box"].map(order)).sort_values("_o").reset_index(drop=True)
    uns = z[z["role"] == "unshadowed"]

    # refraction-only rays for the ambient swell
    T0 = float(np.median([T_from_lambda(r["lambda_obs_m"], r["h"]) for _, r in uns.iterrows()]))
    lon_m, lat_m = z["lon_center"].mean(), z["lat_center"].mean()
    fan = launch_fan(B, lon_m, lat_m, AMBIENT_FROM_DEG, T0, extra_km=12.0)
    FL, FA, FT = fan
    n_rays = np.array([rays_in_box(fan, r["lon_center"], r["lat_center"])[0]
                       for _, r in z.iterrows()], float)
    # reference: the same fan with no bathymetry and no land (straight parallel rays)
    n_free = rays_undisturbed((AMBIENT_FROM_DEG + 180.0) % 360.0)
    z["n_rays"] = n_rays
    z["ray_rel"] = n_rays / n_free

    # observed energy relative to the open-sea reference (median)
    ref = df[df["Box"].isin([box_id(b) for b in Z5_ENERGY_REF])]
    e_ref = float(np.nanmedian(ref["p1_energy"])) if len(ref) else np.nan
    s_ref = float(np.nanmedian(ref["ssh_std_short"])) if len(ref) else np.nan
    z["e_rel"] = z["p1_energy"] / e_ref
    z["s_rel"] = z["ssh_std_short"] / s_ref
    z["p2p1"] = np.where(z["p1_energy"] > 0, z["p2_energy"] / z["p1_energy"], np.nan)

    print(f"  Z5: T0 = {T0:.1f} s, undisturbed rays per box = {n_free:.1f}, "
          f"energy reference = median of "
          f"{', '.join(ref['Box'])} (p1 {e_ref:.3f}, SSH std {s_ref:.3f})")
    print(z[["Box", "role", "h", "theta_from", "assigned", "n_rays", "ray_rel",
             "ang_spread_deg", "n_partitions", "p2p1", "coherence_peak",
             "e_rel", "s_rel"]].round(2).to_string(index=False))

    col_role = {"unshadowed": ["0.55", "0.75"], "shadow": [ZONE_COL["Z5"], "#c3a6dd"]}
    cols, k = [], {"unshadowed": 0, "shadow": 0}
    for rl in z["role"]:
        cols.append(col_role[rl][k[rl] % 2])
        k[rl] += 1
    labels = [f"{b[1:]}\n{'shadow' if rl == 'shadow' else 'no shadow'}"
              for b, rl in zip(z["Box"], z["role"])]
    xs = np.arange(len(z))

    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1, 1], hspace=0.45, wspace=0.35)

    # (a) map
    ax = fig.add_subplot(gs[:, 0])
    x0, x1, y0, y1 = Z5_MAP_EXTENT
    draw_map(ax, B, z, ZONE_COL["Z5"], pad_km=0.0, arrows_up=False, extent=Z5_MAP_EXTENT)
    ax.plot(FL, FA, color="tab:cyan", lw=0.35, alpha=0.6, zorder=1)
    grid = df[df["lon_center"].between(x0, x1) & df["lat_center"].between(y0, y1)
              & ~df["Box"].isin(z["Box"])]
    ax.plot(grid["lon_center"], grid["lat_center"], "o", mfc="w", mec="k", ms=4, mew=0.8, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(f"(a) Z5: refraction-only rays for ambient swell from {AMBIENT_FROM_DEG:.0f}°")

    # (b) refraction-only ray density
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(xs, z["ray_rel"], color=cols)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ray density / undisturbed")
    ax.set_title("(b) refraction-only geometric shadow")

    # (c) spreading
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(xs, z["ang_spread_deg"], color=cols)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("angular spreading (deg)")
    ax.set_title("(c) directional spreading")

    # (d) multimodality and coherence
    ax = fig.add_subplot(gs[1, 1])
    w = 0.38
    ax.bar(xs - w / 2, np.nan_to_num(z["p2p1"]), w, color=cols, label="p2 / p1 energy")
    ax.bar(xs + w / 2, z["coherence_peak"], w, color=cols, hatch="//", edgecolor="w",
           label="coherence")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b[1:]}\n{int(n)} part." for b, n in zip(z["Box"], z["n_partitions"])],
                       fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("(d) multimodality / coherence")
    ax.legend(fontsize=7, loc="upper left")

    # (e) observed energy relative to open sea
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(xs - w / 2, z["e_rel"], w, color=cols, label="p1 energy")
    ax.bar(xs + w / 2, z["s_rel"], w, color=cols, hatch="..", edgecolor="w", label="SSH std")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("box / open-sea median")
    ax.set_title("(e) energy relative to open sea")
    ax.legend(fontsize=7, loc="upper right")
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_boxes()

    all_ids = [i for v in ZONES_SR.values() for i in v] + list(Z5_UNSHADOWED) + list(Z5_SHADOW)
    sub = df[df["Box"].isin([box_id(i) for i in all_ids])]
    bbox = (sub["lon_center"].min() - 0.8, sub["lon_center"].max() + 0.3,
            sub["lat_center"].min() - 0.3, sub["lat_center"].max() + 0.3)
    print("loading bathymetry")
    raw = load_bathy(bbox)
    B = Bathy(*raw)

    zones = {}
    for name, ids in ZONES_SR.items():
        z = pick(df, ids)
        T0, dT = transect_T0(z)
        z, fan, from_deg = add_ray(z, B, T0)
        zones[name] = (z, T0, dT, fan, from_deg)
        print(f"  {name}: T0 = {T0:.2f} s (±{dT:.2f}), "
              f"reference {z['Box'].iloc[0]} at {z['h'].iloc[0]:.0f} m")
        cols = ["Box", "h", "lambda_obs_m", "theta_to", "theta_to_ray", "n_rays",
                "upslope_deg", "flagged"]
        print(z[cols].round(1).to_string(index=False))

        st = refraction_stats(z)
        print(f"    rms rays {st['rms_ray']:.1f}° vs null {st['rms_null']:.1f}° "
              f"-> skill {st['skill']:.2f} (n = {st['n']})")

    print("sensitivity of the refraction prediction")
    sens = sensitivity(zones, raw)

    for fname, fig in (("fig_shoaling.png", fig_shoaling(zones)),
                       ("fig_refraction.png", fig_refraction(zones, B, sens)),
                       ("fig_refraction_sensitivity.png", fig_refraction_sensitivity(zones, sens)),
                       ("fig_diffraction_Z5.png", fig_diffraction(df, B))):
        out = Path(OUT_DIR) / fname
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
