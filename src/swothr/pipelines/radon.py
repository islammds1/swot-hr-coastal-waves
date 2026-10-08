# -*- coding: utf-8 -*-


import glob
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import hann
from scipy.spatial import cKDTree
from skimage.transform import radon
from pyproj import Proj

from swothr.config import path, apply_overrides

ROOT = path("root")             # config/paths.yaml


# ============================================================
# USER SETTINGS
# ============================================================

SWOT_FILES = rf"{ROOT}/Data/SWOT HR/Mathis/SWOT_L2_HR_PIXC_*.nc"
GROUP = "pixel_cloud"

SWOT_VARS = ["latitude", "longitude", "height", "sig0", "geolocation_qual"]
SWOT_GEOM_VARS = ["cross_track", "illumination_time"]

DATE = "2023-04-085"
CYCLE = 481
DOMAIN_NAME = "SWOT_tile"
UTM_ZONE = 30

# ---- Acquisition identity / time ---------------------------------------------
PASS = 16
STRICT_FILE_IDENTITY = True
SWOT_ACQ_TIME_OVERRIDE = None   # ISO string, only if PIXC time cannot be decoded
ACQ_DATE_TOLERANCE_H = 18.0

# ---- SWOT quality control ---------------------------------------------------
# geolocation_qual is a bit field. For this wave-spectrum application we use a
# two-tier strategy rather than deleting every pixel carrying a "suspect" bit.
#
# HARD REJECTS below are defects directly capable of corrupting the phase/
# geolocation or producing spectral artefacts, while preserving the two flags
# that need special treatment for wave work:
#   0x00000001 layover_significant       -> KEEP + diagnose per box
#   0x00000004 phase_unwrapping_suspect  -> KEEP + diagnose per box
#
# The phase-unwrapping bit is intentionally not hard-rejected because, in the
# present Mathis granules, it is set on ~61 % of the pixels. Removing it at the
# pixel level would create a very large structured sampling mask and can itself
# contaminate the FFT. Its fraction is exported so sensitivity can be assessed.
#
# We do hard-reject phase_noise_suspect because it directly affects height and
# costs almost no data in this scene (~0.02 %), plus KaRIn telemetry/phase/orbit
# suspect flags, small/large KaRIn gaps, specular ringing, refloc geolocation,
# and all geolocation "bad" flags.
SWOT_QUAL_VAR = "geolocation_qual"
SWOT_QUAL_BITMASK = (
    0x00000002 |  # phase_noise_suspect
    0x00000400 |  # suspect_karin_telem
    0x00001000 |  # medium_phase_suspect
    0x00002000 |  # tvp_suspect
    #0x00004000 |  # sc_event_suspect -- NELSON ONLY: fires on ~100% of pixels;
                 #  re-enable for Mathis, Mathis and Mathis
    0x00008000 |  # small_karin_gap
    0x00080000 |  # specular_ringing_degraded
    0x01000000 |  # geolocation_is_from_refloc
    0x08000000 |  # no_geolocation_bad
    0x10000000 |  # medium_phase_bad
    0x20000000 |  # tvp_bad
    0x40000000 |  # sc_event_bad
    0x80000000    # large_karin_gap
)
# = 0xF908F402 for the Version-D flag layout printed by the file metadata.
SWOT_QUAL_MAX = np.inf
REPORT_QUAL_BITS = True
QUAL_BIT_POLICY = "require"      # "require", "warn", or "ignore"

# Soft diagnostic bits: retained but summarized per 5-km box.
QUAL_LAYOVER_BIT = 0x00000001
QUAL_PHASE_UNWRAP_BIT = 0x00000004

# ---- Cross-track range ------------------------------------------------------
# V4 loads the inner swath for diagnostics instead of imposing 25 km as a hard
# physical boundary. Local Nyquist + instrument response determine usability.
APPLY_CROSS_TRACK_FILTER = True
CROSS_TRACK_MIN = 10_000.0       # hard data-loading bound only
CROSS_TRACK_MAX = 60_000.0
CROSS_TRACK_OBS_WARN = 25_000.0

# ---- Orbit constants for the posting model ----------------------------------
H_SAT = 891_000.0
R_EARTH = 6_377_000.0
RANGE_GATE_M = 0.75
ALONG_TRACK_POSTING = 22.0

# ---- Rain screening from L2_LR_SSH Expert -----------------------------------
LR_FILES = rf"{ROOT}/Data/SWOT_LR/SWOT_L2_LR_SSH_Expert_481_016_*.nc"
RAIN_SCREEN = True
RAIN_FLAG_VALUES = (2,)
RAIN_RATE_MIN = None
RAIN_USE_INSTRUMENT_BIT = True
RAIN_BUFFER_M = 2000.0
RAIN_MIN_CELL_PIXELS = 1
RAIN_NADIR_GAP_MIN_CT = 8000.0
RAIN_NO_COVERAGE_ACTION = "flag"
LR_MAX_TIME_DELTA_H = 3.0

# ---- Bathymetry -------------------------------------------------------------
BATHY_FILE = path("bathymetry")
BATHY_LON_VAR = "lon"
BATHY_LAT_VAR = "lat"
BATHY_ELEV_VAR = "elevation"
BATHY_POSITIVE_DOWN = False
FALLBACK_DEPTH_M = 37.7
REQUIRE_BATHY = True
MIN_DEPTH_M = 5.0
T_VALID_RANGE = (4.0, 20.0)
PERIOD_HARD_REJECT = False

# ---- Surface current --------------------------------------------------------
# MARC MARS3D-MANGAE2500 3-D hourly files. A glob is allowed: the file whose
# name time (..._YYYYMMDDTHHMMZ.nc) is closest to the SWOT acquisition is used.
CURRENT_FILE = rf"{ROOT}/Data/Surface current/Surface_current_Mathis.nc"
CUR_CROP_MARGIN_DEG = 0.3        # crop the model to the SWOT tile + margin
# Generalized s-coordinate (Song & Haidvogel, MARS3D):
#   z = eta (1 + s) + hc s + (H0 - hc) C(s)
#   C(s) = (1-b) sinh(a s)/sinh(a) + b [tanh(a(s+1/2)) - tanh(a/2)] / (2 tanh(a/2))
CUR_ETA_VAR, CUR_H0_VAR, CUR_HC_VAR = "XE", "H0", "hc"
CUR_CS_VAR, CUR_THETA_VAR, CUR_B_VAR = "Csu_sig", "theta", "b"
CUR_LON_RHO, CUR_LAT_RHO = "longitude", "latitude"
CUR_U_VAR, CUR_V_VAR = "UZ", "VZ"
CUR_LON_U, CUR_LAT_U = "longitude_u", "latitude_u"
CUR_LON_V, CUR_LAT_V = "longitude_v", "latitude_v"
CUR_TIME_VAR = None              # None -> detect time/time_counter/time_instant
CUR_TIME_INDEX_FALLBACK = None   # no silent index-0 fallback in V4
CUR_MAX_TIME_DELTA_H = 3.0
CURRENT_REQUIRE_TIME_MATCH = True
CUR_SURFACE_LEVEL = None
CUR_SIG_VAR = "SIG"
PROPAGATE_ENV_UNCERTAINTY = True

# ---- MARC/WW3 model: diagnostics AND lobe choice for unresolved boxes -------
# The model never changes SWOT wavelength. It is used (i) as sea-state
# diagnostics and (ii) to pick the 180 deg lobe ONLY where SWOT's own
# velocity-bunching vote failed (MODEL_LOBE_MODE).
WW3_FILE = rf"{ROOT}/Data/Wave Validation/MARC_WW3-NORGAS-2M_Mathis.nc"
WW3_LON_CANDIDATES = ("longitude", "lon", "nav_lon")
WW3_LAT_CANDIDATES = ("latitude", "lat", "nav_lat")
WW3_TIME_CANDIDATES = ("time", "time_counter", "valid_time")
WW3_HS_CANDIDATES = ("hs", "VHM0", "swh")
WW3_FP_CANDIDATES = ("fp", "peak_frequency")
WW3_TP_CANDIDATES = ("VTPK", "tp", "t0m1")      # used only if no fp variable
WW3_DIR_CANDIDATES = ("dp", "VPED", "VMDR", "dir")
# "auto" reads standard_name / long_name / comment. Pin "from" or "towards"
# once confirmed. If "auto" cannot decide, WW3_DIR_CONVENTION_FALLBACK is used
# and model-assisted lobe choice is disabled unless MODEL_LOBE_ALLOW_ASSUMED.
WW3_DIR_CONVENTION = "from"     # confirmed for MARC dp (nautical, coming from)
WW3_DIR_CONVENTION_FALLBACK = "from"
MODEL_LOBE_ALLOW_ASSUMED = False
# WW3 integrates the spectrum on the intrinsic (relative) frequency grid, so
# fp is intrinsic even in a current-forced run. Set True only if confirmed
# otherwise for the MARC product.
WW3_FP_IS_ABSOLUTE = False
WW3_MAX_TIME_DELTA_H = 3.0
WW3_MAX_NEAREST_M = 5000.0       # box with no model node: nearest within this
WW3_HS_WARN = None
WW3_STEEPNESS_WARN = None        # Hs/Lp proxy; calibrate against validation data

# ---- Model-assisted 180 deg choice --------------------------------------------
MODEL_LOBE_MODE = "unresolved"   # "off" | "unresolved"
MODEL_LOBE_MAX_DEV = 60.0        # model must be within this of the wave axis
MODEL_LOBE_DEMERIT = 0           # observability demerit for a model-chosen lobe

# ---- Doppler / current ----------------------------------------------------------
# "kirby_chen": wavenumber-weighted current from the vertical profile;
# "surface": top layer only (v4.2 behaviour).
CURRENT_DEPTH_MODE = "kirby_chen"
UNRESOLVED_TC_POLICY = "nan"     # "nan" | "keep" absolute period when lobe unknown

OUT_DIR = Path(rf"{ROOT}/Results/radon/new")
TAG = "Mathis_RADON_v4_3"

# ---- per-storm overrides (config/storms/*.yaml, see README) ----------
apply_overrides(globals(), ("common", "pipeline", "radon"))
OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / f"swot_5km_RADON_{TAG}.csv"
GEOJSON_OUT = OUT_DIR / f"swot_5km_boxes_{TAG}.geojson"
TILT_DIAG_OUT = OUT_DIR / f"diag_tilt_antisymmetry_{TAG}.png"
FIG_TP_OUT = OUT_DIR / f"map_Tp_{TAG}.png"
FIG_LAMBDA_OUT = OUT_DIR / f"map_lambda_{TAG}.png"
FIG_UNCERT_OUT = OUT_DIR / f"map_dir_uncertainty_{TAG}.png"
FIG_OBS_OUT = OUT_DIR / f"map_observability_{TAG}.png"


# ============================================================
# SPECTRAL SETTINGS
# ============================================================

g = 9.81

# ---- adaptive grid ----------------------------------------------------------
RES_MODE = "adaptive"        # "adaptive" or "fixed"
RES_FIXED = 30.0
RES_ADAPT_FACTOR = 1.5       # cell = factor * max(local along, across posting)
RES_MIN, RES_MAX = 20.0, 80.0
RES_ROUND = 1.0

BOX_SIZE = 5000.0
BOX_STRIDE = 5000.0

LWIN = 1500.0
LWIN_SECONDARY = 2500.0
MULTISCALE = True
MULTISCALE_TRIGGER_LAMBDA = 280.0
MULTISCALE_MIN_WINDOWS = 3
MULTISCALE_MAX_REL_DLAMBDA = 0.20
MULTISCALE_MAX_DTHETA_DEG = 20.0
OVERLAP_FRAC = 0.5
ZPAD_FRAC = 1.0              # centroid gives sub-bin resolution; see note 5

# ---- Radon/Fourier-slice estimator -------------------------------------------
RADON_ANGLE_STEP_DEG = 1.0
RADON_SLICE_ZPAD_FACTOR = 2.0  # centroid removes bin snapping, but circle=False
                               # gives n_rho = sqrt(2)*Nwin and unpadded slices
                               # quantise lambda by ~19 m at 200 m, which leaves
                               # partitions too few radial bins to grow into.

MIN_VALID_FRAC_WINDOW = 0.80
MIN_VALID_FRAC_BOX = 0.60
MIN_RAW_PIXELS_BOX = 350
MIN_N_WINDOWS = 5
MIN_PIXELS_PER_CELL = 1

LAMBDA_MIN = 130.0           # observed lambda is biased LONG, so the true
LAMBDA_MAX = 400.0           # 150-250 m band appears at roughly 150-325 m
LAMBDA_RES_MIN_WAVES = 4.0   # demote when too few wavelengths fit in a Welch window
NYQUIST_SAFETY = 2.5         # local minimum wavelength >= 2.5 grid cells

DIR_CONVENTION = "from"

N_BOOT = 300
RANDOM_SEED = 42
APPLY_OVERLAP_CORR = True
N_EFF_FACTOR = 1.0
MAKE_MAPS = True

# ---- noise model ------------------------------------------------------------
SUBTRACT_NOISE_FLOOR = True
NOISE_LAMBDA_LO_FACTOR = 2.2      # noise band starts at factor * RES

# ---- instrument filter response --------------------------------------------
INCLUDE_GRID_RESPONSE = True      # include the RES binning rect in the response
RESPONSE_MIN_MASK = 0.25          # exclude band bins below this power response
RESPONSE_WARN = 0.50              # demote observability below this
APPLY_FILTER_CORRECTION = False   # deliberately off: flag, do not deconvolve
NOISE_RESP_MIN = 0.05             # noise annulus: skip bins near the boxcar zero
RUN_RESPONSE_SENSITIVITY = True   # re-partition with each mask value below
SENS_RESPONSE_MASKS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

# ---- partitions -------------------------------------------------------------
PARTITION_FRAC = 0.10             # retained as per-seed grow fraction
PARTITION_SEED_FRAC = 0.05        # weakest seed relative to global band maximum
PARTITION_MIN_SEP_BINS = 3.0
PARTITION_MIN_BINS = 3
PARTITION_MIN_ENERGY_FRAC = 0.01
N_PARTITIONS_KEEP = 3
HARMONIC_K_TOL = 0.20
HARMONIC_ORDERS = (2, 3)
SIDEBAND_K_TOL = 0.20             # relative vector mismatch tolerance
HARMONIC_ANG_TOL = 25.0           # deg
DOMINANCE_AMBIGUOUS = 0.80        # E2/E1 above this means dominance is unclear
                                  # (calibrate from the printed distribution)
GAP_LEAKAGE_MAX = 0.02            # mask power in the wave band, relative to DC.
                                  # Calibrated on synthetic masks: a fully
                                  # valid box gives 0, 5 % scattered holes give
                                  # 0.008, and a structured stripe pattern
                                  # (every 7th column missing) gives 0.056. The
                                  # metric separates random from structured
                                  # gaps, which is the point.
DEMOTE_ON_GAP_LEAKAGE = True      # set False if the run shows it never fires

# ---- 180 deg lobe resolution ------------------------------------------------
VB_SIGN = -1                      
TILT_SIGN = -1                    
LOBE_MIN_CONF = 0.25
LOBE_MIN_ALIGN = 0.35
LOBE_FALLBACK = "flag"            # "flag" only; "domain" is deliberately gone

# ---- tilt antisymmetry diagnostic -------------------------------------------
TILT_DIAG = True
TILT_DIAG_NBINS = 41
TILT_DIAG_KMAX = 0.07             # rad/m

# ---- spectral internal quality ---------------------------------------------
QC_MIN_R = 0.40
QC_MIN_COHERENCE = 0.05
COHERENCE_ALPHA = 0.05
QC_MAX_THETA_CI = 45.0
QC_MIN_BOOT_SUCCESS = 0.60
QC_MIN_LOBE_STABILITY = 0.70
QC_MIN_DOMINANCE_STABILITY = 0.70
QC_MIN_WINDOW_LOBE_STABILITY = 0.70
QC_MIN_WINDOW_MATCH_FRAC = 0.60

# ---- sigma0 normalization ----------------------------------------------------
SIG0_MODE = "auto"               # "auto", "linear", or "db"

# ---- geographic gridding diagnostic ----------------------------------------
RUN_GRID_SYNTHETIC_DIAG = True


# ============================================================
# STARTUP VALIDATION
# ============================================================

apply_overrides(globals(), ("common", "pipeline", "radon"), final=True)


def validate_config():
    """Refuse unsafe or internally inconsistent configurations."""
    problems = []

    if QUAL_BIT_POLICY not in {"require", "warn", "ignore"}:
        problems.append("QUAL_BIT_POLICY must be require/warn/ignore")
    if RES_MODE not in {"adaptive", "fixed"}:
        problems.append("RES_MODE must be adaptive/fixed")
    if not (0.0 <= PARTITION_SEED_FRAC <= 1.0 and 0.0 < PARTITION_FRAC <= 1.0):
        problems.append("partition thresholds must lie in (0,1]")
    if T_VALID_RANGE[1] > 16.0:
        warnings.warn("T_VALID_RANGE upper limit is >16 s; your requested 16 s cutoff is not active")

    # The primary scale must work at least outside the diagnostic inner swath.
    res_probe = max(RES_FIXED if RES_MODE == "fixed" else 45.0, 1.0)
    nwin = int(round(LWIN / res_probe))
    step = max(1, nwin - int(round(OVERLAP_FRAC * nwin)))
    ncell = int(np.floor(BOX_SIZE / res_probe))
    n_per_axis = len(range(0, max(ncell - nwin + 1, 0), step))
    n_windows = n_per_axis ** 2
    if n_windows < MIN_N_WINDOWS:
        problems.append(
            f"primary LWIN={LWIN:.0f} m in BOX_SIZE={BOX_SIZE:.0f} m yields "
            f"only {n_windows} windows at res={res_probe:.0f} m"
        )

    if MODEL_LOBE_MODE not in {"off", "unresolved"}:
        problems.append("MODEL_LOBE_MODE must be off/unresolved")
    if CURRENT_DEPTH_MODE not in {"surface", "kirby_chen"}:
        problems.append("CURRENT_DEPTH_MODE must be surface/kirby_chen")
    if UNRESOLVED_TC_POLICY not in {"nan", "keep"}:
        problems.append("UNRESOLVED_TC_POLICY must be nan/keep")
    if not (0.0 < MODEL_LOBE_MAX_DEV < 90.0):
        problems.append("MODEL_LOBE_MAX_DEV must lie in (0, 90) deg")
    if RESPONSE_MIN_MASK not in SENS_RESPONSE_MASKS:
        warnings.warn("production RESPONSE_MIN_MASK is not among SENS_RESPONSE_MASKS")

    if LOBE_FALLBACK != "flag":
        problems.append("LOBE_FALLBACK must remain 'flag'")

    if problems:
        raise ValueError("configuration problems:\n  - " + "\n  - ".join(problems))

    print(f"[cfg] primary Welch: ~{n_windows} windows at res={res_probe:.0f} m")
    print(f"[cfg] period analysis range: {T_VALID_RANGE[0]:.1f}-{T_VALID_RANGE[1]:.1f} s"
          + (" (hard reject)" if PERIOD_HARD_REJECT else " (quality flag)"))


# ============================================================
# GEOMETRY AND INSTRUMENT RESPONSE
# ============================================================

def across_track_posting(rho_m, h_sat=H_SAT):
    """
    Ground posting in the across-track direction, metres.

        d_ac = (c / 2 fs) / sin(theta + alpha)

    with theta the incidence angle and alpha = rho / R_E the Earth-curvature
    term. Reproduces the dx column of Yu et al. Table I to a few percent.
    """
    rho = np.abs(np.asarray(rho_m, dtype=float))
    rho = np.maximum(rho, 1.0)
    theta = np.arctan(rho / h_sat)
    alpha = rho / R_EARTH
    return RANGE_GATE_M / np.sin(theta + alpha)


def boxcar3_response(k, d):
    """Amplitude response of a 3-point boxcar of spacing d: (1+2cos(kd))/3."""
    return (1.0 + 2.0 * np.cos(k * d)) / 3.0


def rect_response(k, w):
    """Amplitude response of a rect average of width w: sinc(kw/2)."""
    a = 0.5 * k * w
    return np.where(np.abs(a) < 1e-12, 1.0, np.sin(a) / np.where(a == 0, 1, a))


def instrument_response_power(spec, geom, res, d_ac):
    """
    Power response of the unchanged FFT-v4.2 processing chain, evaluated on
    the active estimator grid.

    For the Radon/Fourier-slice estimator this is evaluated at every
    (frequency, projection-angle) sample using its equivalent ground-plane
    wave vector. For a legacy Cartesian spectrum it falls back to (KX, KY).
    """
    if spec.get("estimator") == "radon_fourier_slice":
        KX, KY = spec["RKX"], spec["RKY"]
    else:
        KX, KY = spec["KX"], spec["KY"]

    if geom["a_hat"] is None or geom["r_hat"] is None:
        H = np.ones_like(KX)
    else:
        kaz = KX * geom["a_hat"][0] + KY * geom["a_hat"][1]
        kr = KX * geom["r_hat"][0] + KY * geom["r_hat"][1]
        H = (boxcar3_response(kaz, ALONG_TRACK_POSTING) *
             boxcar3_response(kr, d_ac))

    if INCLUDE_GRID_RESPONSE:
        H = H * rect_response(KX, res) * rect_response(KY, res)

    return np.clip(H ** 2, 1e-6, None)


# ============================================================
# BASIC HELPERS
# ============================================================

def detrend_plane_valid(Z, valid):
    Ny, Nx = Z.shape
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(Nx * Ny)]
    vm = valid.ravel()
    if vm.sum() < 10:
        return np.zeros_like(Z, dtype=float)
    beta, *_ = np.linalg.lstsq(A[vm], Z.ravel()[vm], rcond=None)
    out = Z - (A @ beta).reshape(Ny, Nx)
    out[~valid] = 0.0
    return out


def overlap_corr(win1d, shift):
    if shift <= 0 or shift >= win1d.size:
        return 0.0
    return float(np.sum(win1d[:-shift] * win1d[shift:]) / np.sum(win1d ** 2))


def eff_n_2d(N, win1d, step):
    if N <= 1:
        return float(N)
    rho = overlap_corr(win1d, step)
    return float(max(N / (1.0 + 4.0 * rho ** 2 + 4.0 * rho ** 4), 1.0))


def angular_difference_deg(theta, ref):
    return ((np.asarray(theta) - ref + 180.0) % 360.0) - 180.0


def circular_resultant_R(theta_deg):
    th = np.asarray(theta_deg, dtype=float)
    th = th[np.isfinite(th)]
    if th.size == 0:
        return np.nan
    t = np.deg2rad(th)
    return float(np.hypot(np.mean(np.cos(t)), np.mean(np.sin(t))))


def circular_mean_deg(theta_deg):
    th = np.asarray(theta_deg, dtype=float)
    th = th[np.isfinite(th)]
    if th.size == 0:
        return np.nan
    t = np.deg2rad(th)
    return float(np.degrees(np.arctan2(np.mean(np.sin(t)),
                                       np.mean(np.cos(t)))) % 360.0)


def circular_sigma_deg(theta_deg):
    R = circular_resultant_R(theta_deg)
    if not np.isfinite(R):
        return np.nan
    R = min(max(R, 1e-12), 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def linear_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi), float(0.5 * (hi - lo))


def circular_ci(theta_boot, theta_ref):
    tb = np.asarray(theta_boot, dtype=float)
    tb = tb[np.isfinite(tb)]
    if tb.size == 0 or not np.isfinite(theta_ref):
        return np.nan, np.nan, np.nan
    d = angular_difference_deg(tb, theta_ref)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return (float((theta_ref + lo) % 360.0),
            float((theta_ref + hi) % 360.0),
            float(0.5 * (hi - lo)))


def periods(lambda_m, theta_to_deg, U_e, U_n, h):
    """Intrinsic period and ocean-current Doppler-shifted absolute period."""
    if not np.isfinite(lambda_m) or lambda_m <= 0:
        return np.nan, np.nan
    if not np.isfinite(h) or h <= 0:
        return np.nan, np.nan
    if not np.isfinite(theta_to_deg):
        return np.nan, np.nan

    k = 2.0 * np.pi / lambda_m
    sigma = np.sqrt(g * k * np.tanh(k * h))
    th = np.deg2rad((90.0 - theta_to_deg) % 360.0)
    omega = sigma + k * np.cos(th) * U_e + k * np.sin(th) * U_n

    return (float(2.0 * np.pi / sigma),
            float(2.0 * np.pi / omega) if omega > 0 else np.nan)


def direction_to_uv(theta_deg):
    th = np.deg2rad(theta_deg)
    return np.sin(th), np.cos(th)


def vec_to_deg_from_north(e, n):
    return float(np.degrees(np.arctan2(e, n)) % 360.0)


def k_to_bearing(kx, ky):
    """Wavevector (east, north) -> propagation bearing, deg clockwise from N."""
    return float((90.0 - (np.degrees(np.arctan2(ky, kx)) % 360.0)) % 360.0)


def axial_difference_deg(a, b):
    """Smallest difference between unoriented axes, in degrees [0,90]."""
    d = np.abs(angular_difference_deg(a, b))
    return np.minimum(d, 180.0 - d)


def _to_datetime64_scalar(v):
    if v is None:
        return None
    try:
        x = np.datetime64(v, "ns")
        if np.isnat(x):
            return None
        return x
    except Exception:
        return None


def _median_datetime64(values):
    a = np.asarray(values)
    if a.size == 0:
        return None
    try:
        a = a.astype("datetime64[ns]").ravel()
    except Exception:
        return None
    a = a[~np.isnat(a)]
    if a.size == 0:
        return None
    ns = a.astype("int64")
    return np.datetime64(int(np.median(ns)), "ns")


def decode_time_da(da):
    """Decode an xarray time-like DataArray without assuming a mission epoch.

    V4.2: numeric CF time arrays are decoded only at finite/non-fill entries.
    Some SWOT LR files contain the netCDF fill value ~9.969e36 in ``time``.
    Passing that value directly to pandas/cftime overflows before xarray can
    decode the useful samples.
    """
    if da is None:
        return None
    vals = np.asarray(da.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]")

    units = str(da.attrs.get("units", ""))
    cal = da.attrs.get("calendar", "standard")
    if np.issubdtype(vals.dtype, np.number) and "since" in units.lower():
        try:
            from xarray.coding.times import decode_cf_datetime

            vf = np.asarray(vals, dtype=float)
            # Reject NaN/Inf and sentinel/fill values.  Legitimate mission
            # times in seconds since an epoch are many orders of magnitude
            # smaller than 1e20.
            good = np.isfinite(vf) & (np.abs(vf) < 1.0e20)
            out = np.full(vf.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            if np.any(good):
                dec = np.asarray(decode_cf_datetime(vf[good], units, cal))
                out[good] = dec.astype("datetime64[ns]")
            return out
        except Exception as e:
            print(f"[time] could not CF-decode {da.name}: {e}")
    return None


def dataset_time_hint(ds):
    """Best available acquisition-time hint from decoded variables or attributes."""
    for name in ("illumination_time", "time", "time_counter", "time_instant"):
        if name in ds.variables:
            t = decode_time_da(ds[name])
            tm = _median_datetime64(t) if t is not None else None
            if tm is not None:
                return tm
    for a in ("time_coverage_start", "start_time", "time_granule_start",
              "first_meas_time", "beginning_date_time"):
        if a in ds.attrs:
            tm = _to_datetime64_scalar(ds.attrs[a])
            if tm is not None:
                return tm
    return None


def hours_between(a, b):
    if a is None or b is None:
        return np.nan
    try:
        return float(abs((np.datetime64(a, "ns") - np.datetime64(b, "ns")) /
                         np.timedelta64(1, "h")))
    except Exception:
        return np.nan


def wavenumber_from_period(T, h, grav=g):
    """Solve sigma^2=g k tanh(kh) for k by Newton iteration."""
    if not np.isfinite(T) or T <= 0 or not np.isfinite(h) or h <= 0:
        return np.nan
    omega = 2.0 * np.pi / T
    k = max(omega ** 2 / grav, 1e-8)
    for _ in range(30):
        kh = k * h
        th = np.tanh(kh)
        f = grav * k * th - omega ** 2
        df = grav * (th + k * h * (1.0 - th ** 2))
        if df <= 0:
            break
        kn = max(k - f / df, 1e-10)
        if abs(kn - k) <= 1e-10 * max(k, 1.0):
            k = kn
            break
        k = kn
    return float(k)


def k_from_absolute_frequency(f_abs, h, u_along, grav=g):
    """Invert 2 pi f = sqrt(g k tanh(kh)) + k u_along for k (bisection)."""
    if not (np.isfinite(f_abs) and f_abs > 0 and np.isfinite(h) and h > 0):
        return np.nan
    u = float(u_along) if np.isfinite(u_along) else 0.0
    omega = 2.0 * np.pi * f_abs
    lo, hi = 1e-5, 2.0
    if np.sqrt(grav * hi * np.tanh(hi * h)) + hi * u < omega:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sqrt(grav * mid * np.tanh(mid * h)) + mid * u < omega:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _file_time_from_name(path):
    m = re.search(r"(\d{8})T(\d{2})(\d{2})?Z", Path(path).name)
    if not m:
        return None
    s = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}T{m.group(2)}:{m.group(3) or '00'}"
    return np.datetime64(s, "ns")


def resolve_current_file(pattern, target_time):
    """Single path, or the glob match whose filename time is nearest the SWOT time."""
    if pattern is None or not any(c in str(pattern) for c in "*?["):
        return pattern
    files = sorted(glob.glob(str(pattern)))
    if not files:
        print(f"[current] no file matches {pattern}")
        return None
    if target_time is None:
        print(f"[current] no SWOT time; using {Path(files[0]).name}")
        return files[0]
    tt = np.datetime64(target_time, "ns")
    dts = [abs((_file_time_from_name(f) - tt) / np.timedelta64(1, "s"))
           if _file_time_from_name(f) is not None else np.inf for f in files]
    j = int(np.argmin(dts))
    print(f"[current] {len(files)} files; nearest by name: {Path(files[j]).name} "
          f"({dts[j]/3600:.2f} h)")
    return files[j]


def s_coordinate_C(s, a, b):
    """MARS3D / Song-Haidvogel stretching function C(s)."""
    s = np.asarray(s, float)
    if not np.isfinite(a) or abs(a) < 1e-9:
        return s
    c1 = np.sinh(a * s) / np.sinh(a)
    c2 = (np.tanh(a * (s + 0.5)) - np.tanh(0.5 * a)) / (2.0 * np.tanh(0.5 * a))
    return (1.0 - b) * c1 + b * c2





def effective_current(cur, k, h):
    """
    Current seen by a wave of wavenumber k (Kirby & Chen 1989):

        U_eff = 2k/sinh(2kD) * int_{-D}^{0} U(z) cosh(2k(z+D)) dz

    Each model layer is treated as uniform between its interfaces, and the
    weight is integrated exactly over the layer. z comes from the MARS3D
    s-coordinate (z = eta(1+s) + hc s + (H0-hc) C(s)), referenced to the free
    surface, with D = H0 + eta the MODEL water depth so that the layers tile
    [-D, 0]. Without the s-coordinate metrics, z = s*h with the pipeline depth.
    """
    out = {"U_e": cur["U_e"], "U_n": cur["U_n"], "U_e_std": cur["U_e_std"],
           "U_n_std": cur["U_n_std"], "U_e_surface": cur["U_e"],
           "U_n_surface": cur["U_n"], "current_mode": "surface",
           "current_weight_top10m": np.nan}
    P = cur.get("profile")
    if CURRENT_DEPTH_MODE != "kirby_chen" or P is None or not np.isfinite(k) or k <= 0:
        return out
    if P["z_w"] is not None:
        z_w, D = np.asarray(P["z_w"], float), float(P["D"])
    else:
        if not np.isfinite(h) or h <= 0:
            return out
        z_w, D = np.asarray(P["s_w"], float) * h, float(h)
    if not np.isfinite(D) or D <= 0:
        return out
    ue, vn = np.asarray(P["ue"], float), np.asarray(P["vn"], float)

    def W(z):   # antiderivative of the normalised weight; W(-D)=0, W(0)=1
        return (np.exp(2*k*z) - np.exp(-2*k*(z + 2*D))) / (1.0 - np.exp(-4*k*D))

    wl = W(z_w[1:]) - W(z_w[:-1])
    ok = np.isfinite(ue) & np.isfinite(vn) & np.isfinite(wl) & (wl > 0)
    if ok.sum() < 2 or wl[ok].sum() <= 0:
        return out
    out["U_e"] = float(np.sum(wl[ok]*ue[ok]) / np.sum(wl[ok]))
    out["U_n"] = float(np.sum(wl[ok]*vn[ok]) / np.sum(wl[ok]))
    out["current_mode"] = "kirby_chen_" + P.get("source", "")
    out["current_weight_top10m"] = float(1.0 - W(max(-10.0, -D)))
    return out


def resolve_lobes_with_model(parts, ww):
    """
    Choose the 180 deg lobe from the model where SWOT could not.

    lobe_resolved keeps its SWOT-only meaning. lobe_source records the final
    decision: "vb" (SWOT velocity bunching), "model", or "unresolved".
    """
    th_m = ww.get("ww3_dir_to_degN", np.nan) if ww else np.nan
    usable_model = (MODEL_LOBE_MODE != "off" and np.isfinite(th_m) and
                    ww.get("ww3_lobe_allowed", False))
    for p in parts:
        p["lobe_method_swot"] = p["lobe_method"]
        p["model_lobe_dev_deg"] = np.nan
        p["vb_agrees_model"] = np.nan
        if np.isfinite(th_m):
            p["model_lobe_dev_deg"] = float(abs(angular_difference_deg(p["theta_to"], th_m)))
        if p["lobe_resolved"]:
            p["lobe_source"] = "vb"
            if np.isfinite(p["model_lobe_dev_deg"]):
                p["vb_agrees_model"] = float(p["model_lobe_dev_deg"] <= 90.0)
            continue
        p["lobe_source"] = "unresolved"
        if not usable_model:
            continue
        had_weak_vote = p["lobe_method"] == "vb_weak"
        if abs(angular_difference_deg(p["theta_to"], th_m)) > 90.0:
            p["kx"], p["ky"] = -p["kx"], -p["ky"]
            for key in ("k_along", "k_across"):
                if np.isfinite(p.get(key, np.nan)):
                    p[key] = -p[key]
            p["theta_to"] = (p["theta_to"] + 180.0) % 360.0
            if had_weak_vote:
                p["vb_agrees_model"] = 0.0
        elif had_weak_vote:
            p["vb_agrees_model"] = 1.0
        dev = float(abs(angular_difference_deg(p["theta_to"], th_m)))
        p["model_lobe_dev_deg"] = dev
        if dev <= MODEL_LOBE_MAX_DEV:
            p["lobe_source"] = "model"
            p["lobe_method"] = "model"


def align_to_lobe(p, kx_ref, ky_ref):
    """Propagation bearing of partition p on the lobe of (kx_ref, ky_ref)."""
    if p["kx"] * kx_ref + p["ky"] * ky_ref < 0:
        return (p["theta_to"] + 180.0) % 360.0
    return p["theta_to"]


def wave_periods_final(lam, th_to, cur_eff, depth, direction_resolved):
    """Intrinsic period, absolute period, and both-lobe absolute bounds."""
    Ti, Tc = periods(lam, th_to, cur_eff["U_e"], cur_eff["U_n"], depth)
    _, Tc_flip = periods(lam, (th_to + 180.0) % 360.0, cur_eff["U_e"], cur_eff["U_n"], depth)
    lo = np.nanmin([Tc, Tc_flip]) if np.isfinite([Tc, Tc_flip]).any() else np.nan
    hi = np.nanmax([Tc, Tc_flip]) if np.isfinite([Tc, Tc_flip]).any() else np.nan
    if not direction_resolved and UNRESOLVED_TC_POLICY == "nan":
        Tc = np.nan
    return Ti, Tc, float(lo), float(hi)


def partition_weights(E, spec, m):
    """Energy weights of a partition mask on the active estimator grid."""
    w = np.maximum(E[m], 0.0)
    if spec.get("estimator") == "radon_fourier_slice" and "RK" in spec:
        w = w * spec["RK"][m]
    return w


def response_sensitivity(E_dn, C, spec, resp_pow, geom_grids, p_main):
    """Re-partition with each SENS_RESPONSE_MASKS value; match to p_main."""
    out = {}
    if not RUN_RESPONSE_SENSITIVITY:
        return out
    kx0, ky0, e0 = p_main["kx"], p_main["ky"], p_main["energy"]
    for mval in SENS_RESPONSE_MASKS:
        tag = f"sens_m{int(round(100 * mval)):02d}_"
        vals = {"lambda_m": np.nan, "theta_degN": np.nan, "resp_pow": np.nan,
                "energy_ratio": np.nan, "match_dk_rel": np.nan}
        try:
            ps = extract_partitions(E_dn, C, spec, resp_pow, geom_grids, resp_min=mval)
        except Exception:
            ps = []
        if ps:
            d = [min(np.hypot(q["kx"] - kx0, q["ky"] - ky0),
                     np.hypot(-q["kx"] - kx0, -q["ky"] - ky0)) for q in ps]
            j = int(np.argmin(d)); q = ps[j]
            th = align_to_lobe(q, kx0, ky0)
            vals.update({"lambda_m": float(q["lambda_m"]),
                         "theta_degN": float((th + 180) % 360 if DIR_CONVENTION == "from" else th),
                         "resp_pow": float(q["resp_pow"]),
                         "energy_ratio": float(q["energy"] / e0) if e0 > 0 else np.nan,
                         "match_dk_rel": float(d[j] / max(p_main["k"], 1e-12))})
            if abs(mval) < 1e-12:
                w = partition_weights(E_dn, spec, q["mask"])
                cut = resp_pow[q["mask"]] < RESPONSE_MIN_MASK
                out["p1_masked_energy_frac"] = (float(np.sum(w[cut]) / np.sum(w))
                                                if np.sum(w) > 0 else np.nan)
        for k_, v_ in vals.items():
            out[tag + k_] = v_
    out.setdefault("p1_masked_energy_frac", np.nan)
    return out


# ============================================================
# LOAD SWOT SCENE
# ============================================================

def _preprocess_pixc(ds):
    if "num_pixc_lines" in ds.dims:
        ds = ds.drop_dims("num_pixc_lines", errors="ignore")
    missing = [v for v in SWOT_VARS if v not in ds.variables]
    if missing:
        raise KeyError(f"granule is missing required variables: {missing}")
    keep = list(SWOT_VARS) + [v for v in SWOT_GEOM_VARS if v in ds.variables]
    return ds[keep]


def report_qual_bits(files, group):
    try:
        ds = xr.open_dataset(files[0], group=group, engine="h5netcdf", mask_and_scale=False)
    except Exception as e:
        print(f"[qual] could not inspect bits: {e}")
        return
    try:
        if SWOT_QUAL_VAR not in ds.variables:
            return
        a = ds[SWOT_QUAL_VAR].attrs
        names = a.get("flag_meanings", "").split()
        masks = a.get("flag_masks", None)
        if masks is None or not names:
            print("[qual] no flag_masks/flag_meanings on geolocation_qual")
            return
        q = np.asarray(ds[SWOT_QUAL_VAR].values).ravel().astype(np.uint64)
        print(f"[qual] {SWOT_QUAL_VAR} bits (first granule, n={q.size}):")
        for nm, mk in zip(names, masks):
            mk = np.uint64(int(mk))
            rate = float(np.mean((q & mk) > 0))
            print(f"        {hex(int(mk)):>12}  {nm:<38} set on {100*rate:6.2f} %")
    finally:
        ds.close()


def _filename_cycle_pass(path):
    """Return the most plausible 3-digit cycle/pass pair from a SWOT filename."""
    name = Path(path).name
    pairs = re.findall(r"_(\d{3})_(\d{3})(?:_|\.)", name)
    if not pairs:
        return None
    # Prefer the configured pair if it is present; otherwise first plausible pair.
    target = (f"{int(CYCLE):03d}", f"{int(PASS):03d}")
    if target in pairs:
        return tuple(map(int, target))
    return tuple(map(int, pairs[0]))


def validate_granule_identity(files, group):
    times = []
    for f in files:
        cp = _filename_cycle_pass(f)
        if cp is not None and STRICT_FILE_IDENTITY and cp != (int(CYCLE), int(PASS)):
            raise ValueError(f"{Path(f).name}: parsed cycle/pass={cp}, expected {(CYCLE, PASS)}")
        try:
            ds0 = xr.open_dataset(f, group=group, engine="h5netcdf", decode_times=True)
            tm = dataset_time_hint(ds0)
            ds0.close()
            if tm is not None:
                times.append(tm)
        except Exception as e:
            print(f"[time] metadata check skipped for {Path(f).name}: {e}")

    acq = _to_datetime64_scalar(SWOT_ACQ_TIME_OVERRIDE)
    if acq is None and times:
        acq = _median_datetime64(times)

    if acq is not None:
        d0 = _to_datetime64_scalar(DATE)
        if d0 is not None:
            # DATE is midnight; compare after centring the date at noon.
            dmid = d0 + np.timedelta64(12, "h")
            if hours_between(acq, dmid) > ACQ_DATE_TOLERANCE_H and STRICT_FILE_IDENTITY:
                raise ValueError(f"PIXC acquisition {acq} is inconsistent with DATE={DATE}")
        print(f"[time] SWOT acquisition ~= {acq}")
    else:
        print("[time] SWOT acquisition time could not be verified")
    return acq


def _sig0_to_linear(sig0, attrs):
    units = str(attrs.get("units", "")).lower()
    mode = SIG0_MODE.lower()
    is_db = mode == "db" or (mode == "auto" and "db" in units)
    if mode not in {"auto", "linear", "db"}:
        raise ValueError("SIG0_MODE must be auto/linear/db")
    s = np.asarray(sig0, dtype=float)
    if is_db:
        s = 10.0 ** (s / 10.0)
    return s, bool(is_db), units


def load_swot_tile(swot_files, group, utm_zone):
    files = (sorted(glob.glob(str(swot_files))) if isinstance(swot_files, (str, Path))
             else sorted(str(f) for f in swot_files))
    if not files:
        raise FileNotFoundError(f"no PIXC granules matched: {swot_files}")
    print(f"[load] {len(files)} granule(s) found")

    if REPORT_QUAL_BITS:
        report_qual_bits(files, group)
    if SWOT_QUAL_BITMASK == 0 and QUAL_BIT_POLICY == "require":
        raise ValueError(
            "SWOT_QUAL_BITMASK=0. V4 production policy requires an explicit bitmask. "
            "Use the bit dictionary printed above, set the bits to reject, and rerun. "
            "If you intentionally want no bit filtering set QUAL_BIT_POLICY='warn' or 'ignore'."
        )
    if SWOT_QUAL_BITMASK == 0 and QUAL_BIT_POLICY == "warn":
        warnings.warn("SWOT geolocation quality bitmask is disabled")
    if SWOT_QUAL_BITMASK:
        print(f"[qual] hard-reject mask = {hex(SWOT_QUAL_BITMASK)}")
        print("[qual] layover_significant and phase_unwrapping_suspect are retained "
              "as per-box soft diagnostics")

    acq_time = validate_granule_identity(files, group)

    # Inspect sigma0 metadata before mfdataset closes attributes across files.
    d0 = xr.open_dataset(files[0], group=group, engine="h5netcdf")
    sig0_attrs = dict(d0["sig0"].attrs) if "sig0" in d0 else {}
    d0.close()

    ds = xr.open_mfdataset(
        files, group=group, concat_dim="points", combine="nested",
        engine="h5netcdf", preprocess=_preprocess_pixc,
        data_vars="minimal", coords="minimal", compat="override",
    )

    h_sat = H_SAT
    for attr in ("mean_sat_alt", "sat_alt"):
        if attr in ds.attrs:
            try:
                v = float(np.asarray(ds.attrs[attr]).ravel()[0])
                if 5e5 < v < 2e6:
                    h_sat = v
            except Exception:
                pass

    lat = np.asarray(ds["latitude"].values).ravel()
    lon = np.asarray(ds["longitude"].values).ravel()
    ssh = np.asarray(ds["height"].values).ravel()
    sig0_raw = np.asarray(ds["sig0"].values).ravel()
    sig0, sig0_from_db, sig0_units = _sig0_to_linear(sig0_raw, sig0_attrs)
    qraw = np.asarray(ds[SWOT_QUAL_VAR].values).ravel()

    geom = {}
    illum_time_decoded = None
    for v in SWOT_GEOM_VARS:
        if v in ds.variables:
            if v == "illumination_time":
                illum_time_decoded = decode_time_da(ds[v])
                raw_t = np.asarray(ds[v].values).ravel()
                if np.issubdtype(raw_t.dtype, np.datetime64):
                    t0 = raw_t[~np.isnat(raw_t)][0] if np.any(~np.isnat(raw_t)) else np.datetime64("1970-01-01")
                    geom[v] = ((raw_t - t0) / np.timedelta64(1, "s")).astype(float)
                else:
                    geom[v] = raw_t.astype(float)
            else:
                geom[v] = np.asarray(ds[v].values).ravel().astype(float)
        else:
            print(f"[warn] {v} absent from the granules")
    if acq_time is None and illum_time_decoded is not None:
        acq_time = _median_datetime64(illum_time_decoded)
    ds.close()

    n_total = lat.size
    ct_all = geom.get("cross_track")
    if APPLY_CROSS_TRACK_FILTER:
        if ct_all is None:
            raise KeyError("cross_track absent while APPLY_CROSS_TRACK_FILTER=True")
        act = np.abs(ct_all)
        ct_ok = np.isfinite(ct_all) & (act >= CROSS_TRACK_MIN) & (act <= CROSS_TRACK_MAX)
    else:
        ct_ok = np.ones(n_total, dtype=bool)

    # Missing/NaN quality values must never be converted to zero (= good).
    qfinite = np.isfinite(qraw)
    qi = np.zeros(n_total, dtype=np.uint64)
    qi[qfinite] = np.asarray(qraw[qfinite], dtype=np.uint64)
    qual_ok = qfinite.copy()
    if SWOT_QUAL_BITMASK:
        qual_ok &= (qi & np.uint64(SWOT_QUAL_BITMASK)) == 0
    if np.isfinite(SWOT_QUAL_MAX):
        qf = np.asarray(qraw, dtype=float)
        qual_ok &= np.where(qfinite, qf < SWOT_QUAL_MAX, False)

    valid = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ssh) &
             np.isfinite(sig0) & (sig0 > 0) & qual_ok & ct_ok)
    print(f"[load] {n_total} pixels read; {int(valid.sum())} kept")
    print(f"[qual] bitmask {hex(SWOT_QUAL_BITMASK)} rejected {int(np.sum(~qual_ok))} pixels")
    if APPLY_CROSS_TRACK_FILTER:
        print(f"[geom] loading |cross_track|={CROSS_TRACK_MIN/1e3:.0f}-{CROSS_TRACK_MAX/1e3:.0f} km")

    lat, lon, ssh, sig0 = lat[valid], lon[valid], ssh[valid], sig0[valid]
    if lat.size == 0:
        raise RuntimeError("no valid SWOT pixels after filtering")
    proj = Proj(proj="utm", zone=utm_zone, ellps="WGS84")
    x, y = proj(lon, lat)

    data = {
        "lon": lon, "lat": lat, "x": np.asarray(x), "y": np.asarray(y),
        "ssh": ssh, "sig0": sig0, "proj": proj, "h_sat": h_sat,
        "geolocation_qual": qi[valid],
        "cross_track": geom["cross_track"][valid] if "cross_track" in geom else None,
        "illum_time": geom["illumination_time"][valid] if "illumination_time" in geom else None,
        "acq_time": acq_time, "sig0_from_db": sig0_from_db,
        "sig0_units_input": sig0_units, "files": files,
    }
    if data["illum_time"] is None:
        print("[warn] illumination_time missing -> along-track geometry may be unavailable")
    print(f"[sig0] input units='{sig0_units}', converted_from_db={sig0_from_db}; "
          "cross-spectra use per-window normalized linear sigma0")
    return data


# ============================================================
# RAIN CELLS FROM L2_LR_SSH EXPERT
# ============================================================

def _lr_bit(ds_raw, var, name):
    if var not in ds_raw.variables:
        return None
    a = ds_raw[var].attrs
    names = a.get("flag_meanings", "").split()
    masks = a.get("flag_masks", None)
    if masks is None or name not in names:
        print(f"  [rain] bit '{name}' not present in {var}")
        return None
    m = np.uint64(int(masks[names.index(name)]))
    return (ds_raw[var].values.astype(np.uint64) & m) > 0


def load_rain_cells(lr_files, proj, target_time=None):
    """Load/time-screen LR rain masks and return ocean pixels plus true rain state."""
    files = sorted(glob.glob(lr_files)) if isinstance(lr_files, str) else list(lr_files)
    if not files:
        print("[rain] no LR granule matched; rain screening disabled")
        return None

    lat_l, lon_l, rf_l, rr_l, sc_l, ct_l, bit_l = [], [], [], [], [], [], []
    used_dt = []
    for path in files:
        cp = _filename_cycle_pass(path)
        if cp is not None and STRICT_FILE_IDENTITY and cp != (int(CYCLE), int(PASS)):
            raise ValueError(f"LR file {Path(path).name}: parsed cycle/pass={cp}, expected {(CYCLE, PASS)}")
        # Normal LR variables benefit from CF decoding.  A few products contain
        # a huge fill value (~9.969e36) in the time variable; if automatic
        # decoding trips on it, reopen without decoding and let decode_time_da()
        # handle only the valid entries.
        try:
            ds = xr.open_dataset(path, decode_times=True)
        except (ValueError, OverflowError) as e:
            print(f"  [rain] automatic time decoding failed for {Path(path).name}; "
                  "reopening with decode_times=False")
            ds = xr.open_dataset(path, decode_times=False)

        # Raw access is needed only to inspect the bit field.  Do NOT ask
        # xarray to decode time here: mask_and_scale=False deliberately exposes
        # the netCDF fill value and it is not a physical timestamp.
        raw = None
        if RAIN_USE_INSTRUMENT_BIT:
            raw = xr.open_dataset(path, mask_and_scale=False, decode_times=False)

        tm = dataset_time_hint(ds)
        dt = hours_between(tm, target_time)
        if target_time is not None and np.isfinite(dt) and dt > LR_MAX_TIME_DELTA_H:
            print(f"  [rain skip] {Path(path).name}: dt={dt:.2f} h")
            ds.close()
            if raw is not None:
                raw.close()
            continue
        print(f"  [rain] {Path(path).name}" + (f" dt={dt:.2f} h" if np.isfinite(dt) else ""))
        used_dt.append(dt)

        lon = np.asarray(ds["longitude"].values, dtype=float)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lat_l.append(np.asarray(ds["latitude"].values, dtype=float))
        lon_l.append(lon)
        rf_l.append(np.asarray(ds["rain_flag"].values, dtype=float))
        rr_l.append(np.asarray(ds["rain_rate"].values, dtype=float))
        sc_l.append(np.asarray(ds["ancillary_surface_classification_flag"].values, dtype=float))
        ct_l.append(np.asarray(ds["cross_track_distance"].values, dtype=float))
        b = (_lr_bit(raw, "swh_karin_qual", "suspect_rain_likely")
             if raw is not None else None)
        bit_l.append(b if b is not None else np.zeros_like(rf_l[-1], dtype=bool))
        ds.close()
        if raw is not None:
            raw.close()

    if not lat_l:
        print("[rain] no LR files within time tolerance")
        return None
    cat = lambda L: np.concatenate(L, axis=0)
    lat, lon = cat(lat_l), cat(lon_l)
    rf, rr, sc, ct = cat(rf_l), cat(rr_l), cat(sc_l), cat(ct_l)
    bit = cat(bit_l)
    ok = np.isfinite(lat) & np.isfinite(lon)
    mask = np.isin(rf, RAIN_FLAG_VALUES)
    if RAIN_RATE_MIN is not None:
        mask |= rr >= RAIN_RATE_MIN
    if RAIN_USE_INSTRUMENT_BIT:
        mask |= bit
    mask &= ok & (np.abs(ct) >= RAIN_NADIR_GAP_MIN_CT)

    struct = ndimage.generate_binary_structure(2, 2)
    labels, n = ndimage.label(mask, structure=struct)
    if n and RAIN_MIN_CELL_PIXELS > 1:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        labels[np.isin(labels, np.where(sizes < RAIN_MIN_CELL_PIXELS)[0])] = 0
        labels, n = ndimage.label(labels > 0, structure=struct)
    sizes = np.bincount(labels.ravel()) if n else np.zeros(1, dtype=int)
    if n: sizes[0] = 0
    area_km2 = sizes.astype(float) * 4.0

    x = np.full(lat.shape, np.nan); y = np.full(lat.shape, np.nan)
    x[ok], y[ok] = proj(lon[ok], lat[ok])
    ocean = ok & (sc == 0)
    rain_pt = ocean & (labels > 0)
    print(f"[rain] {n} rain cell(s), ocean rain fraction={np.mean(rain_pt[ocean]) if ocean.any() else np.nan:.3f}")

    return {
        "x_all": x[ocean], "y_all": y[ocean], "rr_all": rr[ocean],
        "rain_all": (labels[ocean] > 0), "label_all": labels[ocean],
        "x_rain": x[rain_pt], "y_rain": y[rain_pt], "lab_rain": labels[rain_pt],
        "area_km2": area_km2,
        "tree_all": cKDTree(np.c_[x[ocean], y[ocean]]) if ocean.any() else None,
        "tree_rain": cKDTree(np.c_[x[rain_pt], y[rain_pt]]) if rain_pt.any() else None,
        "time_delta_h": float(np.nanmin(used_dt)) if np.any(np.isfinite(used_dt)) else np.nan,
    }


def _rect_distance(px, py, x0, x1, y0, y1):
    dx = np.maximum.reduce([x0 - px, np.zeros_like(px), px - x1])
    dy = np.maximum.reduce([y0 - py, np.zeros_like(py), py - y1])
    return np.hypot(dx, dy)


def box_rain(rain, box):
    """Rain diagnostics; rain_frac is the actual fraction of LR ocean pixels in the box."""
    out = {"lr_n": 0, "rain_frac": np.nan, "rain_rate_max": np.nan,
           "dist_to_rain_km": np.nan, "rain_cell_area_km2": np.nan,
           "rain_status": "no_lr_coverage", "rain_reject": False,
           "rain_time_delta_h": np.nan}
    if rain is None:
        out["rain_status"] = "not_screened"
        return out
    out["rain_time_delta_h"] = rain.get("time_delta_h", np.nan)

    x0, x1, y0, y1 = box["x_min"], box["x_max"], box["y_min"], box["y_max"]
    xc, yc = 0.5*(x0+x1), 0.5*(y0+y1)
    half_diag = BOX_SIZE * np.sqrt(2.0) / 2.0

    if rain["tree_all"] is not None:
        cand = rain["tree_all"].query_ball_point((xc, yc), half_diag)
        if cand:
            cand = np.asarray(cand, dtype=int)
            cx, cy = rain["x_all"][cand], rain["y_all"][cand]
            ins = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
            k = cand[ins]
            out["lr_n"] = int(k.size)
            if k.size:
                out["rain_rate_max"] = float(np.nanmax(rain["rr_all"][k]))
                out["rain_frac"] = float(np.mean(rain["rain_all"][k]))
                out["rain_status"] = "rain_core" if np.any(rain["rain_all"][k]) else "clean"

    if rain["tree_rain"] is not None:
        r = RAIN_BUFFER_M + half_diag
        cand = rain["tree_rain"].query_ball_point((xc, yc), r)
        if cand:
            cand = np.asarray(cand, dtype=int)
            d = _rect_distance(rain["x_rain"][cand], rain["y_rain"][cand], x0, x1, y0, y1)
            j = int(np.argmin(d))
            out["dist_to_rain_km"] = float(d[j] / 1000.0)
            cid = int(rain["lab_rain"][cand[j]])
            out["rain_cell_area_km2"] = float(rain["area_km2"][cid])
        else:
            dd, _ = rain["tree_rain"].query((xc, yc))
            out["dist_to_rain_km"] = float(max(dd-half_diag, 0.0)/1000.0)

    d = out["dist_to_rain_km"]
    if out["rain_status"] == "clean" and np.isfinite(d) and d*1000.0 <= RAIN_BUFFER_M:
        out["rain_status"] = "rain_buffer"
    out["rain_reject"] = out["rain_status"] in ("rain_core", "rain_buffer")
    if RAIN_NO_COVERAGE_ACTION == "reject" and out["rain_status"] == "no_lr_coverage":
        out["rain_reject"] = True
    return out


# ============================================================
# PER-BOX SWATH GEOMETRY
# ============================================================

def _plane_gradient(xs, ys, vs, max_pts=5000):
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(vs)
    if ok.sum() < 20:
        return np.nan, np.nan
    xs, ys, vs = xs[ok], ys[ok], vs[ok]
    if xs.size > max_pts:
        sel = np.linspace(0, xs.size - 1, max_pts).astype(int)
        xs, ys, vs = xs[sel], ys[sel], vs[sel]
    A = np.c_[xs - xs.mean(), ys - ys.mean(), np.ones(xs.size)]
    try:
        beta, *_ = np.linalg.lstsq(A, vs - vs.mean(), rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return float(beta[0]), float(beta[1])


def box_swath_geometry(data, sel):
    """
    a_hat : along-track unit vector, direction of increasing illumination_time.
    r_hat : look direction in the horizontal plane, AWAY from nadir, signed by
            the box-mean cross_track so it is correct on both half-swaths.
    """
    xs, ys = data["x"][sel], data["y"][sel]
    a_hat = r_hat = None
    ct_mean = np.nan

    if data["illum_time"] is not None:
        dx, dy = _plane_gradient(xs, ys, data["illum_time"][sel])
        nrm = np.hypot(dx, dy)
        if np.isfinite(nrm) and nrm > 0:
            a_hat = (dx / nrm, dy / nrm)

    if data["cross_track"] is not None:
        ct = data["cross_track"][sel]
        ct_mean = float(np.nanmean(ct))
        dx, dy = _plane_gradient(xs, ys, ct)
        nrm = np.hypot(dx, dy)
        if np.isfinite(nrm) and nrm > 0 and np.isfinite(ct_mean):
            s = 1.0 if ct_mean >= 0.0 else -1.0
            r_hat = (s * dx / nrm, s * dy / nrm)

    return {"a_hat": a_hat, "r_hat": r_hat, "cross_track_mean": ct_mean}


def box_geolocation_quality(data, sel):
    """Soft geolocation-quality diagnostics for retained pixels in one box."""
    q = data.get("geolocation_qual")
    if q is None:
        return {"layover_flag_frac": np.nan,
                "phase_unwrap_suspect_frac": np.nan,
                "n_quality_pixels": 0}
    qb = np.asarray(q[sel], dtype=np.uint64)
    if qb.size == 0:
        return {"layover_flag_frac": np.nan,
                "phase_unwrap_suspect_frac": np.nan,
                "n_quality_pixels": 0}
    return {
        "layover_flag_frac": float(np.mean((qb & np.uint64(QUAL_LAYOVER_BIT)) != 0)),
        "phase_unwrap_suspect_frac": float(np.mean((qb & np.uint64(QUAL_PHASE_UNWRAP_BIT)) != 0)),
        "n_quality_pixels": int(qb.size),
    }


# ============================================================
# BATHYMETRY AND CURRENT
# ============================================================

def load_bathymetry(bathy_file, proj):
    if not bathy_file:
        return None
    dsb = xr.open_dataset(bathy_file)
    lon = np.asarray(dsb[BATHY_LON_VAR].values, dtype=float)
    lat = np.asarray(dsb[BATHY_LAT_VAR].values, dtype=float)
    elev = np.squeeze(np.asarray(dsb[BATHY_ELEV_VAR].values, dtype=float))
    if elev.shape == (lon.size, lat.size) and lon.size != lat.size:
        elev = elev.T
    LON, LAT = np.meshgrid(lon, lat)
    X, Y = proj(LON, LAT)
    depth = elev if BATHY_POSITIVE_DOWN else -elev
    dsb.close()
    return {"x": X.ravel(), "y": Y.ravel(), "depth": depth.ravel()}


def box_depth_stats(bathy, box):
    if bathy is None:
        return {"mean": float(FALLBACK_DEPTH_M), "median": float(FALLBACK_DEPTH_M),
                "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    inside = ((bathy["x"] >= box["x_min"]) & (bathy["x"] <= box["x_max"]) &
              (bathy["y"] >= box["y_min"]) & (bathy["y"] <= box["y_max"]))
    d = bathy["depth"][inside]
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size == 0:
        return {"mean": float(FALLBACK_DEPTH_M), "median": float(FALLBACK_DEPTH_M),
                "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {"mean": float(np.mean(d)), "median": float(np.median(d)),
            "std": float(np.std(d)), "min": float(np.min(d)),
            "max": float(np.max(d)), "n": int(d.size)}


def _surface_level_index(dsc):
    if CUR_SURFACE_LEVEL is not None:
        return CUR_SURFACE_LEVEL
    if CUR_SIG_VAR in dsc.variables:
        sig = np.asarray(dsc[CUR_SIG_VAR].values, float)
        if sig.ndim == 1 and sig.size > 1 and np.all(np.isfinite(sig)):
            return int(np.argmin(np.abs(sig)))
    return -1


def _detect_time_dim(da):
    if CUR_TIME_VAR is not None and CUR_TIME_VAR in da.dims:
        return CUR_TIME_VAR
    for tname in ("time", "time_counter", "time_instant"):
        if tname in da.dims:
            return tname
    return None


def load_current_field(current_file, proj, target_time=None, lonlat_bbox=None):
    """
    MARS3D current, time-matched, cropped to the tile.

    Surface: top s-level. Profile (CURRENT_DEPTH_MODE='kirby_chen'): every
    level, with the s-coordinate metrics needed to place each level in z.
    UZ/VZ are on a regular lon/lat C-grid, i.e. eastward/northward.
    """
    current_file = resolve_current_file(current_file, target_time)
    if not current_file:
        return None
    dsc = xr.open_dataset(current_file, decode_times=True)
    u, v = dsc[CUR_U_VAR], dsc[CUR_V_VAR]
    tname = _detect_time_dim(u)
    time_delta_h = np.nan
    time_selected = None
    it = None
    if tname is not None:
        tvals = decode_time_da(dsc[tname]) if tname in dsc.variables else None
        if tvals is None and np.issubdtype(np.asarray(dsc[tname].values).dtype, np.datetime64):
            tvals = np.asarray(dsc[tname].values).astype("datetime64[ns]")
        if target_time is not None and tvals is not None:
            flat = np.asarray(tvals).astype("datetime64[ns]").ravel()
            delta = np.abs((flat - np.datetime64(target_time, "ns")) / np.timedelta64(1, "s"))
            it = int(np.nanargmin(delta))
            time_delta_h = float(delta[it] / 3600.0)
            time_selected = flat[it]
            if time_delta_h > CUR_MAX_TIME_DELTA_H and CURRENT_REQUIRE_TIME_MATCH:
                dsc.close()
                raise ValueError(f"nearest current time is {time_delta_h:.2f} h from SWOT")
        elif CUR_TIME_INDEX_FALLBACK is not None:
            it = int(CUR_TIME_INDEX_FALLBACK)
            print(f"[current] acquisition time unavailable -> explicit fallback index {it}")
        else:
            dsc.close()
            if CURRENT_REQUIRE_TIME_MATCH:
                raise ValueError("cannot time-match current field: no decoded SWOT/current time")
            return None
        u = u.isel({tname: it}); v = v.isel({tname: it})
    elif CURRENT_REQUIRE_TIME_MATCH:
        print("[current] field has no time dimension; treating it as a static field")

    for vn_, da_ in ((CUR_U_VAR, u), (CUR_V_VAR, v)):
        un = str(da_.attrs.get("units", "?"))
        print(f"[current] {vn_}: units={un!r} std_name={da_.attrs.get('standard_name', '')!r}")
        if un not in ("m/s", "m s-1", "m.s-1", "meter/second", "m s**-1", "?"):
            warnings.warn(f"[current] unexpected units {un!r} for {vn_}")

    # ---- crop every grid (u, v, rho) to the tile ---------------------------
    def crop_index(lon2d, lat2d):
        if lonlat_bbox is None:
            return np.ones(lon2d.shape, bool)
        lo0, lo1, la0, la1 = lonlat_bbox; mgn = CUR_CROP_MARGIN_DEG
        return ((lon2d >= lo0 - mgn) & (lon2d <= lo1 + mgn) &
                (lat2d >= la0 - mgn) & (lat2d <= la1 + mgn))

    def grid(lon_name, lat_name):
        lo = np.asarray(dsc[lon_name].values, float); la = np.asarray(dsc[lat_name].values, float)
        if lo.ndim == 1 and la.ndim == 1:
            lo, la = np.meshgrid(lo, la)
        return lo, la

    lon_u, lat_u = grid(CUR_LON_U, CUR_LAT_U)
    lon_v, lat_v = grid(CUR_LON_V, CUR_LAT_V)
    cu, cv = crop_index(lon_u, lat_u).ravel(), crop_index(lon_v, lat_v).ravel()

    ksurf = _surface_level_index(dsc)
    lname = next((n for n in ("level", "s_rho", "z", "depth") if n in u.dims), None)
    out_prof = None
    if lname is not None:
        nlev = u.sizes[lname]
        if CURRENT_DEPTH_MODE == "kirby_chen" and CUR_SIG_VAR in dsc.variables:
            s = np.asarray(dsc[CUR_SIG_VAR].values, float).ravel()
            if s.size == nlev and np.all(np.isfinite(s)) and s.min() >= -1.0001 and s.max() <= 0.0001:
                up = np.asarray(u.transpose(lname, ...).values, float).reshape(nlev, -1)[:, cu]
                vp = np.asarray(v.transpose(lname, ...).values, float).reshape(nlev, -1)[:, cv]
                a = float(dsc[CUR_THETA_VAR].values) if CUR_THETA_VAR in dsc.variables else 0.0
                b = float(dsc[CUR_B_VAR].values) if CUR_B_VAR in dsc.variables else 0.0
                Cs = (np.asarray(dsc[CUR_CS_VAR].values, float).ravel()
                      if CUR_CS_VAR in dsc.variables else s_coordinate_C(s, a, b))
                # rho-point metrics for z(s)
                rho = None
                if all(n in dsc.variables for n in (CUR_H0_VAR, CUR_LON_RHO, CUR_LAT_RHO)):
                    lon_r, lat_r = grid(CUR_LON_RHO, CUR_LAT_RHO)
                    cr = crop_index(lon_r, lat_r).ravel()
                    H0 = np.asarray(dsc[CUR_H0_VAR].values, float).ravel()[cr]
                    hc = (np.asarray(dsc[CUR_HC_VAR].values, float).ravel()[cr]
                          if CUR_HC_VAR in dsc.variables else np.zeros_like(H0))
                    eta = np.zeros_like(H0)
                    if CUR_ETA_VAR in dsc.variables:
                        e = dsc[CUR_ETA_VAR]
                        if tname is not None and tname in e.dims and it is not None:
                            e = e.isel({tname: it})
                        eta = np.asarray(np.squeeze(e.values), float).ravel()[cr]
                    xr_, yr_ = proj(lon_r.ravel()[cr], lat_r.ravel()[cr])
                    okr = np.isfinite(H0) & (H0 > 0) & np.isfinite(xr_)
                    rho = {"x": np.asarray(xr_)[okr], "y": np.asarray(yr_)[okr],
                           "H0": H0[okr], "hc": hc[okr],
                           "eta": np.where(np.isfinite(eta[okr]), eta[okr], 0.0)}
                out_prof = {"s": s, "Cs": Cs, "a": a, "b": b, "ue": up, "vn": vp, "rho": rho}
                print(f"[current] 3-D profile: {nlev} s-levels, theta={a:g}, b={b:g}, "
                      f"z from {'s-coordinate metrics' if rho else 'z = s*h (no H0 found)'}")
            else:
                print(f"[current] {CUR_SIG_VAR} is not an s-coordinate in [-1,0] -> surface only")
        u = u.isel({lname: ksurf}); v = v.isel({lname: ksurf})
    mode = "kirby_chen" if out_prof is not None else "surface"
    if CURRENT_DEPTH_MODE == "kirby_chen" and mode == "surface":
        print("[current] no usable vertical profile -> Doppler uses the SURFACE current")
    print(f"[current] file={Path(current_file).name} surface level={ksurf}, "
          f"time={time_selected}, dt={time_delta_h}, depth mode={mode}")

    ue = np.asarray(np.squeeze(u.values), dtype=float).ravel()[cu]
    vn = np.asarray(np.squeeze(v.values), dtype=float).ravel()[cv]
    dsc.close()
    xu, yu = proj(lon_u.ravel()[cu], lat_u.ravel()[cu]); xv, yv = proj(lon_v.ravel()[cv], lat_v.ravel()[cv])
    xu, yu, xv, yv = map(np.asarray, (xu, yu, xv, yv))
    mu = np.isfinite(ue) & np.isfinite(xu) & np.isfinite(yu)
    mv = np.isfinite(vn) & np.isfinite(xv) & np.isfinite(yv)
    out = {"xu": xu[mu], "yu": yu[mu], "ue": ue[mu],
           "xv": xv[mv], "yv": yv[mv], "vn": vn[mv],
           "time_selected": time_selected, "time_delta_h": time_delta_h,
           "mode": mode, "prof": None}
    if out_prof is not None:
        out_prof["ue"] = out_prof["ue"][:, mu]; out_prof["vn"] = out_prof["vn"][:, mv]
        out["prof"] = out_prof
    print(f"[current] {mu.sum()} u / {mv.sum()} v wet points in the cropped tile")
    return out


def _nearest_value(xc, yc, xs, ys, vs):
    if xs.size == 0: return np.nan
    return float(vs[np.argmin((xs-xc)**2 + (ys-yc)**2)])


def box_current_stats(current, box):
    if current is None:
        return {"U_e": 0.0, "U_n": 0.0, "U_e_std": np.nan, "U_n_std": np.nan,
                "speed_std": np.nan, "n_u": 0, "n_v": 0, "time_delta_h": np.nan,
                "profile": None}
    xc, yc = 0.5*(box["x_min"]+box["x_max"]), 0.5*(box["y_min"]+box["y_max"])

    def inside(x, y):
        return (x >= box["x_min"]) & (x <= box["x_max"]) & (y >= box["y_min"]) & (y <= box["y_max"])

    def nearest(x, y):
        return int(np.argmin((x-xc)**2 + (y-yc)**2)) if x.size else None

    bu = inside(current["xu"], current["yu"]); bv = inside(current["xv"], current["yv"])
    ue = current["ue"][bu]; vn = current["vn"][bv]
    ju = nearest(current["xu"], current["yu"]); jv = nearest(current["xv"], current["yv"])
    U_e = float(np.nanmean(ue)) if ue.size else (float(current["ue"][ju]) if ju is not None else np.nan)
    U_n = float(np.nanmean(vn)) if vn.size else (float(current["vn"][jv]) if jv is not None else np.nan)

    profile = None
    P = current.get("prof")
    if P is not None and ju is not None and jv is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pu = np.nanmean(P["ue"][:, bu], axis=1) if bu.any() else P["ue"][:, ju]
            pv = np.nanmean(P["vn"][:, bv], axis=1) if bv.any() else P["vn"][:, jv]
        s = P["s"]; Cs = P["Cs"]
        # interface s-values: half-way between centres, closed at -1 and 0
        s_w = np.r_[-1.0, 0.5*(s[1:] + s[:-1]), 0.0]
        C_w = s_coordinate_C(s_w, P["a"], P["b"])
        R = P.get("rho")
        if R is not None and R["x"].size:
            br = inside(R["x"], R["y"])
            if not br.any():
                br = np.zeros(R["x"].size, bool); br[nearest(R["x"], R["y"])] = True
            H0 = float(np.mean(R["H0"][br])); hc = float(np.mean(R["hc"][br]))
            eta = float(np.mean(R["eta"][br]))
            hc = min(max(hc, 0.0), H0)
            z_w = eta*(1+s_w) + hc*s_w + (H0-hc)*C_w          # bottom -> top
            z_c = eta*(1+s) + hc*s + (H0-hc)*Cs
            profile = {"z_w": z_w - eta, "z_c": z_c - eta, "D": H0 + eta,
                       "ue": pu, "vn": pv, "source": "s_coordinate"}
        else:
            profile = {"z_w": None, "z_c": None, "D": None, "s_w": s_w,
                       "ue": pu, "vn": pv, "source": "sigma_times_h"}
    return {"U_e": U_e, "U_n": U_n,
            "U_e_std": float(np.nanstd(ue)) if ue.size > 1 else np.nan,
            "U_n_std": float(np.nanstd(vn)) if vn.size > 1 else np.nan,
            "speed_std": float(np.hypot(np.nanstd(ue), np.nanstd(vn))) if ue.size > 1 and vn.size > 1 else np.nan,
            "n_u": int(ue.size), "n_v": int(vn.size),
            "time_delta_h": current.get("time_delta_h", np.nan),
            "profile": profile}


def _first_name(ds, candidates):
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    return None


def ww3_direction_convention(da):
    """('from'|'towards', source) from a direction variable's metadata."""
    if WW3_DIR_CONVENTION in ("from", "towards"):
        return WW3_DIR_CONVENTION, "pinned"
    txt = " ".join(str(da.attrs.get(a, "")) for a in
                   ("standard_name", "long_name", "convention", "comment",
                    "direction_reference", "description")).lower()
    if "to_direction" in txt or "going to" in txt or "toward" in txt:
        return "towards", "metadata"
    if ("from_direction" in txt or "coming from" in txt or "nautical" in txt
            or "direction from" in txt):
        return "from", "metadata"
    return WW3_DIR_CONVENTION_FALLBACK, "assumed"


def load_ww3_field(path, proj, target_time=None):
    """MARC/WW3 field on a regular or unstructured grid, time-matched to SWOT."""
    if not path:
        return None
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as e:
        print(f"[ww3] cannot open {path}: {e}"); return None
    tname = _first_name(ds, WW3_TIME_CANDIDATES)
    dt_h = np.nan; tsel = None
    if tname is not None and tname in ds.dims and target_time is not None:
        tv = decode_time_da(ds[tname])
        if tv is None and np.issubdtype(np.asarray(ds[tname].values).dtype, np.datetime64):
            tv = np.asarray(ds[tname].values).astype("datetime64[ns]")
        if tv is not None:
            flat = np.asarray(tv).astype("datetime64[ns]").ravel()
            dd = np.abs((flat - np.datetime64(target_time, "ns")) / np.timedelta64(1, "s"))
            it = int(np.nanargmin(dd)); dt_h = float(dd[it] / 3600.0); tsel = flat[it]
            if dt_h > WW3_MAX_TIME_DELTA_H:
                print(f"[ww3] nearest time is {dt_h:.2f} h away -> model disabled")
                ds.close(); return None
            ds = ds.isel({tname: it})
    lon_n, lat_n = _first_name(ds, WW3_LON_CANDIDATES), _first_name(ds, WW3_LAT_CANDIDATES)
    dir_n = _first_name(ds, WW3_DIR_CANDIDATES)
    fp_n, tp_n = _first_name(ds, WW3_FP_CANDIDATES), _first_name(ds, WW3_TP_CANDIDATES)
    hs_n = _first_name(ds, WW3_HS_CANDIDATES)
    if lon_n is None or lat_n is None or dir_n is None or (fp_n is None and tp_n is None):
        print(f"[ww3] missing lon/lat/dir/fp in {path} -> model disabled"); ds.close(); return None

    lon_da, lat_da = ds[lon_n], ds[lat_n]
    unstructured = (lon_da.ndim == 1 and lat_da.ndim == 1 and lon_da.dims == lat_da.dims)
    if unstructured:
        lon = np.asarray(lon_da.values, float); lat = np.asarray(lat_da.values, float)
        space_dims = lon_da.dims
    elif lon_da.ndim == 1 and lat_da.ndim == 1:
        lon, lat = np.meshgrid(np.asarray(lon_da.values, float), np.asarray(lat_da.values, float))
        space_dims = (lat_da.dims[0], lon_da.dims[0])
    else:
        lon = np.asarray(lon_da.values, float); lat = np.asarray(lat_da.values, float)
        space_dims = lon_da.dims

    def field(name):
        da = ds[name].squeeze(drop=True)
        if set(space_dims).issubset(da.dims):
            da = da.transpose(*space_dims)
        arr = np.asarray(da.values, float)
        if arr.size != lon.size:
            raise ValueError(f"[ww3] {name} shape {arr.shape} vs grid {lon.shape}")
        return arr.ravel()

    conv, conv_src = ww3_direction_convention(ds[dir_n])
    print(f"[ww3] {'unstructured' if unstructured else 'regular'} grid, time={tsel} "
          f"(dt={dt_h:.2f} h), dir={dir_n} "
          f"std_name={ds[dir_n].attrs.get('standard_name', '')!r} "
          f"long_name={ds[dir_n].attrs.get('long_name', '')!r} -> '{conv}' ({conv_src})")
    lobe_allowed = conv_src != "assumed" or MODEL_LOBE_ALLOW_ASSUMED
    if MODEL_LOBE_MODE != "off" and not lobe_allowed:
        print("[ww3] WARNING direction convention not confirmed by metadata -> "
              "model-assisted lobe choice DISABLED. Pin WW3_DIR_CONVENTION.")
    try:
        if fp_n is not None:
            fp = field(fp_n); fp = np.where(fp > 0, fp, np.nan)
        else:
            tp = field(tp_n); fp = np.where(tp > 0, 1.0 / tp, np.nan)
        dirs = field(dir_n) % 360.0
        hs = field(hs_n) if hs_n is not None else np.full(lon.size, np.nan)
    except ValueError as e:
        print(e); ds.close(); return None
    ds.close()
    X, Y = proj(lon.ravel(), lat.ravel())
    X, Y = np.asarray(X), np.asarray(Y)
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(fp) & np.isfinite(dirs)
    return {"x": X[ok], "y": Y[ok], "fp": fp[ok], "Dir": dirs[ok], "Hs": hs[ok],
            "dt_h": dt_h, "time": tsel, "dir_convention": conv,
            "dir_convention_source": conv_src, "lobe_allowed": bool(lobe_allowed)}


def box_ww3_stats(ww3, box, depth, cur=None):
    keys = ("ww3_Hs_m", "ww3_fp_hz", "ww3_Tp_s", "ww3_dir_degN", "ww3_dir_to_degN",
            "ww3_dir_spread_deg", "ww3_lambda_p_m", "ww3_Hs_over_Lp",
            "ww3_time_delta_h", "ww3_n_points")
    empty = {k: np.nan for k in keys}
    empty.update({"ww3_lobe_allowed": False, "ww3_dir_convention": ""})
    if ww3 is None or ww3["x"].size == 0:
        return empty
    m = ((ww3["x"] >= box["x_min"]) & (ww3["x"] <= box["x_max"]) &
         (ww3["y"] >= box["y_min"]) & (ww3["y"] <= box["y_max"]))
    if not np.any(m):
        xc, yc = 0.5*(box["x_min"]+box["x_max"]), 0.5*(box["y_min"]+box["y_max"])
        d2 = (ww3["x"]-xc)**2 + (ww3["y"]-yc)**2; j = int(np.argmin(d2))
        if np.sqrt(d2[j]) > WW3_MAX_NEAREST_M:
            return empty
        m = np.zeros(ww3["x"].size, bool); m[j] = True
    fp = float(np.nanmean(ww3["fp"][m])); hs = float(np.nanmean(ww3["Hs"][m]))
    dirs = ww3["Dir"][m]
    d_raw = circular_mean_deg(dirs)
    d_to = (d_raw + 180.0) % 360.0 if ww3["dir_convention"] == "from" else d_raw
    d_rep = (d_to + 180.0) % 360.0 if DIR_CONVENTION == "from" else d_to
    if WW3_FP_IS_ABSOLUTE and cur is not None and np.isfinite(d_to):
        ux, uy = direction_to_uv(d_to)
        k = k_from_absolute_frequency(fp, depth, cur["U_e"]*ux + cur["U_n"]*uy)
    else:
        k = wavenumber_from_period(1.0/fp if fp > 0 else np.nan, depth)
    lp = 2*np.pi/k if np.isfinite(k) and k > 0 else np.nan
    return {"ww3_Hs_m": hs, "ww3_fp_hz": fp,
            "ww3_Tp_s": float(1.0/fp) if fp > 0 else np.nan,
            "ww3_dir_degN": float(d_rep), "ww3_dir_to_degN": float(d_to),
            "ww3_dir_spread_deg": circular_sigma_deg(dirs),
            "ww3_lambda_p_m": float(lp),
            "ww3_Hs_over_Lp": float(hs/lp) if np.isfinite(lp) and lp > 0 else np.nan,
            "ww3_time_delta_h": ww3.get("dt_h", np.nan), "ww3_n_points": int(m.sum()),
            "ww3_lobe_allowed": bool(ww3.get("lobe_allowed", False)),
            "ww3_dir_convention": ww3.get("dir_convention", "")}


# ============================================================
# BOXES AND GRIDDING
# ============================================================

def make_tile_boxes(data):
    x, y = data["x"], data["y"]
    x0 = np.floor(np.nanmin(x) / BOX_SIZE) * BOX_SIZE
    x1 = np.ceil(np.nanmax(x) / BOX_SIZE) * BOX_SIZE
    y0 = np.floor(np.nanmin(y) / BOX_SIZE) * BOX_SIZE
    y1 = np.ceil(np.nanmax(y) / BOX_SIZE) * BOX_SIZE

    boxes = []
    idx = 0
    for xx in np.arange(x0, x1, BOX_STRIDE):
        for yy in np.arange(y0, y1, BOX_STRIDE):
            inside = ((x >= xx) & (x <= xx + BOX_SIZE) &
                      (y >= yy) & (y <= yy + BOX_SIZE))
            n_raw = int(np.sum(inside))
            if n_raw < MIN_RAW_PIXELS_BOX:
                continue
            idx += 1
            boxes.append({
                "box": f"B{idx:04d}",
                "grid_ix": int(round(xx / BOX_SIZE)),
                "grid_iy": int(round(yy / BOX_SIZE)),
                "x_min": float(xx), "x_max": float(xx + BOX_SIZE),
                "y_min": float(yy), "y_max": float(yy + BOX_SIZE),
                "n_raw_pixels": n_raw,
            })
    return boxes


def choose_res(ct_mean, h_sat):
    """Cell size from the local postings; see the irregular-sampling note."""
    if RES_MODE == "fixed" or not np.isfinite(ct_mean):
        return float(RES_FIXED), float(across_track_posting(
            ct_mean if np.isfinite(ct_mean) else CROSS_TRACK_MIN, h_sat))
    d_ac = float(across_track_posting(ct_mean, h_sat))
    res = RES_ADAPT_FACTOR * max(ALONG_TRACK_POSTING, d_ac)
    res = float(np.clip(round(res / RES_ROUND) * RES_ROUND, RES_MIN, RES_MAX))
    return res, d_ac


def _grid_values(xb, yb, values, box, res):
    xe = np.arange(box["x_min"], box["x_max"] + res, res)
    ye = np.arange(box["y_min"], box["y_max"] + res, res)
    cnt, _, _ = np.histogram2d(yb, xb, bins=[ye, xe])
    ss, _, _ = np.histogram2d(yb, xb, bins=[ye, xe], weights=values)
    out = np.full_like(ss, np.nan, dtype=float)
    ok = cnt >= MIN_PIXELS_PER_CELL
    out[ok] = ss[ok] / cnt[ok]
    return out, cnt


def grid_one_box(data, box, res):
    x, y = data["x"], data["y"]
    inside = ((x >= box["x_min"]) & (x <= box["x_max"]) &
              (y >= box["y_min"]) & (y <= box["y_max"]))
    xb, yb = x[inside], y[inside]
    sshb, sigb = data["ssh"][inside], data["sig0"][inside]
    if xb.size < MIN_RAW_PIXELS_BOX:
        return None
    ssh_g, cnt = _grid_values(xb, yb, sshb, box, res)
    sig_g, _ = _grid_values(xb, yb, sigb, box, res)
    valid_frac_box = float(np.mean(np.isfinite(ssh_g) & np.isfinite(sig_g)))
    if valid_frac_box < MIN_VALID_FRAC_BOX:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        sig_db = 10.0*np.log10(np.where(sig_g>0,sig_g,np.nan))
    diag = {"sig0_db_med":float(np.nanmedian(sig_db)),
            "ssh_std_short":_short_scale_std(ssh_g),
            "sig0_std_short":_short_scale_std(sig_db),
            "gap_leakage":_gap_leakage(np.isfinite(ssh_g)&np.isfinite(sig_g),res),
            "mean_raw_per_cell":float(np.nanmean(cnt[cnt>0])) if np.any(cnt>0) else np.nan}
    return ssh_g, sig_g, valid_frac_box, inside, diag


def _boxcar2d_nan(Z, n):
    valid = np.isfinite(Z).astype(float)
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    k = np.ones(n) / n
    sm = lambda A, ax: np.apply_along_axis(
        lambda m: np.convolve(m, k, mode="same"), ax, A)
    num = sm(sm(Zf, 0), 1)
    den = sm(sm(valid, 0), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.25] = np.nan
    return out


def _short_scale_std(Z, n=5):
    """Std of the residual after a short boxcar: a sub-wave-band noise proxy."""
    if Z.size == 0 or not np.isfinite(Z).any():
        return np.nan
    return float(np.nanstd(Z - _boxcar2d_nan(Z, n)))


def _gap_leakage(mask, res):
    """
    Spectral power of the validity mask landing INSIDE the wave band, relative
    to its DC term.

    Zero-filled cells make the transform h_obs^ = M^ * h^, so what matters is
    how much mask power sits where the swell is, not how much sits outside a
    central lobe. The previous version removed the mean first, which for a
    nearly full mask leaves scattered impulses whose power is almost entirely
    off-centre; it reported ~0.99 for boxes whose windows were 100% valid.

    Randomly scattered holes give ~1e-6. Stripe-shaped gaps give orders of
    magnitude more. Calibrate GAP_LEAKAGE_MAX from the printed distribution.
    """
    m = mask.astype(float)
    if m.mean() >= 0.999:
        return 0.0

    ny, nx = m.shape
    M = np.abs(fftshift(fft2(m))) ** 2

    kyv = fftshift(2.0 * np.pi * fftfreq(ny, d=res))
    kxv = fftshift(2.0 * np.pi * fftfreq(nx, d=res))
    KXm, KYm = np.meshgrid(kxv, kyv)
    kk = np.hypot(KXm, KYm)

    bnd = ((kk >= 2.0 * np.pi / LAMBDA_MAX) &
           (kk <= 2.0 * np.pi / LAMBDA_MIN))

    dc = M[ny // 2, nx // 2]
    if dc <= 0 or not bnd.any():
        return np.nan

    return float(M[bnd].sum() / dc)


# ============================================================
# WELCH WINDOWS + RADON / FOURIER-SLICE SPECTRA
# ============================================================

def compute_welch_spectra(ssh_g, sig_g, res, lwin=LWIN, min_windows=MIN_N_WINDOWS):
    """
    FFT-v4.2 preprocessing with a Radon/Fourier-slice retrieval core.

    Unchanged from FFT v4.2 before the estimator:
      * adaptive gridded SSH and linear sigma0 arrive from grid_one_box;
      * the same valid-window threshold is used;
      * sigma0 is normalized as delta-sigma0 / mean-sigma0;
      * SSH and sigma0 are plane-detrended using valid cells only;
      * the same 2-D Hann window and Welch overlap are used.

    The SSH energy estimator is then Radon + 1-D Fourier slice. The original
    Cartesian FFTs are retained ONLY as auxiliary cross-spectral diagnostics
    for the unchanged velocity-bunching/tilt/coherence post-processing.
    """
    ny, nx = ssh_g.shape
    Nwin = int(round(lwin / res))
    step = max(1, Nwin - int(round(OVERLAP_FRAC * Nwin)))
    Nfft = int(np.ceil(Nwin * (1.0 + ZPAD_FRAC)))
    Nfft += Nfft % 2
    if Nwin > ny or Nwin > nx:
        return None

    win = hann(Nwin)
    win2d = win[:, None] * win[None, :]
    Uwin = np.mean(win2d ** 2)
    norm_fft = res * res / (Nwin * Nwin * Uwin) / (2 * np.pi) ** 2

    # Cartesian FFT grid retained for cross-spectrum, tilt diagnostic and
    # magnitude-squared coherence exactly as in FFT v4.2.
    kx = fftshift(2 * np.pi * fftfreq(Nfft, d=res))
    KX, KY = np.meshgrid(kx, kx)
    kmag = np.hypot(KX, KY)
    lam_grid = 2 * np.pi / np.maximum(kmag, 1e-12)
    theta_to_grid = (90.0 - (np.degrees(np.arctan2(KY, KX)) % 360.0)) % 360.0
    lambda_min_local = max(LAMBDA_MIN, NYQUIST_SAFETY * res)
    band_fft = ((kmag >= 2 * np.pi / LAMBDA_MAX) &
                (kmag <= 2 * np.pi / lambda_min_local))

    radon_thetas = np.arange(0.0, 180.0, RADON_ANGLE_STEP_DEG)
    if radon_thetas.size < 2:
        raise ValueError("RADON_ANGLE_STEP_DEG must give at least two angles")

    R_list, E_fft_list, G_fft_list, C_list, vfrac = [], [], [], [], []
    radon_freqs = None
    hann_rho = None
    n_pad = None

    for i in range(0, ny - Nwin + 1, step):
        for j in range(0, nx - Nwin + 1, step):
            sp = ssh_g[i:i + Nwin, j:j + Nwin]
            gp = sig_g[i:i + Nwin, j:j + Nwin]
            valid = np.isfinite(sp) & np.isfinite(gp) & (gp > 0)
            if valid.mean() < MIN_VALID_FRAC_WINDOW:
                continue

            gmean = float(np.nanmean(gp[valid]))
            if not np.isfinite(gmean) or gmean <= 0:
                continue
            gp_norm = gp / gmean - 1.0

            # EXACT FFT-v4.2 detrending and 2-D tapering.
            sp_d = detrend_plane_valid(np.where(valid, sp, 0.0), valid)
            gp_d = detrend_plane_valid(np.where(valid, gp_norm, 0.0), valid)
            hp = sp_d * win2d
            gpw = gp_d * win2d

            # Radon/Fourier-slice estimator. circle=False avoids introducing a
            # circular spatial mask that was not present in FFT v4.2 preprocessing.
            sino = radon(hp, theta=radon_thetas, circle=False, preserve_range=True)
            if hann_rho is None:
                n_rho = sino.shape[0]
                n_pad = max(n_rho, int(np.ceil(n_rho * RADON_SLICE_ZPAD_FACTOR)))
                hann_rho = hann(n_rho)
                radon_freqs = np.fft.rfftfreq(n_pad, d=res)
                U_rho = np.mean(hann_rho ** 2)
                norm_radon = res / max(n_rho * U_rho, 1e-30)

            prof = sino - np.mean(sino, axis=0, keepdims=True)
            Fr = np.fft.rfft(prof * hann_rho[:, None], n=n_pad, axis=0)
            R_list.append(norm_radon * np.abs(Fr) ** 2)

            # Auxiliary FFT cross-spectrum: unchanged FFT-v4.2 calculation.
            Fs = fftshift(fft2(hp, s=(Nfft, Nfft)))
            Fg = fftshift(fft2(gpw, s=(Nfft, Nfft)))
            E_fft_list.append(norm_fft * np.abs(Fs) ** 2)
            G_fft_list.append(norm_fft * np.abs(Fg) ** 2)
            C_list.append(norm_fft * Fs * np.conj(Fg))
            vfrac.append(float(valid.mean()))

    if len(R_list) < min_windows:
        return None

    n_win = len(R_list)
    n_eff = eff_n_2d(n_win, win, step) if APPLY_OVERLAP_CORR else float(n_win)
    n_eff = max(float(n_eff) * N_EFF_FACTOR, 1.0)

    RF = np.broadcast_to(radon_freqs[:, None],
                         (radon_freqs.size, radon_thetas.size)).copy()
    # For skimage.radon, a wave propagating at bearing b has the strongest
    # Fourier slice at projection angle theta = b - 90 deg (mod 180).
    RTH_TO = np.broadcast_to(((radon_thetas + 90.0) % 180.0)[None, :], RF.shape).copy()
    RK = 2 * np.pi * RF
    rth = np.deg2rad(RTH_TO)
    RKX = RK * np.sin(rth)
    RKY = RK * np.cos(rth)
    with np.errstate(divide="ignore", invalid="ignore"):
        RLAM = np.where(RF > 0, 1.0 / RF, np.inf)

    radon_band = ((RF >= 1.0 / LAMBDA_MAX) &
                  (RF <= 1.0 / lambda_min_local))
    df = float(radon_freqs[1] - radon_freqs[0]) if radon_freqs.size > 1 else np.nan
    dtheta_rad = float(np.deg2rad(RADON_ANGLE_STEP_DEG))

    # Pre-map every Cartesian FFT bin to its nearest Radon sample. This lets
    # each Radon partition reuse the FFT-v4.2 cross-spectrum on the same
    # physical wave system for lobe resolution and coherence.
    f_fft = kmag / (2 * np.pi)
    if np.isfinite(df) and df > 0:
        fft_ri = np.clip(np.rint(f_fft / df).astype(int), 0, radon_freqs.size - 1)
    else:
        fft_ri = np.zeros_like(kmag, dtype=int)
    bearing_mod = theta_to_grid % 180.0
    theta_r = (bearing_mod - 90.0) % 180.0
    fft_rj = np.rint(theta_r / RADON_ANGLE_STEP_DEG).astype(int) % radon_thetas.size

    return {
        "estimator": "radon_fourier_slice",
        "E_arr": np.stack(R_list),              # Radon/Fourier-slice SSH power
        "E_arr_fft": np.stack(E_fft_list),      # unchanged FFT-v4.2 auxiliary
        "G_arr_fft": np.stack(G_fft_list),
        "C_arr": np.stack(C_list),              # unchanged FFT-v4.2 cross-spectrum
        "KX": KX, "KY": KY, "kmag": kmag, "lam_grid": lam_grid,
        "theta_to_grid": theta_to_grid, "band": band_fft,
        "radon_freqs": radon_freqs, "radon_thetas": radon_thetas,
        "RF": RF, "RK": RK, "RKX": RKX, "RKY": RKY,
        "radon_theta_to": RTH_TO, "radon_lam_grid": RLAM,
        "radon_band": radon_band, "radon_df": df, "radon_dtheta": dtheta_rad,
        "fft_radon_i": fft_ri, "fft_radon_j": fft_rj,
        "nfft": Nfft, "n_win": n_win, "n_eff": n_eff,
        "res": res, "lwin": float(lwin), "lambda_min_local": lambda_min_local,
        "dk": float(2 * np.pi / (Nfft * res)),
        "radon_dk": float(2 * np.pi * df) if np.isfinite(df) else np.nan,
        "valid_frac_win_med": float(np.median(vfrac)),
    }


def coherence_map(spec):
    """Unchanged FFT-v4.2 magnitude-squared coherence on the Cartesian grid."""
    E = np.mean(spec["E_arr_fft"], axis=0)
    G = np.mean(spec["G_arr_fft"], axis=0)
    C = np.mean(spec["C_arr"], axis=0)
    return np.clip(np.abs(C) ** 2 / np.maximum(E * G, 1e-30), 0.0, 1.0)


def coherence_significance(n_eff, alpha=COHERENCE_ALPHA):
    """Approximate MSC null threshold for n_eff independent Welch averages."""
    if not np.isfinite(n_eff) or n_eff <= 1.0:
        return np.nan
    return float(1.0 - alpha ** (1.0 / (n_eff - 1.0)))


# ============================================================
# NOISE MODEL
# ============================================================

def denoise(E, spec, resp_pow):
    """FFT-v4.2 response-shaped noise subtraction on the Radon spectral grid."""
    if not SUBTRACT_NOISE_FLOOR:
        return E, np.nan

    lam = spec["radon_lam_grid"]
    nb = ((lam >= NOISE_LAMBDA_LO_FACTOR * spec["res"]) &
          (lam < spec.get("lambda_min_local", LAMBDA_MIN)))
    if not np.any(nb):
        return E, np.nan

    # Skip bins next to the boxcar zero (lambda = 3d), where E/|H|^2 blows up.
    nb = nb & (resp_pow >= NOISE_RESP_MIN)
    if not np.any(nb):
        return E, np.nan
    n0 = float(np.nanmedian(E[nb] / np.maximum(resp_pow[nb], 1e-6)))
    if not np.isfinite(n0) or n0 <= 0:
        return E, np.nan
    return np.clip(E - n0 * resp_pow, 0.0, None), n0


# ============================================================
# RADON PARTITION EXTRACTION + FFT-v4.2 180 DEG RESOLUTION
# ============================================================

def _local_peak_seeds(Eb, band, spec):
    # Frequency is bounded, while projection angle is periodic over 180 deg.
    mx = ndimage.maximum_filter(Eb, size=(3, 3), mode=("constant", "wrap"))
    gmax = float(np.nanmax(Eb[band])) if np.any(band) else 0.0
    cand = band & (Eb == mx) & (Eb >= PARTITION_SEED_FRAC * gmax)
    ij = np.argwhere(cand)
    if ij.size == 0:
        return []
    ij = sorted(ij, key=lambda q: Eb[tuple(q)], reverse=True)
    kept = []
    minsep = PARTITION_MIN_SEP_BINS * spec["radon_dk"]
    for q in ij:
        i, j = map(int, q)
        kx, ky = spec["RKX"][i, j], spec["RKY"][i, j]
        if all(np.hypot(kx - spec["RKX"][ii, jj],
                        ky - spec["RKY"][ii, jj]) >= minsep
               for ii, jj in kept):
            kept.append((i, j))
    return kept


def _periodic_seed_component(grow, seed_i, seed_j):
    """Connected component containing a seed, with the angle axis periodic."""
    ntheta = grow.shape[1]
    shift = ntheta // 2 - seed_j
    gr = np.roll(grow, shift, axis=1)
    sj = (seed_j + shift) % ntheta
    lab, _ = ndimage.label(gr, ndimage.generate_binary_structure(2, 2))
    L = int(lab[seed_i, sj]) if gr[seed_i, sj] else 0
    if L == 0:
        return np.zeros_like(grow, dtype=bool)
    return np.roll(lab == L, -shift, axis=1)


def _radon_partition_to_fft_mask(m, spec, kx_c, ky_c):
    mapped = m[spec["fft_radon_i"], spec["fft_radon_j"]]
    # Keep only one Hermitian lobe, matching FFT-v4.2 partition handling.
    hemi = (spec["KX"] * kx_c + spec["KY"] * ky_c) > 0
    out = mapped & hemi & spec["band"]
    if np.any(out):
        return out

    # Very narrow Radon partitions can fall between Cartesian bins. Preserve
    # the same cross-spectral diagnostic by selecting the nearest valid bin.
    d2 = (spec["KX"] - kx_c) ** 2 + (spec["KY"] - ky_c) ** 2
    d2 = np.where(hemi & spec["band"], d2, np.inf)
    if np.isfinite(d2).any():
        out = np.zeros_like(spec["band"], dtype=bool)
        out[np.unravel_index(np.argmin(d2), d2.shape)] = True
    return out


def extract_partitions(E_dn, C, spec, resp_pow, geom_grids, resp_min=None):
    """
    Seed/grow Radon Fourier-slice partitions and apply the unchanged FFT-v4.2
    cross-spectrum lobe-resolution rules to each physical partition.
    """
    rmin = RESPONSE_MIN_MASK if resp_min is None else resp_min
    band = spec["radon_band"] & (resp_pow >= rmin)
    if not np.any(band):
        return []
    Eb = np.where(band, E_dn, 0.0)
    gmax = float(np.max(Eb))
    if gmax <= 0:
        return []

    seeds = _local_peak_seeds(Eb, band, spec)
    if not seeds:
        return []

    claimed = np.zeros(Eb.shape, bool)
    parts = []
    total_energy_density = float(np.sum(Eb[band] * spec["RK"][band]))

    for ii, jj in seeds:
        seed_val = float(Eb[ii, jj])
        grow = band & (Eb >= PARTITION_FRAC * seed_val) & (~claimed)
        m = _periodic_seed_component(grow, ii, jj)
        if m.sum() < PARTITION_MIN_BINS:
            continue

        # Polar spectral integration needs the Jacobian k dk dtheta.
        w = Eb[m] * spec["RK"][m]
        wsum = float(np.sum(w))
        if wsum <= 0:
            continue
        if total_energy_density > 0 and wsum < PARTITION_MIN_ENERGY_FRAC * total_energy_density:
            continue
        claimed |= m

        # The projection-angle axis is periodic modulo 180 deg and that seam is
        # exactly the k -> -k flip, so a partition allowed to cross it (which
        # _periodic_seed_component deliberately permits) contains bins whose
        # canonical vectors are nearly antiparallel to the seed. Averaging them
        # unreflected makes the vector centroid cancel: a 200 m wave at bearing
        # 0 returned 1714 m, at 1 deg 718 m, at 5 deg 250 m. Reflect every bin
        # onto the seed's lobe first. The doubled-angle spread below is already
        # invariant to this reflection and is left unchanged.
        kxs, kys = spec["RKX"][ii, jj], spec["RKY"][ii, jj]
        kx_m, ky_m = spec["RKX"][m], spec["RKY"][m]
        seam_flip = (kx_m * kxs + ky_m * kys) < 0
        kx_m = np.where(seam_flip, -kx_m, kx_m)
        ky_m = np.where(seam_flip, -ky_m, ky_m)

        kx_c = float(np.sum(kx_m * w) / wsum)
        ky_c = float(np.sum(ky_m * w) / wsum)
        kmod = float(np.hypot(kx_c, ky_c))
        if kmod <= 0:
            continue

        fft_mask = _radon_partition_to_fft_mask(m, spec, kx_c, ky_c)
        Csel = C[fft_mask]
        sum_re = float(np.sum(np.real(Csel))) if Csel.size else 0.0
        sum_im = float(np.sum(np.imag(Csel))) if Csel.size else 0.0
        sum_abs = float(np.sum(np.abs(Csel))) if Csel.size else 0.0
        conf = abs(sum_re) / sum_abs if sum_abs > 0 else np.nan

        resolved = False
        method = "unresolved"
        align = np.nan
        vote = np.nan
        if geom_grids["a_hat"] is not None:
            kaz_c = kx_c * geom_grids["a_hat"][0] + ky_c * geom_grids["a_hat"][1]
            align = abs(kaz_c) / kmod
            if np.isfinite(conf) and sum_re != 0:
                vote = float(np.sign(VB_SIGN * sum_re))
                if np.sign(kaz_c) != vote:
                    kx_c, ky_c = -kx_c, -ky_c
                    kaz_c = -kaz_c
                    # Re C is even under k -> -k but Im C is odd, and sum_im was
                    # accumulated on the pre-flip lobe. Without this the tilt
                    # diagnostic inverts in exactly the boxes where the flip
                    # fires, driving its agreement rate towards 50 % whatever
                    # the data quality.
                    sum_im = -sum_im
                resolved = ((conf >= LOBE_MIN_CONF) and
                            (align >= LOBE_MIN_ALIGN))
                method = "vb" if resolved else "vb_weak"

        kaz_c = (kx_c * geom_grids["a_hat"][0] + ky_c * geom_grids["a_hat"][1]
                 if geom_grids["a_hat"] else np.nan)
        kr_c = (kx_c * geom_grids["r_hat"][0] + ky_c * geom_grids["r_hat"][1]
                if geom_grids["r_hat"] else np.nan)

        theta_tilt = np.nan
        if geom_grids["r_hat"] is not None and sum_im != 0 and np.isfinite(kr_c):
            s_t = np.sign(TILT_SIGN * sum_im)
            kx_t, ky_t = ((kx_c, ky_c) if np.sign(kr_c) == s_t
                          else (-kx_c, -ky_c))
            theta_tilt = k_to_bearing(kx_t, ky_t)

        lam = 2 * np.pi / kmod
        th = np.deg2rad(spec["radon_theta_to"][m])
        Rw = np.hypot(np.sum(w * np.cos(2 * th)),
                      np.sum(w * np.sin(2 * th))) / wsum
        spread = float(np.rad2deg(np.sqrt(max(-2 * np.log(max(Rw, 1e-12)), 0.0))) / 2.0)

        energy = float(wsum * spec["radon_dk"] * spec["radon_dtheta"])
        parts.append({
            "kx": kx_c, "ky": ky_c, "k": kmod, "lambda_m": float(lam),
            "theta_to": k_to_bearing(kx_c, ky_c), "theta_tilt": theta_tilt,
            "k_along": float(kaz_c) if np.isfinite(kaz_c) else np.nan,
            "k_across": float(kr_c) if np.isfinite(kr_c) else np.nan,
            "energy": energy, "n_bins": int(m.sum()),
            "ang_spread_deg": spread,
            "lobe_conf": float(conf) if np.isfinite(conf) else np.nan,
            "lobe_align": float(align) if np.isfinite(align) else np.nan,
            "lobe_resolved": bool(resolved), "lobe_method": method,
            "vb_vote": vote,
            "resp_pow": float(np.average(resp_pow[m], weights=w)),
            "mask": m, "fft_mask": fft_mask, "seed_value": seed_val,
        })

    parts.sort(key=lambda p: -p["energy"])
    return parts


def coherence_at(parts, spec, i):
    """FFT-v4.2 MSC averaged over the Cartesian bins mapped to a Radon partition."""
    m = parts[i]["fft_mask"]
    coh = coherence_map(spec)
    E = np.mean(spec["E_arr_fft"], axis=0)
    w = np.maximum(E[m], 0.0)
    return float(np.average(coh[m], weights=w)) if np.sum(w) > 0 else np.nan


def flag_nonlinear_artifacts(parts):
    """Flag candidate 2k/3k harmonics and vector k_i +/- k_j sidebands."""
    for p in parts:
        p["possible_harmonic"] = False
        p["harmonic_order"] = np.nan
        p["possible_sideband"] = False
        p["artifact_type"] = "none"
    if not parts:
        return
    p0 = parts[0]
    for p in parts[1:]:
        dth = float(axial_difference_deg(p["theta_to"], p0["theta_to"]))
        ratio = p["k"] / p0["k"] if p0["k"] > 0 else np.nan
        for order in HARMONIC_ORDERS:
            if (np.isfinite(ratio) and abs(ratio - order) <= HARMONIC_K_TOL
                    and dth <= HARMONIC_ANG_TOL):
                p["possible_harmonic"] = True
                p["harmonic_order"] = float(order)
                p["artifact_type"] = f"{order}k_harmonic"
                break
    for j in range(2, len(parts)):
        pj = parts[j]
        kj = np.array([pj["kx"], pj["ky"]])
        for a in range(j):
            for b in range(a + 1, j):
                ka = np.array([parts[a]["kx"], parts[a]["ky"]])
                kb = np.array([parts[b]["kx"], parts[b]["ky"]])
                for name, kv in (("sum", ka + kb), ("difference", ka - kb),
                                 ("difference", kb - ka)):
                    km = np.hypot(*kv)
                    if km <= 0:
                        continue
                    rel = min(np.hypot(*(kj - kv)), np.hypot(*(kj + kv))) / km
                    if rel <= SIDEBAND_K_TOL:
                        pj["possible_sideband"] = True
                        pj["artifact_type"] = f"{name}_sideband"
                        break
                if pj["possible_sideband"]:
                    break
            if pj["possible_sideband"]:
                break


# ============================================================
# TILT ANTISYMMETRY DIAGNOSTIC
# ============================================================

class TiltDiagnostic:
    """
    Scene-level confirmation of the cross-spectrum sign conventions.

    Yu et al. Sec. V-B require Im<h sigma0*> < 0 wherever k.r_hat > 0. This
    accumulates the |C|-weighted fraction of band bins that satisfy it, plus a
    2D map of mean Im C in the (k.r_hat, k.a_hat) plane. If the antisymmetry
    is absent, VB_SIGN is equally suspect and every direction is unreliable.
    """

    def __init__(self):
        e = np.linspace(-TILT_DIAG_KMAX, TILT_DIAG_KMAX, TILT_DIAG_NBINS + 1)
        self.edges = e
        self.sum_im = np.zeros((TILT_DIAG_NBINS, TILT_DIAG_NBINS))
        self.cnt = np.zeros((TILT_DIAG_NBINS, TILT_DIAG_NBINS))
        self.w_agree = 0.0
        self.w_total = 0.0

    def add(self, C, spec, geom_grids):
        if geom_grids["kr"] is None or geom_grids["kaz"] is None:
            return
        band = spec["band"]
        kr = geom_grids["kr"][band]
        ka = geom_grids["kaz"][band]
        im = np.imag(C[band])
        w = np.abs(C[band])

        ok = np.isfinite(im) & np.isfinite(w) & (np.abs(kr) > 1e-4)
        if not ok.any():
            return

        expected = -np.sign(kr[ok])          # TILT_SIGN = -1
        self.w_agree += float(np.sum(w[ok] * (np.sign(im[ok]) == expected)))
        self.w_total += float(np.sum(w[ok]))

        ir = np.digitize(kr[ok], self.edges) - 1
        ia = np.digitize(ka[ok], self.edges) - 1
        good = (ir >= 0) & (ir < TILT_DIAG_NBINS) & (ia >= 0) & (ia < TILT_DIAG_NBINS)
        np.add.at(self.sum_im, (ia[good], ir[good]), im[ok][good])
        np.add.at(self.cnt, (ia[good], ir[good]), 1.0)

    def report(self, path):
        if self.w_total <= 0:
            print("[tilt] no data accumulated")
            return np.nan

        frac = self.w_agree / self.w_total
        print(f"[tilt] Im C antisymmetry: {100*frac:.1f} % of |C|-weighted band "
              f"power has sign(Im C) == -sign(k.r_hat)")
        if frac >= 0.65:
            print("       -> conventions CONFIRMED; TILT_SIGN = -1 and, by the "
                  "same chain, VB_SIGN = -1 are consistent with the data")
        elif frac <= 0.35:
            print("       -> conventions INVERTED. Flip TILT_SIGN and VB_SIGN "
                  "and rerun before using any direction.")
        else:
            print("       -> INCONCLUSIVE. The tilt signature is weak; treat "
                  "every direction as unverified.")

        with np.errstate(invalid="ignore"):
            M = np.where(self.cnt > 0, self.sum_im / self.cnt, np.nan)

        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        pc = ax.pcolormesh(self.edges, self.edges, M, cmap="RdBu_r",
                           vmin=-v, vmax=v)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"$k\cdot\hat{r}$  [rad m$^{-1}$]  (away from nadir)")
        ax.set_ylabel(r"$k\cdot\hat{a}$  [rad m$^{-1}$]  (along-track)")
        ax.set_title(f"mean Im C   ({100*frac:.0f} % agree with $-$sign$(k\\cdot\\hat r)$)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(pc, ax=ax, label="Im C")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[tilt] diagnostic figure: {path}")
        return frac


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_box(spec, geom_grids, resp_pow, main, depth_stats, current_stats,
                  theta_ref, rng, direction_resolved=True):
    """
    Resample Welch windows. Every matched partition is put on the FINAL lobe of
    the main partition (VB- or model-chosen) before direction and Doppler are
    evaluated, so the CIs describe spread about the chosen lobe; lobe-sign
    stability is measured separately through the VB votes.
    current_stats is the EFFECTIVE current dict returned by effective_current.
    """
    E_arr,C_arr=spec["E_arr"],spec["C_arr"]; N=E_arr.shape[0]
    n_pick=int(round(min(max(spec["n_eff"],2.0),float(N))))
    kx0,ky0=main["kx"],main["ky"]
    b_lam=[]; b_th=[]; b_Ti=[]; b_Tc=[]; b_conf=[]; b_res=[]; b_dom=[]; b_vote=[]
    success=0
    for _ in range(N_BOOT):
        ind=rng.integers(0,N,size=n_pick); Eb=np.mean(E_arr[ind],axis=0); Cb=np.mean(C_arr[ind],axis=0)
        E_dn,_=denoise(Eb,spec,resp_pow); parts=extract_partitions(E_dn,Cb,spec,resp_pow,geom_grids)
        if not parts: continue
        d=[min(np.hypot(p["kx"]-kx0,p["ky"]-ky0),np.hypot(-p["kx"]-kx0,-p["ky"]-ky0)) for p in parts]
        j=int(np.argmin(d)); p=parts[j]; success+=1
        th_to=align_to_lobe(p,kx0,ky0); th_rep=(th_to+180)%360 if DIR_CONVENTION=="from" else th_to
        depth=depth_stats["mean"]; ue=current_stats["U_e"]; un=current_stats["U_n"]
        if PROPAGATE_ENV_UNCERTAINTY:
            if np.isfinite(depth_stats.get("std",np.nan)) and depth_stats["std"]>0:
                depth=float(rng.normal(depth,depth_stats["std"]))
                if np.isfinite(depth_stats.get("min",np.nan)): depth=max(depth,depth_stats["min"])
                if np.isfinite(depth_stats.get("max",np.nan)): depth=min(depth,depth_stats["max"])
            for key,sdkey in (("U_e","U_e_std"),("U_n","U_n_std")):
                sd=current_stats.get(sdkey,np.nan)
                if np.isfinite(sd) and sd>0:
                    if key=="U_e": ue=float(rng.normal(ue,sd))
                    else: un=float(rng.normal(un,sd))
        Ti,Tc=periods(p["lambda_m"],th_to,ue,un,depth)
        if not direction_resolved and UNRESOLVED_TC_POLICY=="nan": Tc=np.nan
        b_lam.append(p["lambda_m"]); b_th.append(th_rep); b_Ti.append(Ti); b_Tc.append(Tc)
        b_conf.append(p["lobe_conf"]); b_res.append(float(p["lobe_resolved"])); b_dom.append(float(j==0)); b_vote.append(p.get("vb_vote",np.nan))
    if not b_lam:
        return {k:np.nan for k in ["lambda_CI_low","lambda_CI_high","lambda_CI_half","Ti_CI_low","Ti_CI_high","Ti_CI_half","Tc_CI_low","Tc_CI_high","Tc_CI_half","theta_CI_low","theta_CI_high","theta_CI_half","theta_sigma_boot","lobe_conf_boot","lobe_resolved_frac_boot","dominance_stability","bootstrap_success_frac","lobe_vote_stability_boot"]}
    b_th=np.asarray(b_th); lam_lo,lam_hi,lam_h=linear_ci(b_lam); Ti_lo,Ti_hi,Ti_h=linear_ci(b_Ti); Tc_lo,Tc_hi,Tc_h=linear_ci(b_Tc); th_lo,th_hi,th_h=circular_ci(b_th,theta_ref)
    vote=np.asarray(b_vote,float); vote=vote[np.isfinite(vote)&(vote!=0)]
    main_vote=main.get("vb_vote",np.nan)
    vote_stab=float(np.mean(vote==main_vote)) if vote.size and np.isfinite(main_vote) else np.nan
    return {"lambda_CI_low":lam_lo,"lambda_CI_high":lam_hi,"lambda_CI_half":lam_h,
            "Ti_CI_low":Ti_lo,"Ti_CI_high":Ti_hi,"Ti_CI_half":Ti_h,
            "Tc_CI_low":Tc_lo,"Tc_CI_high":Tc_hi,"Tc_CI_half":Tc_h,
            "theta_CI_low":th_lo,"theta_CI_high":th_hi,"theta_CI_half":th_h,
            "theta_sigma_boot":circular_sigma_deg(b_th),"lobe_conf_boot":float(np.nanmean(b_conf)),
            "lobe_resolved_frac_boot":float(np.nanmean(b_res)),"dominance_stability":float(np.nanmean(b_dom)),
            "bootstrap_success_frac":float(success/N_BOOT),"lobe_vote_stability_boot":vote_stab}


def grid_synthetic_diagnostic(data, box, res, p):
    """Inject one sinusoid at real PIXC locations and quantify gridding-only error."""
    if not RUN_GRID_SYNTHETIC_DIAG:
        return {"grid_diag_amp":np.nan,"grid_diag_lambda_m":np.nan,"grid_diag_dlambda_pct":np.nan,"grid_diag_dtheta_deg":np.nan}
    x,y=data["x"],data["y"]
    sel=((x>=box["x_min"])&(x<=box["x_max"])&(y>=box["y_min"])&(y<=box["y_max"]))
    if np.sum(sel)<MIN_RAW_PIXELS_BOX:
        return {"grid_diag_amp":np.nan,"grid_diag_lambda_m":np.nan,"grid_diag_dlambda_pct":np.nan,"grid_diag_dtheta_deg":np.nan}
    xc,yc=0.5*(box["x_min"]+box["x_max"]),0.5*(box["y_min"]+box["y_max"])
    phi=p["kx"]*(x[sel]-xc)+p["ky"]*(y[sel]-yc); z=np.cos(phi)
    zg,_=_grid_values(x[sel],y[sel],z,box,res)
    ny,nx=zg.shape; xx=box["x_min"]+(np.arange(nx)+0.5)*res; yy=box["y_min"]+(np.arange(ny)+0.5)*res; XX,YY=np.meshgrid(xx,yy)
    ok=np.isfinite(zg); phase=p["kx"]*(XX-xc)+p["ky"]*(YY-yc)
    if ok.sum()<20:
        return {"grid_diag_amp":np.nan,"grid_diag_lambda_m":np.nan,"grid_diag_dlambda_pct":np.nan,"grid_diag_dtheta_deg":np.nan}
    A=np.c_[np.cos(phase[ok]),np.sin(phase[ok]),np.ones(ok.sum())]; beta,*_=np.linalg.lstsq(A,zg[ok],rcond=None); amp=float(np.hypot(beta[0],beta[1]))

    # Use the active Radon estimator for the spectral-location part while
    # preserving the same diagnostic outputs and axial comparison.
    sg=np.where(np.isfinite(zg),1.0+0.1*zg,np.nan)
    sp=compute_welch_spectra(zg,sg,res,lwin=LWIN,min_windows=MIN_N_WINDOWS)
    if sp is None:
        return {"grid_diag_amp":amp,"grid_diag_lambda_m":np.nan,"grid_diag_dlambda_pct":np.nan,"grid_diag_dtheta_deg":np.nan}
    E=np.mean(sp["E_arr"],axis=0); C=np.mean(sp["C_arr"],axis=0)
    rp=np.ones_like(E)
    gg={"kaz":None,"kr":None,"a_hat":None,"r_hat":None}
    Ed,_=denoise(E,sp,rp); ps=extract_partitions(Ed,C,sp,rp,gg)
    if not ps:
        return {"grid_diag_amp":amp,"grid_diag_lambda_m":np.nan,"grid_diag_dlambda_pct":np.nan,"grid_diag_dtheta_deg":np.nan}
    d=[float(axial_difference_deg(q["theta_to"],p["theta_to"])) for q in ps]
    q=ps[int(np.argmin(d))]
    lam=q["lambda_m"]
    return {"grid_diag_amp":amp,"grid_diag_lambda_m":float(lam),
            "grid_diag_dlambda_pct":float(100*(lam-p["lambda_m"])/p["lambda_m"]),
            "grid_diag_dtheta_deg":float(axial_difference_deg(q["theta_to"],p["theta_to"]))}


def multiscale_check(ssh_g, sig_g, res, geom, d_ac, main):
    out={"multiscale_available":False,"multiscale_lambda_m":np.nan,"multiscale_theta_degN":np.nan,"multiscale_dlambda_rel":np.nan,"multiscale_dtheta_deg":np.nan,"multiscale_consistent":True}
    if not MULTISCALE or main["lambda_m"]<MULTISCALE_TRIGGER_LAMBDA: return out
    sp=compute_welch_spectra(ssh_g,sig_g,res,lwin=LWIN_SECONDARY,min_windows=MULTISCALE_MIN_WINDOWS)
    if sp is None: return out
    gg={"kaz":(sp["KX"]*geom["a_hat"][0]+sp["KY"]*geom["a_hat"][1]) if geom["a_hat"] else None,
        "kr":(sp["KX"]*geom["r_hat"][0]+sp["KY"]*geom["r_hat"][1]) if geom["r_hat"] else None,"a_hat":geom["a_hat"],"r_hat":geom["r_hat"]}
    rp=instrument_response_power(sp,geom,res,d_ac); E=np.mean(sp["E_arr"],axis=0); C=np.mean(sp["C_arr"],axis=0); Ed,_=denoise(E,sp,rp); ps=extract_partitions(Ed,C,sp,rp,gg)
    if not ps: return out
    d=[min(np.hypot(p["kx"]-main["kx"],p["ky"]-main["ky"]),np.hypot(-p["kx"]-main["kx"],-p["ky"]-main["ky"])) for p in ps]; p=ps[int(np.argmin(d))]
    dl=abs(p["lambda_m"]-main["lambda_m"])/main["lambda_m"]; dt=float(axial_difference_deg(p["theta_to"],main["theta_to"]))
    out.update({"multiscale_available":True,"multiscale_lambda_m":p["lambda_m"],"multiscale_theta_degN":p["theta_to"],"multiscale_dlambda_rel":float(dl),"multiscale_dtheta_deg":dt,"multiscale_consistent":bool(dl<=MULTISCALE_MAX_REL_DLAMBDA and dt<=MULTISCALE_MAX_DTHETA_DEG)})
    return out


# ============================================================
# OBSERVABILITY GRADING
# ============================================================

def observability_grade(row):
    demerits=0; reasons=[]
    if row.get("rain_reject",False): return "D","rain_cell"
    resp=row.get("resp_pow_at_peak",np.nan)
    if int(row.get("n_partitions", 1) or 0) == 0: return "D", "no_partition"
    if np.isfinite(resp):
        if resp<RESPONSE_MIN_MASK: demerits+=2; reasons.append("filter_response<mask")
        elif resp<RESPONSE_WARN: demerits+=1; reasons.append("filter_response<0.5")
    rho=abs(row.get("cross_track_m",np.nan))
    if np.isfinite(rho) and rho<CROSS_TRACK_OBS_WARN: demerits+=1; reasons.append("inner_swath")
    vf=row.get("valid_frac_win_med",np.nan)
    if np.isfinite(vf) and vf<0.90: demerits+=1; reasons.append("gapped_windows")
    gl=row.get("gap_leakage",np.nan)
    if DEMOTE_ON_GAP_LEAKAGE and np.isfinite(gl) and gl>GAP_LEAKAGE_MAX: demerits+=1; reasons.append("gap_leakage")
    src=row.get("lobe_source","vb" if row.get("lobe_resolved",False) else "unresolved")
    if src=="unresolved": demerits+=1; reasons.append("lobe_unresolved")
    elif src=="model":
        demerits+=int(MODEL_LOBE_DEMERIT); reasons.append("lobe_from_model")
    e1,e2=row.get("p1_energy",np.nan),row.get("p2_energy",np.nan)
    if np.isfinite(e1) and np.isfinite(e2) and e1>0 and e2/e1>DOMINANCE_AMBIGUOUS: demerits+=1; reasons.append("dominance_ambiguous")
    if row.get("p2_possible_harmonic",False) or row.get("p3_possible_harmonic",False): demerits+=1; reasons.append("harmonic_present")
    if row.get("p3_possible_sideband",False): demerits+=1; reasons.append("sideband_present")
    lam=row.get("lambda_obs_m",np.nan); lres=row.get("lambda_res_m",np.nan)
    if np.isfinite(lam) and np.isfinite(lres) and lres>0 and lam/lres<LAMBDA_RES_MIN_WAVES: demerits+=1; reasons.append("too_few_waves_per_window")
    lmin=row.get("lambda_min_local_m",LAMBDA_MIN)
    if np.isfinite(lam) and not (lmin<=lam<=LAMBDA_MAX): demerits+=1; reasons.append("centroid_outside_local_band")
    tp=row.get("T_absolute_from_lambda_obs_current_s",np.nan)
    if not np.isfinite(tp): tp=row.get("T_intrinsic_from_lambda_obs_s",np.nan)
    if np.isfinite(tp) and not (T_VALID_RANGE[0]<=tp<=T_VALID_RANGE[1]): demerits+=2; reasons.append("T_out_of_range")
    if row.get("multiscale_available",False) and not row.get("multiscale_consistent",True): demerits+=1; reasons.append("multiscale_inconsistent")
    # WW3 sea-state terms only enter grading when the user has calibrated thresholds.
    if WW3_HS_WARN is not None and np.isfinite(row.get("ww3_Hs_m",np.nan)) and row["ww3_Hs_m"]>WW3_HS_WARN: demerits+=1; reasons.append("ww3_Hs_high")
    if WW3_STEEPNESS_WARN is not None and np.isfinite(row.get("ww3_Hs_over_Lp",np.nan)) and row["ww3_Hs_over_Lp"]>WW3_STEEPNESS_WARN: demerits+=1; reasons.append("ww3_steepness_high")
    grade=["A","B","C","D"][min(demerits,3)]
    return grade,";".join(reasons) if reasons else "none"


# ============================================================
# PROCESS ONE BOX
# ============================================================
def empty_retrieval_row(data, box, geom, spec, depth, cur, ww, gdiag, qdiag,
                        rain, res, d_ac, valid_frac_box, n0, status):
    """Estimator-independent row for a box that produced no usable retrieval."""
    proj = data["proj"]
    xc, yc = 0.5*(box["x_min"]+box["x_max"]), 0.5*(box["y_min"]+box["y_max"])
    lon_c, lat_c = proj(xc, yc, inverse=True)
    a_hat, r_hat = geom["a_hat"], geom["r_hat"]
    ct = geom["cross_track_mean"]
    row = {"Date": DATE, "Cycle": CYCLE, "Pass": PASS, "Domain": DOMAIN_NAME,
           "Box": box["box"], "grid_ix": box["grid_ix"], "grid_iy": box["grid_iy"],
           "acq_time": str(data["acq_time"]) if data.get("acq_time") is not None else None,
           "lon_center": float(lon_c), "lat_center": float(lat_c),
           "x_center": float(xc), "y_center": float(yc),
           "x_min": box["x_min"], "x_max": box["x_max"],
           "y_min": box["y_min"], "y_max": box["y_max"],
           "cross_track_m": ct,
           "swath_side": ("left" if ct < 0 else "right") if np.isfinite(ct) else "unknown",
           "along_track_degN": vec_to_deg_from_north(*a_hat) if a_hat else np.nan,
           "look_dir_degN": vec_to_deg_from_north(*r_hat) if r_hat else np.nan,
           "posting_along_m": ALONG_TRACK_POSTING, "posting_across_m": float(d_ac),
           "res_m": float(res), "lambda_min_local_m": float(spec["lambda_min_local"]),
           "valid_frac_box": float(valid_frac_box),
           "valid_frac_win_med": spec["valid_frac_win_med"],
           "layover_flag_frac": qdiag["layover_flag_frac"],
           "phase_unwrap_suspect_frac": qdiag["phase_unwrap_suspect_frac"],
           "n_quality_pixels": qdiag["n_quality_pixels"],
           "noise_amp": float(n0) if np.isfinite(n0) else np.nan,
           "Depth_m": depth["mean"], "Depth_median_m": depth["median"],
           "Depth_std_m": depth["std"], "Depth_min_m": depth["min"],
           "Depth_max_m": depth["max"], "n_depth_pixels": depth["n"],
           "U_e_mps": cur["U_e"], "U_n_mps": cur["U_n"],
           "U_e_std_mps": cur["U_e_std"], "U_n_std_mps": cur["U_n_std"],
           "current_speed_mps": float(np.hypot(cur["U_e"], cur["U_n"])),
           "current_time_delta_h": cur["time_delta_h"],
           "n_raw_pixels": int(box["n_raw_pixels"]),
           "n_win": int(spec["n_win"]), "n_eff": float(spec["n_eff"]),
           "n_partitions": 0,
           "coherence_sig95": coherence_significance(spec["n_eff"]),
           "lobe_resolved": False, "lobe_method": "none",
           "lobe_source": "none", "direction_resolved": False,
           "retrieval_status": status}
    row.update(gdiag); row.update(box_rain(rain, box)); row.update(ww)
    for n in range(1, N_PARTITIONS_KEEP+1):
        pref = f"p{n}_"
        for s in ("lambda_m", "theta_degN", "energy", "ang_spread_deg",
                  "resp_pow", "harmonic_order"):
            row[pref+s] = np.nan
        row[pref+"lobe_resolved"] = False; row[pref+"possible_harmonic"] = False
        row[pref+"possible_sideband"] = False; row[pref+"artifact_type"] = "none"
    return row
def process_one_box(data, bathy, current, rain, ww3, box, rng, tilt_diag):
    ct_guess=np.nan
    if data["cross_track"] is not None:
        inside=((data["x"]>=box["x_min"])&(data["x"]<=box["x_max"])&(data["y"]>=box["y_min"])&(data["y"]<=box["y_max"]))
        if inside.any(): ct_guess=float(np.nanmean(data["cross_track"][inside]))
    res,d_ac=choose_res(ct_guess,data["h_sat"])
    gridded=grid_one_box(data,box,res)
    if gridded is None: return None
    ssh_g,sig_g,valid_frac_box,sel,gdiag=gridded
    spec=compute_welch_spectra(ssh_g,sig_g,res,lwin=LWIN,min_windows=MIN_N_WINDOWS)
    if spec is None: return None
    geom=box_swath_geometry(data,sel)
    qdiag=box_geolocation_quality(data,sel)
    geom_grids={"kaz":(spec["KX"]*geom["a_hat"][0]+spec["KY"]*geom["a_hat"][1]) if geom["a_hat"] else None,
                "kr":(spec["KX"]*geom["r_hat"][0]+spec["KY"]*geom["r_hat"][1]) if geom["r_hat"] else None,
                "a_hat":geom["a_hat"],"r_hat":geom["r_hat"]}
    resp_pow=instrument_response_power(spec,geom,res,d_ac)
    depth=box_depth_stats(bathy,box)
    if REQUIRE_BATHY and depth["n"]==0: return None
    if np.isfinite(depth["mean"]) and depth["mean"]<MIN_DEPTH_M: return None
    cur=box_current_stats(current,box); ww=box_ww3_stats(ww3,box,depth["mean"],cur)
    E_m=np.mean(spec["E_arr"],axis=0); C_m=np.mean(spec["C_arr"],axis=0)
    if TILT_DIAG and tilt_diag is not None: tilt_diag.add(C_m,spec,geom_grids)
    E_dn,n0=denoise(E_m,spec,resp_pow); parts=extract_partitions(E_dn,C_m,spec,resp_pow,geom_grids)
    if not parts:
        return empty_retrieval_row(data, box, geom, spec, depth, cur, ww, gdiag,
                               qdiag, rain, res, d_ac, valid_frac_box, n0,
                               "no_partition")
    flag_nonlinear_artifacts(parts); resolve_lobes_with_model(parts, ww); p1=parts[0]
    direction_resolved=p1["lobe_source"] in ("vb","model")
    th_to=p1["theta_to"]; th_rep=(th_to+180)%360 if DIR_CONVENTION=="from" else th_to
    th_tilt_rep=p1["theta_tilt"]
    if DIR_CONVENTION=="from" and np.isfinite(th_tilt_rep): th_tilt_rep=(th_tilt_rep+180)%360
    cur_eff=effective_current(cur,p1["k"],depth["mean"])
    Ti,Tc,Tc_lo_lobe,Tc_hi_lobe=wave_periods_final(p1["lambda_m"],th_to,cur_eff,depth["mean"],direction_resolved)
    T_check=Tc if np.isfinite(Tc) else Ti
    if PERIOD_HARD_REJECT and (not np.isfinite(T_check) or not (T_VALID_RANGE[0]<=T_check<=T_VALID_RANGE[1])):
        return None
    ux_,uy_=direction_to_uv(th_to); U_along=cur_eff["U_e"]*ux_+cur_eff["U_n"]*uy_
    sens=response_sensitivity(E_dn,C_m,spec,resp_pow,geom_grids,p1)

    # Window-level physical-partition matching + velocity-bunching vote stability.
    win_theta=[]; win_vote=[]; win_match=0
    for Ew,Cw in zip(spec["E_arr"],spec["C_arr"]):
        Edw,_=denoise(Ew,spec,resp_pow); pw=extract_partitions(Edw,Cw,spec,resp_pow,geom_grids)
        if not pw: continue
        d=[min(np.hypot(p["kx"]-p1["kx"],p["ky"]-p1["ky"]),np.hypot(-p["kx"]-p1["kx"],-p["ky"]-p1["ky"])) for p in pw]
        p=pw[int(np.argmin(d))]; win_match+=1; t=align_to_lobe(p,p1["kx"],p1["ky"])
        win_theta.append((t+180)%360 if DIR_CONVENTION=="from" else t); win_vote.append(p.get("vb_vote",np.nan))
    win_theta=np.asarray(win_theta,float); R_window=circular_resultant_R(win_theta)
    main_vote=p1.get("vb_vote",np.nan); wv=np.asarray(win_vote,float); wv=wv[np.isfinite(wv)&(wv!=0)]
    window_lobe_stability=float(np.mean(wv==main_vote)) if wv.size and np.isfinite(main_vote) else np.nan
    window_match_frac=float(win_match/spec["n_win"]) if spec["n_win"] else np.nan

    boot=bootstrap_box(spec,geom_grids,resp_pow,p1,depth,cur_eff,th_rep,rng,direction_resolved)
    rain_info=box_rain(rain,box); ms=multiscale_check(ssh_g,sig_g,res,geom,d_ac,p1); gd=grid_synthetic_diagnostic(data,box,res,p1)
    proj=data["proj"]; xc,yc=0.5*(box["x_min"]+box["x_max"]),0.5*(box["y_min"]+box["y_max"]); lon_c,lat_c=proj(xc,yc,inverse=True)
    a_hat,r_hat=geom["a_hat"],geom["r_hat"]
    coh=coherence_at(parts,spec,0); coh_sig=coherence_significance(spec["n_eff"])

    row={"spectral_estimator":"radon_fourier_slice","Date":DATE,"Cycle":CYCLE,"Pass":PASS,"Domain":DOMAIN_NAME,"Box":box["box"],"grid_ix":box["grid_ix"],"grid_iy":box["grid_iy"],
         "acq_time":str(data["acq_time"]) if data.get("acq_time") is not None else None,
         "lon_center":float(lon_c),"lat_center":float(lat_c),"x_center":float(xc),"y_center":float(yc),"x_min":box["x_min"],"x_max":box["x_max"],"y_min":box["y_min"],"y_max":box["y_max"],
         "cross_track_m":geom["cross_track_mean"],"swath_side":("left" if geom["cross_track_mean"]<0 else "right") if np.isfinite(geom["cross_track_mean"]) else "unknown",
         "along_track_degN":vec_to_deg_from_north(*a_hat) if a_hat else np.nan,"look_dir_degN":vec_to_deg_from_north(*r_hat) if r_hat else np.nan,
         "posting_along_m":ALONG_TRACK_POSTING,"posting_across_m":float(d_ac),"res_m":float(res),"lambda_min_local_m":float(spec["lambda_min_local"]),
         "valid_frac_box":float(valid_frac_box),"valid_frac_win_med":spec["valid_frac_win_med"],"gap_leakage":gdiag["gap_leakage"],"mean_raw_per_cell":gdiag["mean_raw_per_cell"],
         "layover_flag_frac":qdiag["layover_flag_frac"],"phase_unwrap_suspect_frac":qdiag["phase_unwrap_suspect_frac"],"n_quality_pixels":qdiag["n_quality_pixels"],
         "resp_pow_at_peak":p1["resp_pow"],"noise_amp":float(n0) if np.isfinite(n0) else np.nan,"lambda_res_m":float(p1["lambda_m"]**2/LWIN),
         "Depth_m":depth["mean"],"Depth_median_m":depth["median"],"Depth_std_m":depth["std"],"Depth_min_m":depth["min"],"Depth_max_m":depth["max"],"n_depth_pixels":depth["n"],
         "U_e_mps":cur["U_e"],"U_n_mps":cur["U_n"],"U_e_std_mps":cur["U_e_std"],"U_n_std_mps":cur["U_n_std"],"current_speed_mps":float(np.hypot(cur["U_e"],cur["U_n"])),"current_time_delta_h":cur["time_delta_h"],
         "n_raw_pixels":int(box["n_raw_pixels"]),"n_win":int(spec["n_win"]),"n_eff":float(spec["n_eff"]),"n_partitions":int(len(parts)),
         "lambda_obs_m":float(p1["lambda_m"]),"lambda_CI_low_m":boot["lambda_CI_low"],"lambda_CI_high_m":boot["lambda_CI_high"],"lambda_CI_half_m":boot["lambda_CI_half"],
         "theta_obs_degN":float(th_rep),"theta_CI_low_degN":boot["theta_CI_low"],"theta_CI_high_degN":boot["theta_CI_high"],"theta_CI_half_deg":boot["theta_CI_half"],"theta_sigma_boot_deg":boot["theta_sigma_boot"],"theta_sigma_window_deg":circular_sigma_deg(win_theta),"ang_spread_deg":p1["ang_spread_deg"],
         "k_along_obs_radm":p1["k_along"],"k_across_obs_radm":p1["k_across"],"kx_obs_radm":p1["kx"],"ky_obs_radm":p1["ky"],
         "lobe_conf":p1["lobe_conf"],"lobe_align":p1["lobe_align"],"lobe_resolved":bool(p1["lobe_resolved"]),"lobe_method":p1["lobe_method"],"lobe_conf_boot":boot["lobe_conf_boot"],"lobe_resolved_frac_boot":boot["lobe_resolved_frac_boot"],"lobe_vote_stability_boot":boot["lobe_vote_stability_boot"],"window_lobe_stability":window_lobe_stability,"window_match_frac":window_match_frac,"theta_tilt_degN":th_tilt_rep,
         "T_intrinsic_from_lambda_obs_s":float(Ti),"T_intrinsic_from_lambda_obs_CI_low_s":boot["Ti_CI_low"],"T_intrinsic_from_lambda_obs_CI_high_s":boot["Ti_CI_high"],
         "T_absolute_from_lambda_obs_current_s":float(Tc),"T_absolute_from_lambda_obs_current_CI_low_s":boot["Tc_CI_low"],"T_absolute_from_lambda_obs_current_CI_high_s":boot["Tc_CI_high"],"T_absolute_from_lambda_obs_current_CI_half_s":boot["Tc_CI_half"],
         "coherence_peak":coh,"coherence_sig95":coh_sig,"R_window":float(R_window),"dominance_stability":boot["dominance_stability"],"bootstrap_success_frac":boot["bootstrap_success_frac"],
         "sig0_db_med":gdiag["sig0_db_med"],"sig0_std_short":gdiag["sig0_std_short"],"ssh_std_short":gdiag["ssh_std_short"],"sig0_input_units":data.get("sig0_units_input",""),"sig0_converted_from_db":bool(data.get("sig0_from_db",False))}
    # Legacy v3 aliases retained for downstream notebooks.
    row["Tp_obs_intrinsic_s"]=row["T_intrinsic_from_lambda_obs_s"]; row["Tp_obs_current_corr_s"]=row["T_absolute_from_lambda_obs_current_s"]
    # v4.3: final lobe decision, effective current and Doppler diagnostics.
    row.update({"lobe_source":p1["lobe_source"],"direction_resolved":bool(direction_resolved),
                "lobe_method_swot":p1.get("lobe_method_swot",p1["lobe_method"]),
                "model_lobe_dev_deg":p1.get("model_lobe_dev_deg",np.nan),
                "vb_agrees_model":p1.get("vb_agrees_model",np.nan),
                "U_e_mps":cur_eff["U_e"],"U_n_mps":cur_eff["U_n"],
                "U_e_surface_mps":cur_eff["U_e_surface"],"U_n_surface_mps":cur_eff["U_n_surface"],
                "current_speed_mps":float(np.hypot(cur_eff["U_e"],cur_eff["U_n"])),
                "current_depth_mode":cur_eff["current_mode"],"current_weight_top10m":cur_eff["current_weight_top10m"],
                "U_along_wave_mps":float(U_along) if direction_resolved else np.nan,
                "doppler_rel_pct":float(100*(Tc-Ti)/Ti) if np.isfinite(Tc) and np.isfinite(Ti) and Ti>0 else np.nan,
                "T_absolute_lobe_bound_low_s":Tc_lo_lobe,"T_absolute_lobe_bound_high_s":Tc_hi_lobe})
    row.update(sens)
    row.update(rain_info); row.update(ms); row.update(gd); row.update(ww)
    for n in range(1,N_PARTITIONS_KEEP+1):
        pref=f"p{n}_"
        if n<=len(parts):
            p=parts[n-1]; th=p["theta_to"]
            row[pref+"lambda_m"]=float(p["lambda_m"]); row[pref+"theta_degN"]=float((th+180)%360 if DIR_CONVENTION=="from" else th); row[pref+"energy"]=p["energy"]; row[pref+"ang_spread_deg"]=p["ang_spread_deg"]; row[pref+"lobe_resolved"]=bool(p["lobe_resolved"]); row[pref+"lobe_source"]=p.get("lobe_source","unresolved"); row[pref+"resp_pow"]=p["resp_pow"]; row[pref+"possible_harmonic"]=bool(p.get("possible_harmonic",False)); row[pref+"harmonic_order"]=p.get("harmonic_order",np.nan); row[pref+"possible_sideband"]=bool(p.get("possible_sideband",False)); row[pref+"artifact_type"]=p.get("artifact_type","none")
        else:
            for s in ("lambda_m","theta_degN","energy","ang_spread_deg","resp_pow","harmonic_order"): row[pref+s]=np.nan
            row[pref+"lobe_resolved"]=False; row[pref+"possible_harmonic"]=False; row[pref+"possible_sideband"]=False; row[pref+"artifact_type"]="none"
    return row


# ============================================================
# QUALITY FLAGS
# ============================================================

def add_quality_flags(df):
    df=df.copy()
    coh_req=np.maximum(QC_MIN_COHERENCE,df["coherence_sig95"].fillna(QC_MIN_COHERENCE))
    # VB sign stability is a property of the VB decision, so it is required
    # only where VB made the decision. Model-chosen and unresolved lobes are
    # judged on the wavelength/axis estimate alone.
    src=df["lobe_source"] if "lobe_source" in df else pd.Series("vb",index=df.index)
    lobe_ok=(~src.eq("vb"))|((df["window_lobe_stability"].fillna(0)>=QC_MIN_WINDOW_LOBE_STABILITY)&
                             (df["lobe_vote_stability_boot"].fillna(0)>=QC_MIN_LOBE_STABILITY))
    ok=((df["R_window"]>=QC_MIN_R)&(df["coherence_peak"]>=coh_req)&
        (df["theta_CI_half_deg"]<=QC_MAX_THETA_CI)&
        (df["bootstrap_success_frac"]>=QC_MIN_BOOT_SUCCESS)&
        (df["dominance_stability"]>=QC_MIN_DOMINANCE_STABILITY)&
        (df["window_match_frac"]>=QC_MIN_WINDOW_MATCH_FRAC)&lobe_ok)
    df["spectral_internal_quality"]=ok.fillna(False).astype(bool)
    grades=[]; reasons=[]
    for _,r in df.iterrows():
        g,why=observability_grade(r); grades.append(g); reasons.append(why)
    df["instrument_observability"]=grades; df["observability_reason"]=reasons
    df["usable"]=(df["spectral_internal_quality"]&df["instrument_observability"].isin(["A","B"]))
    return df


# ============================================================
# OUTPUT
# ============================================================

def clean_json_value(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v


def export_geojson(df, data, out_path):
    proj = data["proj"]
    features = []
    for _, row in df.iterrows():
        corners = [(row["x_min"], row["y_min"]), (row["x_max"], row["y_min"]),
                   (row["x_max"], row["y_max"]), (row["x_min"], row["y_max"]),
                   (row["x_min"], row["y_min"])]
        ll = []
        for xx, yy in corners:
            lo, la = proj(xx, yy, inverse=True)
            ll.append([float(lo), float(la)])
        props = {k: clean_json_value(v) for k, v in row.to_dict().items()
                 if k not in ["x_min", "x_max", "y_min", "y_max"]}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [ll]}})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "name": "SWOT_HR_wave_boxes",
                   "crs": {"type": "name",
                           "properties": {"name": "EPSG:4326"}},
                   "features": features}, f, indent=2)


def plot_map(df, value_col, title, cb_label, out_path, categorical=False):
    d = df[np.isfinite(df["theta_obs_degN"])].copy()
    if categorical:
        d = d[d[value_col].notna()]
    else:
        d = d[np.isfinite(d[value_col])]
    if len(d) == 0:
        print(f"[map skip] no data for {value_col}")
        return

    good = d["usable"].astype(bool).values
    fig, ax = plt.subplots(figsize=(10, 8))

    if categorical:
        cmap = {"A": "#1a7f37", "B": "#9bd07a", "C": "#f0a202", "D": "#c1121f"}
        for gkey, c in cmap.items():
            s = d[value_col].values == gkey
            if s.any():
                ax.scatter(d.loc[s, "lon_center"], d.loc[s, "lat_center"],
                           c=c, s=70, edgecolors="k", linewidths=0.3,
                           label=f"{gkey} ({int(s.sum())})")
        ax.legend(loc="best", fontsize=8)
    else:
        sc = ax.scatter(d.loc[good, "lon_center"], d.loc[good, "lat_center"],
                        c=d.loc[good, value_col], s=85,
                        edgecolors="black", linewidths=0.4)
        if np.any(~good):
            ax.scatter(d.loc[~good, "lon_center"], d.loc[~good, "lat_center"],
                       c=d.loc[~good, value_col], s=45, marker="x")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cb_label)

    if good.any():
        # theta_obs is written in the DIR_CONVENTION of the run. Arrows show
        # PROPAGATION, so convert first or every arrow points backwards.
        th = d.loc[good, "theta_obs_degN"].values
        if DIR_CONVENTION == "from":
            th = (th + 180.0) % 360.0
        u, v = direction_to_uv(th)
        ax.quiver(d.loc[good, "lon_center"], d.loc[good, "lat_center"], u, v,
                  angles="xy", scale=26, width=0.0042,
                  headwidth=3.6, headlength=4.2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    validate_config(); rng=np.random.default_rng(RANDOM_SEED)
    print("[1/8] Loading SWOT scene ...")
    data=load_swot_tile(SWOT_FILES,GROUP,UTM_ZONE)
    print("[2/8] Loading bathymetry, current, rain and optional WW3 ...")
    bathy=load_bathymetry(BATHY_FILE,data["proj"])
    if bathy is None: print(f"  bathymetry unavailable; fallback depth {FALLBACK_DEPTH_M} m")
    bbox=(float(np.nanmin(data["lon"])),float(np.nanmax(data["lon"])),float(np.nanmin(data["lat"])),float(np.nanmax(data["lat"])))
    current=load_current_field(CURRENT_FILE,data["proj"],data.get("acq_time"),bbox)
    rain=load_rain_cells(LR_FILES,data["proj"],data.get("acq_time")) if RAIN_SCREEN else None
    ww3=load_ww3_field(WW3_FILE,data["proj"],data.get("acq_time")) if WW3_FILE else None
    print("[3/8] Creating boxes ...")
    boxes=make_tile_boxes(data); print(f"  {len(boxes)} candidate boxes")
    tilt_diag=TiltDiagnostic() if TILT_DIAG else None
    print("[4/8] Processing boxes ...")
    rows=[]
    for i,box in enumerate(boxes,1):
        try:
            row=process_one_box(data,bathy,current,rain,ww3,box,rng,tilt_diag)
            if row is None: continue
            rows.append(row)
            if i%25==0 or len(rows)<=5:
                print(f"  [{i}/{len(boxes)}] {box['box']} res={row['res_m']:.0f}m "
                      f"lam={row['lambda_obs_m']:.0f}m th={row['theta_obs_degN']:.0f} "
                      f"T={row['T_absolute_from_lambda_obs_current_s']:.1f}s "
                      f"resp={row['resp_pow_at_peak']:.2f} lobe={row['lobe_method']}")
        except Exception as e:
            print(f"  [skip] {box['box']}: {e}")
    if not rows: raise RuntimeError("no valid boxes were processed")
    if MODEL_LOBE_MODE!="off" and ww3 is None:
        print("  [ww3] no model field -> unresolved lobes stay unresolved")
    print("[5/8] Flagging quality ...")
    df=add_quality_flags(pd.DataFrame(rows)); n=len(df)
    print(f"  {n} boxes retained after hard physical/data filters")
    print(f"  spectral internal quality: {int(df['spectral_internal_quality'].sum())}/{n}")
    print("  instrument observability: "+"  ".join(f"{k}={v}" for k,v in df['instrument_observability'].value_counts().sort_index().items()))
    print(f"  usable (internal AND A/B): {int(df['usable'].sum())}/{n}")
    if "lobe_source" in df:
        print("  lobe source: "+"  ".join(f"{k}={v}" for k,v in df["lobe_source"].value_counts().items()))
        vb=df[df["lobe_source"].eq("vb")]["vb_agrees_model"].dropna() if "vb_agrees_model" in df else pd.Series(dtype=float)
        if vb.size: print(f"  VB lobe agrees with model direction in {100*vb.mean():.1f}% of {vb.size} VB-resolved boxes")
        wk=df[df["lobe_method_swot"].eq("vb_weak")]["vb_agrees_model"].dropna() if "lobe_method_swot" in df else pd.Series(dtype=float)
        if wk.size: print(f"  weak VB vote agrees with model in {100*wk.mean():.1f}% of {wk.size} weak boxes")
        dp=df["doppler_rel_pct"].dropna() if "doppler_rel_pct" in df else pd.Series(dtype=float)
        if dp.size: print(f"  Doppler (Tabs-Tint)/Tint %: pct 5/50/95 = "+" ".join(f"{x:.2f}" for x in np.percentile(dp,[5,50,95])))
        if "current_depth_mode" in df: print(f"  current depth mode: {df['current_depth_mode'].dropna().unique().tolist()}")
    for col,cfg in (("gap_leakage",GAP_LEAKAGE_MAX),("p2_over_p1_energy",DOMINANCE_AMBIGUOUS)):
        v=((df["p2_energy"]/df["p1_energy"]) if col=="p2_over_p1_energy" else df[col]).replace([np.inf,-np.inf],np.nan).dropna().values
        if v.size:
            q=np.percentile(v,[5,25,50,75,95]); print(f"  {col:<26} pct 5/25/50/75/95 = "+" ".join(f"{x:.3g}" for x in q)+f"  threshold={cfg:g}")
    print("[6/8] Tilt antisymmetry diagnostic ...")
    if tilt_diag is not None: tilt_diag.report(TILT_DIAG_OUT)
    print("[7/8] Saving tables ...")
    fc=df.select_dtypes(include=[float]).columns; df[fc]=df[fc].round(6); df.to_csv(CSV_OUT,index=False); export_geojson(df,data,GEOJSON_OUT)
    print(f"  CSV: {CSV_OUT}\n  GeoJSON: {GEOJSON_OUT}")
    print("[8/8] Saving maps ...")
    if MAKE_MAPS:
        plot_map(df,"T_absolute_from_lambda_obs_current_s","SWOT HR: period derived from observed wavelength (current-corrected)","T from observed lambda (s)",FIG_TP_OUT)
        plot_map(df,"lambda_obs_m","SWOT HR: observed wavelength and propagation direction","Observed wavelength (m)",FIG_LAMBDA_OUT)
        plot_map(df,"theta_CI_half_deg","SWOT HR: directional uncertainty (statistical + environmental bootstrap)","Direction 95% CI half-width (deg)",FIG_UNCERT_OUT)
        plot_map(df,"instrument_observability","SWOT HR: instrument observability grade","",FIG_OBS_OUT,categorical=True)
    print("\nFinished V4.")
    print("Important: lambda_obs and theta_obs remain apparent KaRIn quantities. V4 characterises and rejects/flags poorly observable regimes; it does not claim to invert the nonlinear imaging transform.")


if __name__ == "__main__":
    main()
