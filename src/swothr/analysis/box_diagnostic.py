#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects as pe

# =============================================================================
# USER SETTINGS  (Ciarán)
# =============================================================================
from swothr.config import path, apply_overrides

BASE = path("root")             # config/paths.yaml

FFT_SCRIPT   = str(Path(__file__).resolve().parents[1] / "pipelines" / "fft.py")
RADON_SCRIPT = str(Path(__file__).resolve().parents[1] / "pipelines" / "radon.py")

PIXC_GLOB = f"{BASE}/Data/SWOT HR/Ciaran/SWOT_L2_HR_PIXC_*.nc"
DATE, CYCLE, PASS = "2023-11-04", 6, 42
UTM_ZONE = 30

FFT_CSV   = f"{BASE}/Results/FFT/swot_5km_FFT_ciaran_FFT_v4.csv"
RADON_CSV = None      # e.g. f"{BASE}/Results/Radon/swot_5km_RADON_ciaran_RADON_v4.csv"

# One figure per group; rows in the order given. Box names are FFT names
# ("B0355", "0355" or 355).
GROUPS = {
    "Z1": ["B0593", "B0592", "B0591", "B0590"],             # north -> south
    "Z3": ["B0355", "B0354", "B0353", "B0352", "B0351"],    # north -> south
    "Z5": ["B0328", "B0369", "B0327", "B0368"],             # no shadow -> Sark shadow
}

OUT_DIR = f"{BASE}/Results/box_diagnostics"
OUT_NAME = "box_diagnostics_Ciaran_{group}.png"     # one image per group
KPLANE_NAME = "kplane_overlay_Ciaran_{group}.png"
MAKE_KPLANE_OVERLAY = True

# --- panel appearance --------------------------------------------------------
RADON_PANEL = "sinogram"        # "sinogram" or "slice"
SPEC_DENOISED = True            # show the noise-subtracted spectrum (what is picked)
SPEC_DB_RANGE = 25.0            # dB below the in-band peak
SSH_CLIP_PCT = (2.0, 98.0)
SHOW_GEOMETRY_COMPASS = False   # inset arrows for satellite heading / look direction
SHOW_RADON_ARROW = False        # FFT arrow only on the SSH panel
SHOW_DEPTH = True               # depth in the row titles and k-plane labels
KMAX_FACTOR = 1.15              # k-axis limit = factor * 2pi / LAMBDA_MIN
ARROW_KM = 1.8                  # arrow half-length on the SSH panel

CMAP_SSH, CMAP_SPEC, CMAP_SINO, CMAP_CROSS = "ocean", "turbo", "viridis", "RdBu_r"
COL_FFT, COL_RAD = "tab:orange", "tab:orange"

PANEL_IN = 2.6
DPI = 250
from swothr.utils import apply_paper_style
apply_paper_style()


# =============================================================================
# IMPORT AND CONFIGURE THE TWO PIPELINES
# =============================================================================
apply_overrides(globals(), "box_diag", final=True)


def load_module(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main() is behind __main__, not run
    print(f"  imported {path.name}")
    return mod


def configure(M):
    """Point a pipeline module at the Ciarán scene (read at call time)."""
    M.SWOT_FILES = PIXC_GLOB
    M.DATE, M.CYCLE, M.PASS = DATE, CYCLE, PASS
    M.UTM_ZONE = UTM_ZONE
    M.MAKE_MAPS = False
    M.TILT_DIAG = False


# =============================================================================
# BOXES FROM THE CSVs
# =============================================================================
def box_id(x):
    s = str(x).strip().upper()
    return s if s.startswith("B") else f"B{int(s):04d}"


def load_targets():
    fdf = pd.read_csv(FFT_CSV)
    fdf["Box"] = fdf["Box"].astype(str).str.strip()
    rdf = pd.read_csv(RADON_CSV) if RADON_CSV else None

    targets = []
    for gname, ids in GROUPS.items():
        for bid in map(box_id, ids):
            r = fdf[fdf["Box"] == bid]
            if r.empty:
                print(f"  ! {bid} not in the FFT CSV, skipped")
                continue
            r = r.iloc[0]
            box = {"box": bid, "grid_ix": int(r["grid_ix"]), "grid_iy": int(r["grid_iy"]),
                   "x_min": float(r["x_min"]), "x_max": float(r["x_max"]),
                   "y_min": float(r["y_min"]), "y_max": float(r["y_max"]),
                   "n_raw_pixels": int(r.get("n_raw_pixels", 0))}
            rad_name = None
            if rdf is not None:
                m = rdf[(rdf["grid_ix"] == box["grid_ix"]) & (rdf["grid_iy"] == box["grid_iy"])]
                rad_name = str(m["Box"].iloc[0]) if len(m) else None
            targets.append({"group": gname, "box": box, "csv": r, "radon_name": rad_name})
    return targets


# =============================================================================
# RE-RUN ONE BOX THROUGH A PIPELINE (same call order as process_one_box)
# =============================================================================
class SinogramCapture:
    """Wraps skimage.radon inside the Radon module to keep its sinograms."""
    def __init__(self, fn):
        self.fn, self.sinos = fn, []

    def __call__(self, image, *a, **kw):
        s = self.fn(image, *a, **kw)
        self.sinos.append(s)
        return s


def run_box(M, data, box, capture_sinogram=False):
    x, y = data["x"], data["y"]
    inside = ((x >= box["x_min"]) & (x <= box["x_max"]) &
              (y >= box["y_min"]) & (y <= box["y_max"]))
    ct_guess = (float(np.nanmean(data["cross_track"][inside]))
                if data["cross_track"] is not None and inside.any() else np.nan)
    res, d_ac = M.choose_res(ct_guess, data["h_sat"])

    gridded = M.grid_one_box(data, box, res)
    if gridded is None:
        return None
    ssh_g, sig_g, valid_frac, sel, _ = gridded

    cap = None
    if capture_sinogram:
        cap = SinogramCapture(M.radon)
        M.radon = cap
    try:
        spec = M.compute_welch_spectra(ssh_g, sig_g, res, lwin=M.LWIN,
                                       min_windows=M.MIN_N_WINDOWS)
    finally:
        if cap is not None:
            M.radon = cap.fn
    if spec is None:
        return None

    geom = M.box_swath_geometry(data, sel)
    geom_grids = {
        "kaz": (spec["KX"] * geom["a_hat"][0] + spec["KY"] * geom["a_hat"][1]) if geom["a_hat"] else None,
        "kr": (spec["KX"] * geom["r_hat"][0] + spec["KY"] * geom["r_hat"][1]) if geom["r_hat"] else None,
        "a_hat": geom["a_hat"], "r_hat": geom["r_hat"]}
    resp = M.instrument_response_power(spec, geom, res, d_ac)

    E_m = np.mean(spec["E_arr"], axis=0)
    C_m = np.mean(spec["C_arr"], axis=0)
    E_dn, n0 = M.denoise(E_m, spec, resp)
    parts = M.extract_partitions(E_dn, C_m, spec, resp, geom_grids)
    if parts:
        M.flag_nonlinear_artifacts(parts)
    coh = M.coherence_at(parts, spec, 0) if parts else np.nan

    out = {"ssh": ssh_g, "sig": sig_g, "res": res, "spec": spec, "E": E_m,
           "E_dn": E_dn, "C": C_m, "parts": parts, "coh": coh, "geom": geom}
    if cap is not None and cap.sinos:
        out["sino"] = np.mean(np.stack(cap.sinos), axis=0)
    return out


# =============================================================================
# PANELS
# =============================================================================
def _ring(ax, lam, color, lw=0.7, alpha=0.8):
    t = np.linspace(0, 2 * np.pi, 240)
    k = 2 * np.pi / lam
    ax.plot(k * np.cos(t), k * np.sin(t), color=color, lw=lw, ls="--", alpha=alpha)


def _kaxes(ax, kmax):
    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(-kmax, kmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (rad m$^{-1}$)")
    ax.set_ylabel(r"$k_y$ (rad m$^{-1}$)")


def _dir_arrow(ax, x0, y0, theta_to, L, color, resolved, ls="-"):
    """Arrow along bearing theta_to; double-headed if the lobe is unresolved."""
    t = np.deg2rad(theta_to)
    dx, dy = L * np.sin(t), L * np.cos(t)
    style = "-|>" if resolved else "<|-|>"
    ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0 - dx, y0 - dy),
                arrowprops=dict(arrowstyle=style, color=color, lw=2.0,
                                linestyle=ls, mutation_scale=12,
                                path_effects=[pe.withStroke(linewidth=3.5, foreground="w")]))


def _bearing(v):
    """Unit (east, north) vector -> bearing clockwise from north."""
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360.0) if v else np.nan


def geometry_bearings(f):
    g = f["geom"] if f else None
    if g is None:
        return np.nan, np.nan
    return _bearing(g["a_hat"]), _bearing(g["r_hat"])


def _compass(ax, track, look):
    """Satellite heading and look direction, drawn in the upper-left corner."""
    x0, y0, L = 0.15, 0.79, 0.10
    ax.add_patch(plt.Rectangle((0.01, 0.62), 0.30, 0.37, transform=ax.transAxes,
                               fc="w", ec="0.3", lw=0.5, alpha=0.85, zorder=6))
    for b, col, lab in ((track, "k", "sat"), (look, "tab:red", "look")):
        if not np.isfinite(b):
            continue
        t = np.deg2rad(b)
        dx, dy = L * np.sin(t), L * np.cos(t)
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                    mutation_scale=9), zorder=7)
        ax.text(x0 + 1.75 * dx, y0 + 1.75 * dy, lab, transform=ax.transAxes,
                color=col, fontsize=6, ha="center", va="center", zorder=7)


def panel_ssh(ax, M, f, r):
    Z = f["ssh"].copy()
    valid = np.isfinite(Z)
    Z = M.detrend_plane_valid(np.where(valid, Z, 0.0), valid)
    Z = np.where(valid, Z, np.nan)
    v = Z[np.isfinite(Z)]
    norm = None
    if v.size:
        lo, hi = np.percentile(v, SSH_CLIP_PCT)
        m = max(abs(lo), abs(hi))
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
    L = Z.shape[1] * f["res"] / 1e3
    H = Z.shape[0] * f["res"] / 1e3
    ax.imshow(Z, origin="lower", cmap=CMAP_SSH, norm=norm, extent=[0, L, 0, H])
    if f["parts"]:
        p = f["parts"][0]
        _dir_arrow(ax, L / 2, H / 2, p["theta_to"], ARROW_KM, COL_FFT, p["lobe_resolved"])
    if SHOW_RADON_ARROW and r is not None and r["parts"]:
        p = r["parts"][0]
        _dir_arrow(ax, L / 2, H / 2, p["theta_to"], ARROW_KM * 0.8, COL_RAD,
                   p["lobe_resolved"], ls="--")
    if SHOW_GEOMETRY_COMPASS:
        _compass(ax, *geometry_bearings(f))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def panel_fft(ax, M, f, kmax):
    spec = f["spec"]
    E = f["E_dn"] if SPEC_DENOISED else f["E"]
    Z = 10 * np.log10(np.maximum(E, 1e-30))
    top = np.nanmax(Z[spec["band"]]) if spec["band"].any() else np.nanmax(Z)
    ax.pcolormesh(spec["KX"], spec["KY"], Z, cmap=CMAP_SPEC,
                  vmin=top - SPEC_DB_RANGE, vmax=top, shading="auto")
    for lam in (M.LAMBDA_MIN, M.LAMBDA_MAX):
        _ring(ax, lam, "w")
    _kaxes(ax, kmax)


def panel_radon(ax, R, r):
    spec = r["spec"]
    if RADON_PANEL == "sinogram" and "sino" in r:
        S = r["sino"]
        th = spec["radon_thetas"]
        rho = (np.arange(S.shape[0]) - S.shape[0] / 2.0) * r["res"] / 1e3
        ax.pcolormesh(th, rho, S, cmap=CMAP_SINO, shading="auto")
        if r["parts"]:
            # skimage.radon: a wave at bearing b peaks at projection angle b - 90 (mod 180)
            ax.axvline((r["parts"][0]["theta_to"] - 90.0) % 180.0, color="r", lw=1.0, ls="--")
        ax.set_xlabel("projection angle (deg)")
        ax.set_ylabel(r"offset $\rho$ (km)")
        ax.set_ylim(rho.min() * 0.6, rho.max() * 0.6)
    else:
        # Fourier-slice spectrum on (axial bearing, k); this is what is partitioned
        E = r["E_dn"] if SPEC_DENOISED else r["E"]
        Z = 10 * np.log10(np.maximum(E, 1e-30))
        band = spec["radon_band"]
        top = np.nanmax(Z[band]) if band.any() else np.nanmax(Z)
        bear = spec["radon_theta_to"][0]
        o = np.argsort(bear)
        k = spec["RK"][:, 0]
        ax.pcolormesh(bear[o], k, Z[:, o], cmap=CMAP_SPEC,
                      vmin=top - SPEC_DB_RANGE, vmax=top, shading="auto")
        for lam in (R.LAMBDA_MIN, R.LAMBDA_MAX):
            ax.axhline(2 * np.pi / lam, color="w", lw=0.7, ls="--")
        ax.set_ylim(0, 2 * np.pi / R.LAMBDA_MIN * KMAX_FACTOR)
        ax.set_xlabel("axial bearing (deg)")
        ax.set_ylabel(r"$k$ (rad m$^{-1}$)")


def panel_cross(ax, M, f, kmax):
    spec = f["spec"]
    Re = np.real(f["C"])
    ref = np.abs(Re[spec["band"]]) if spec["band"].any() else np.abs(Re)
    m = float(np.nanpercentile(ref, 99)) if ref.size else 1.0
    m = m if np.isfinite(m) and m > 0 else 1.0
    ax.pcolormesh(spec["KX"], spec["KY"], Re, cmap=CMAP_CROSS, vmin=-m, vmax=m,
                  shading="auto")
    for lam in (M.LAMBDA_MIN, M.LAMBDA_MAX):
        _ring(ax, lam, "k", lw=0.6, alpha=0.6)
    if f["parts"]:
        p = f["parts"][0]
        txt = (f"resolved, conf={p['lobe_conf']:.2f}" if p["lobe_resolved"]
               else f"UNRESOLVED, conf={p['lobe_conf']:.2f}")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.5,
                bbox=dict(fc="w", ec="none", alpha=0.8, pad=1))
    _kaxes(ax, kmax)


def box_depth(t):
    return abs(float(t["csv"].get("Depth_median_m", np.nan)))


def row_title(t, f, r):
    c = t["csv"]

    def frm(th):
        return (th + 180.0) % 360.0

    lf = f["parts"][0]["lambda_m"] if f and f["parts"] else np.nan
    tf = frm(f["parts"][0]["theta_to"]) if f and f["parts"] else np.nan
    name = t["box"]["box"]
    trk, look = geometry_bearings(f)
    depth = f"   h = {box_depth(t):.0f} m" if SHOW_DEPTH else ""
    return (f"{name}{depth}   sat {trk:.0f}°, look {look:.0f}°\n"
            f"FFT: $\\lambda$ = {lf:.0f} m   $\\theta_{{from}}$ = {tf:.0f}°\n"
            f"R = {float(c.get('R_window', np.nan)):.2f}   "
            f"±{float(c.get('theta_CI_half_deg', np.nan)):.0f}°   "
            f"coh = {f['coh'] if f else np.nan:.2f}")


def panel_kplane(ax, FFT, targets, results):
    kmax = KMAX_FACTOR * 2 * np.pi / FFT.LAMBDA_MIN * 0.75
    rows = [(t, f) for t, (f, _) in zip(targets, results) if f is not None and f["parts"]]
    cmap = plt.get_cmap("viridis")
    hs = [box_depth(t) for t, _ in rows]
    hn = mcolors.Normalize(vmin=min(hs) if hs else 0, vmax=max(hs) if hs else 1)
    for lam in (FFT.LAMBDA_MIN, 200.0, 300.0, FFT.LAMBDA_MAX):
        if 2 * np.pi / lam > kmax:
            continue
        _ring(ax, lam, "0.6", lw=0.5, alpha=0.7)
        ax.text(0, 2 * np.pi / lam, f"{lam:.0f} m", fontsize=6, color="0.4",
                ha="center", va="bottom")
    for (t, f), h in zip(rows, hs):
        p = f["parts"][0]
        col = cmap(0.85 * (1.0 - hn(h)))   # deep = dark, shallow = green
        ax.annotate("", xy=(p["kx"], p["ky"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>" if p["lobe_resolved"] else "-",
                                    color=col, lw=2))
        lab = f" {t['box']['box'][1:]}" + (f" ({h:.0f} m)" if SHOW_DEPTH else "")
        ax.text(p["kx"], p["ky"], lab, fontsize=6.5, color=col)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    _kaxes(ax, kmax)
    ax.tick_params(labelsize=7)


# =============================================================================
# FIGURES (two separate images per group)
# =============================================================================
def make_rows(FFT, RAD, targets, results):
    """Box diagnostics: one row of four panels per box, no figure title."""
    kmax = KMAX_FACTOR * 2 * np.pi / FFT.LAMBDA_MIN
    n = len(targets)
    fig, axes = plt.subplots(n, 4, figsize=(4 * PANEL_IN + 1.2, n * (PANEL_IN + 0.35) + 0.8))
    axes = np.atleast_2d(axes)
    for i, (t, (f, r)) in enumerate(zip(targets, results)):
        if f is not None:
            panel_ssh(axes[i, 0], FFT, f, r)
            panel_fft(axes[i, 1], FFT, f, kmax)
            panel_cross(axes[i, 3], FFT, f, kmax)
        if r is not None:
            panel_radon(axes[i, 2], RAD, r)
        axes[i, 0].set_title(row_title(t, f, r), fontsize=7, loc="left")
        for j in range(4):
            axes[i, j].tick_params(labelsize=7)
            if (j in (0, 1, 3) and f is None) or (j == 2 and r is None):
                axes[i, j].text(0.5, 0.5, "no retrieval", ha="center",
                                transform=axes[i, j].transAxes)
    heads = ["SSH (detrended)", "FFT spectrum (dB)",
             "Radon sinogram" if RADON_PANEL == "sinogram" else "Radon Fourier-slice (dB)",
             r"cross-spectrum Re $C$"]
    for j, h in enumerate(heads):
        axes[0, j].annotate(h, xy=(0.5, 1.0), xycoords="axes fraction",
                            xytext=(0, 38), textcoords="offset points",
                            ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=3.0)
    return fig


def make_kplane(FFT, targets, results, group):
    """Dominant k-vector of every box in the group on one k-plane."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    panel_kplane(ax, FFT, targets, results)
    ax.set_title(f"{group}: dominant k-vector", fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("[1/4] importing the pipelines")
    FFT = load_module(FFT_SCRIPT, "fft_v4")
    RAD = load_module(RADON_SCRIPT, "radon_v42")
    configure(FFT)
    configure(RAD)

    print("[2/4] resolving boxes")
    targets = load_targets()
    if not targets:
        raise SystemExit("no boxes resolved")

    print("[3/4] loading the pixel cloud (once per pipeline, with its own QC)")
    data_f = FFT.load_swot_tile(FFT.SWOT_FILES, FFT.GROUP, UTM_ZONE)
    data_r = RAD.load_swot_tile(RAD.SWOT_FILES, RAD.GROUP, UTM_ZONE)

    results = []
    for t in targets:
        b = t["box"]
        print(f"  {t['group']} {b['box']}")
        f = run_box(FFT, data_f, b)
        r = run_box(RAD, data_r, b, capture_sinogram=True)
        if f is None:
            print("    ! FFT: no retrieval")
        if r is None:
            print("    ! Radon: no retrieval")
        if f and f["parts"]:
            th_csv = float(t["csv"]["theta_obs_degN"])
            th_now = (f["parts"][0]["theta_to"] + 180.0) % 360.0
            d = abs((th_now - th_csv + 180.0) % 360.0 - 180.0)
            if d > 1.0:
                print(f"    ! FFT θ={th_now:.1f}° differs from the CSV ({th_csv:.1f}°): "
                      "check that the script version matches the CSV run")
        results.append((f, r))

    print("[4/4] plotting, one image per group")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        idx = [i for i, t in enumerate(targets) if t["group"] == group]
        if not idx:
            continue
        tg = [targets[i] for i in idx]
        rg = [results[i] for i in idx]
        fig = make_rows(FFT, RAD, tg, rg)
        out = Path(OUT_DIR) / OUT_NAME.format(group=group)
        fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
        print(f"  saved {out}")
        if MAKE_KPLANE_OVERLAY:
            fig2 = make_kplane(FFT, tg, rg, group)
            out2 = Path(OUT_DIR) / KPLANE_NAME.format(group=group)
            fig2.savefig(out2, dpi=DPI, facecolor="white", bbox_inches="tight")
            print(f"  saved {out2}")
    plt.show()


if __name__ == "__main__":
    main()
