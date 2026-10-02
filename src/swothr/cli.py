"""
Command-line entry point.

    swothr list
    swothr paths
    swothr fft   -c config/storms/nelson_42.yaml
    swothr radon -c config/storms/*.yaml          # one run per file
    swothr compare
"""
from __future__ import annotations

import argparse
import glob
import os
import runpy
import sys
import time

# name -> (module, one-line description, typical input)
STEPS = {
    "fft":       ("swothr.pipelines.fft",
                  "FFT directional spectrum on 5 km boxes (v4.3)", "storm config"),
    "radon":     ("swothr.pipelines.radon",
                  "Radon / Fourier-slice retrieval on the same boxes (v4.3)", "storm config"),
    "rain-flag": ("swothr.pipelines.rain_flag",
                  "HR-internal rain screen, validated against L2_LR_SSH flags", "storm config"),
    "compare":   ("swothr.analysis.compare_ww3",
                  "FFT / Radon vs MARC-WW3, all storms (model Tp = 1/fp)", "pipeline CSVs"),
    "compare-recomputed-T": ("swothr.analysis.compare_ww3_recomputed_T",
                  "Same, model periods recomputed from lambda_p (alternative)", "pipeline CSVs"),
    "validate":  ("swothr.analysis.validate_profiles",
                  "Single-scene SWOT vs WW3 + along-track transfer (v4 CSV)", "one FFT CSV"),
    "bias":      ("swothr.analysis.along_track_bias",
                  "Origin of the wavelength bias (swath, depth, k components)", "compare output"),
    "mask-sens": ("swothr.analysis.mask_sensitivity",
                  "Fig. S3: sensitivity to the power-response mask", "FFT CSVs"),
    "synthetic": ("swothr.analysis.response_synthetic",
                  "Synthetic linear-filter centroid bias (no data needed)", "none"),
    "bathy":     ("swothr.analysis.bathymetric_control",
                  "Shoaling / refraction / diffraction vs bathymetry (Ciaran)", "FFT CSV + bathy"),
    "box-diag":  ("swothr.analysis.box_diagnostic",
                  "Per-box SSH, spectrum, sinogram, cross-spectrum panels", "PIXC + CSVs"),
    "storms":    ("swothr.plotting.storm_tracker",
                  "2x2 map of storm gust footprints and tracks", "storm NetCDF/CSV"),
}


def run_step(name: str, config: str | None) -> None:
    module = STEPS[name][0]
    if config:
        os.environ["SWOTHR_CONFIG"] = os.path.abspath(config)
    else:
        os.environ.pop("SWOTHR_CONFIG", None)
    print(f"\n=== {name}  ({module})" + (f"  config={config}" if config else ""))
    t0 = time.time()
    saved = sys.argv
    sys.argv = [module]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved
    print(f"=== {name} done in {time.time() - t0:.0f} s")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="swothr", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["list", "paths", *STEPS], help="step to run")
    p.add_argument("-c", "--config", nargs="*", default=[],
                   help="run config(s); globs allowed, one run per file")
    p.add_argument("--keep-going", action="store_true",
                   help="with several configs, continue after a failure")
    a = p.parse_args(argv)

    if a.step == "list":
        w = max(map(len, STEPS))
        for k, (mod, desc, inp) in STEPS.items():
            print(f"  {k:<{w}}  {desc}   [input: {inp}]")
        return 0
    if a.step == "paths":
        from swothr.config import paths, _paths_file
        print(f"paths file: {_paths_file() or '(none)'}")
        for k, v in paths().items():
            print(f"  {k:<11} {v}   {'ok' if os.path.exists(v) else 'MISSING'}")
        return 0

    configs = sorted({f for c in a.config for f in (glob.glob(c) or [c])
                      if not os.path.basename(f).startswith("_")})
    failed = []
    for cfg in configs or [None]:
        try:
            run_step(a.step, cfg)
        except Exception as e:                       # noqa: BLE001
            failed.append(cfg)
            if not a.keep_going:
                raise
            print(f"!!! {a.step} failed for {cfg}: {e!r}")
    if failed:
        print("failed:", ", ".join(map(str, failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
