"""
Path and run configuration.

Two layers, both optional:

1. ``config/paths.yaml`` (machine-specific, git-ignored) -- where the data live:

       root:        /media/saiful/LaCie/Coastal_Enginnering_SWOT_HR
       bathymetry:  /media/saiful/LaCie/Data/Bathymetry/E4_2024.nc
       storms_dir:  /media/saiful/LaCie/SWOT_HR_Project/Storms

   Looked up in this order: $SWOTHR_PATHS, <repo>/config/paths.yaml,
   ~/.config/swothr/paths.yaml. Without a file, $SWOTHR_ROOT sets ``root``.

2. A run file (e.g. ``config/storms/nelson_42.yaml``) given with ``--config``
   or $SWOTHR_CONFIG. Its keys are the UPPER-CASE settings of the scripts,
   grouped by section::

       common:   {DATE: "2024-03-29", CYCLE: 13, PASS: 42}
       pipeline: {SWOT_FILES: "{root}/Data/SWOT HR/Nelson_42/*.nc"}   # fft + radon
       fft:      {TAG: Nelson_42_FFT_v4.3}

   Strings may contain {root}, {bathymetry}, {storms_dir}. A script only
   receives keys it already defines; anything else is reported as unknown.
"""
from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLIED = "__swothr_applied__"


class _Keep(dict):
    """format_map helper: leave unknown {placeholders} untouched."""
    def __missing__(self, key):
        return "{" + key + "}"


def _paths_file() -> Path | None:
    for p in (os.environ.get("SWOTHR_PATHS"),
              REPO_ROOT / "config" / "paths.yaml",
              Path.home() / ".config" / "swothr" / "paths.yaml"):
        if p and Path(p).is_file():
            return Path(p)
    return None


@lru_cache(maxsize=None)
def paths() -> dict:
    f = _paths_file()
    p = yaml.safe_load(f.read_text()) if f else {}
    p = {k: str(Path(v).expanduser()) for k, v in (p or {}).items() if v is not None}
    if "root" not in p:
        if os.environ.get("SWOTHR_ROOT"):
            p["root"] = os.environ["SWOTHR_ROOT"]
        else:
            p["root"] = str(REPO_ROOT / "workspace")
            warnings.warn("No config/paths.yaml and no $SWOTHR_ROOT: using "
                          f"{p['root']}. Copy config/paths.example.yaml to "
                          "config/paths.yaml and edit it.", stacklevel=2)
    p.setdefault("bathymetry", str(Path(p["root"]) / "Data" / "Bathymetry" / "E4_2024.nc"))
    p.setdefault("storms_dir", str(Path(p["root"]) / "Data" / "Storms"))
    return p


def path(key: str) -> str:
    """One entry of paths.yaml, as a string (scripts mix str and Path)."""
    return paths()[key]


def expand(value):
    """Substitute {root}, {bathymetry}, ... recursively in strings."""
    if isinstance(value, str):
        return value.format_map(_Keep(paths()))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def config_file() -> Path | None:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a in ("--config", "-c") and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--config="):
            return Path(a.split("=", 1)[1])
    env = os.environ.get("SWOTHR_CONFIG")
    return Path(env) if env else None


def load_run_config(file: Path | None = None) -> dict:
    file = file or config_file()
    if file is None:
        return {}
    if not file.is_file():
        raise FileNotFoundError(f"run config not found: {file}")
    return yaml.safe_load(file.read_text()) or {}


def _cast(new, old):
    if isinstance(old, Path):
        return Path(new)
    if isinstance(old, tuple) and isinstance(new, list):
        return tuple(tuple(v) if isinstance(v, list) else v for v in new)
    return new


def apply_overrides(namespace: dict, sections, final: bool = False) -> dict:
    """
    Overwrite module-level settings in ``namespace`` (pass ``globals()``)
    from the run config. Call once where paths are derived (before
    ``OUT_DIR / f"...{TAG}..."``) and once with ``final=True`` at the end
    of the settings block: the second call only sets keys the first one
    could not see, then reports keys no section of the script defines.
    """
    if isinstance(sections, str):
        sections = (sections,)
    cfg = load_run_config()
    merged = {}
    for s in sections:
        merged.update(cfg.get(s) or {})
    done = namespace.setdefault(_APPLIED, set())
    for k, v in merged.items():
        if k in done or k not in namespace:
            continue
        namespace[k] = _cast(expand(v), namespace[k])
        done.add(k)
    if final and merged:
        unknown = sorted(set(merged) - done)
        if unknown:
            warnings.warn(f"run config keys not used by this script: {unknown}",
                          stacklevel=2)
        if done:
            print(f"[config] {len(done)} setting(s) from "
                  f"{config_file()}: {', '.join(sorted(done))}")
    return namespace
