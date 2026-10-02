from pathlib import Path

import pytest

from swothr import config


def test_overrides_expand_and_cast(tmp_path, monkeypatch):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "common: {CYCLE: 13}\n"
        "fft: {SWOT_FILES: '{root}/x/*.nc', OUT_DIR: '{root}/out', T_RANGE: [4, 20], NOPE: 1}\n")
    monkeypatch.setenv("SWOTHR_CONFIG", str(cfg))
    ns = {"CYCLE": 6, "SWOT_FILES": "", "OUT_DIR": Path("/old"), "T_RANGE": (1, 2)}
    config.apply_overrides(ns, ("common", "fft"))
    assert ns["CYCLE"] == 13
    assert ns["SWOT_FILES"] == f"{tmp_path}/x/*.nc"
    assert ns["OUT_DIR"] == tmp_path / "out" and isinstance(ns["OUT_DIR"], Path)
    assert ns["T_RANGE"] == (4, 20)
    with pytest.warns(UserWarning, match="NOPE"):
        config.apply_overrides(ns, ("common", "fft"), final=True)


def test_late_keys_are_set_by_final_call(tmp_path, monkeypatch):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("fft: {TAG: a, LATE: 2}\n")
    monkeypatch.setenv("SWOTHR_CONFIG", str(cfg))
    ns = {"TAG": "x"}
    config.apply_overrides(ns, "fft")
    ns["TAG"] = "derived-from-a"      # a later assignment must not be overwritten
    ns["LATE"] = 0
    config.apply_overrides(ns, "fft", final=True)
    assert ns["TAG"] == "derived-from-a" and ns["LATE"] == 2


def test_shipped_storm_configs_parse():
    import yaml
    for f in (Path(__file__).parents[1] / "config" / "storms").glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        assert {"common", "pipeline"} <= set(d), f.name
