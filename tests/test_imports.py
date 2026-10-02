"""Every step imports without data (settings + function definitions only)."""
import importlib

import pytest

from swothr.cli import STEPS

# These two run their analysis at module level; they are exercised by the CLI.
SCRIPT_STYLE = {"swothr.analysis.validate_profiles"}


@pytest.mark.parametrize("module", sorted(m for m, *_ in STEPS.values()))
def test_import(module):
    if module in SCRIPT_STYLE:
        pytest.skip("module-level script")
    if module.endswith("storm_tracker"):
        pytest.importorskip("cartopy")
    importlib.import_module(module)
