"""Figure style used across the paper figures."""
import matplotlib

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold", "font.size": 8,
}


def apply_paper_style(**overrides):
    """Times, bold, 8 pt. Keyword overrides use rcParams names with '_' for '.'."""
    matplotlib.rcParams.update(PAPER_STYLE)
    matplotlib.rcParams.update({k.replace("_", "."): v for k, v in overrides.items()})
