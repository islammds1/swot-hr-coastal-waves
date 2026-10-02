"""Small helpers shared by the analysis scripts."""
from .plotting import apply_paper_style
from .response import h3, sinc_half, response_power, choose_res
from .stats import boot_median, binned

__all__ = ["apply_paper_style", "h3", "sinc_half", "response_power", "choose_res",
           "boot_median", "binned"]
