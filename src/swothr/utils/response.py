"""
Linear instrument power response of the L2 HR pixel cloud, as used by the
v4.3 pipelines:

    R(k) = [ H3(k.a, d_al) H3(k.r, d_ac) sinc(k_a res/2) sinc(k_r res/2) ]^2
    H3(k, d) = (1 + 2 cos(k d)) / 3        (3x3 boxcar; zero at lambda = 3d)
"""
import numpy as np

ALONG_TRACK_POSTING = 22.0     # m
RES_ADAPT_FACTOR = 1.5


def h3(k, d):
    return (1.0 + 2.0 * np.cos(k * d)) / 3.0


def sinc_half(k, w):
    a = 0.5 * np.asarray(k, float) * w
    return np.where(np.abs(a) < 1e-12, 1.0, np.sin(a) / np.where(a == 0, 1, a))


def response_power(k_across, k_along, d_ac, res, d_al=ALONG_TRACK_POSTING, floor=1e-6):
    """Power response on a (k_across, k_along) grid, clipped at ``floor``."""
    H = h3(k_along, d_al) * h3(k_across, d_ac)
    H = H * sinc_half(k_across, res) * sinc_half(k_along, res)
    return np.clip(H ** 2, floor, None)


def choose_res(d_ac, d_al=ALONG_TRACK_POSTING, factor=RES_ADAPT_FACTOR):
    """Adaptive gridding resolution (m)."""
    return float(np.round(factor * max(d_al, d_ac)))
