import numpy as np

from swothr.utils.response import h3, response_power, choose_res


def test_boxcar_zero_at_three_postings():
    d = 22.0
    assert abs(h3(2 * np.pi / (3 * d), d)) < 1e-12


def test_boxcar_minus_3db_at_6p44_postings():
    d = 22.0
    p = h3(2 * np.pi / (6.44 * d), d) ** 2
    assert abs(10 * np.log10(p) + 3.0) < 0.05


def test_response_matches_original_formula():
    k = np.linspace(-0.06, 0.06, 41)
    KX, KY = np.meshgrid(k, k)
    d_ac, res = 20.0, choose_res(20.0)

    def sinc_half(k, w):
        a = 0.5 * k * w
        return np.where(np.abs(a) < 1e-12, 1.0, np.sin(a) / np.where(a == 0, 1, a))
    H = h3(KY, 22.0) * h3(KX, d_ac) * sinc_half(KX, res) * sinc_half(KY, res)
    np.testing.assert_allclose(response_power(KX, KY, d_ac, res), np.clip(H ** 2, 1e-6, None))
