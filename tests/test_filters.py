import numpy as np

from handsfree.filters import OneEuroFilter

FPS = 30
DT = 1 / FPS


def run(filt, signal):
    return np.array([filt(x, i * DT) for i, x in enumerate(signal)])


def test_first_sample_passes_through():
    f = OneEuroFilter()
    np.testing.assert_allclose(f([3.0, 4.0], 0.0), [3.0, 4.0])


def test_reduces_jitter_when_still():
    rng = np.random.default_rng(0)
    noisy = 500 + rng.normal(0, 5, size=(300, 2))
    out = run(OneEuroFilter(), noisy)
    assert out[30:].std(axis=0).max() < noisy[30:].std(axis=0).min() / 3


def test_much_less_lag_than_fixed_exponential_smoothing():
    ramp = np.stack([np.arange(60) * 60.0, np.zeros(60)], axis=1)
    ema = ramp[0].copy()
    for x in ramp:
        ema += (x - ema) / 8
    one_euro = run(OneEuroFilter(), ramp)[-1]
    assert abs(one_euro[0] - ramp[-1, 0]) < abs(ema[0] - ramp[-1, 0]) / 10


def test_higher_beta_tracks_fast_motion_with_less_lag():
    ramp = np.stack([np.arange(60) * 60.0, np.zeros(60)], axis=1)  # 1800 px/s
    lag_low = np.abs(run(OneEuroFilter(beta=0.0), ramp) - ramp)[-1, 0]
    lag_high = np.abs(run(OneEuroFilter(beta=0.005), ramp) - ramp)[-1, 0]
    assert lag_high < lag_low / 2


def test_non_increasing_timestamp_returns_previous_value():
    f = OneEuroFilter()
    f(0.0, 0.0)
    prev = f(10.0, DT)
    np.testing.assert_allclose(f(99.0, DT), prev)


def test_reset_restarts_from_next_sample():
    f = OneEuroFilter()
    run(f, np.zeros(10))
    f.reset()
    np.testing.assert_allclose(f(42.0, 5.0), 42.0)
