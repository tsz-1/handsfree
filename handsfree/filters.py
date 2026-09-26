import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez et al., CHI 2012).

    Slow movements get a low cutoff (strong smoothing, less jitter); fast movements raise
    the cutoff (less lag). Works on scalars or numpy vectors.
    """

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return self._x.copy()
        if t <= self._t:
            return self._x.copy()

        dt = t - self._t
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()
