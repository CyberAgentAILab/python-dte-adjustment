"""Statistical-correctness tests against hand-computed and simulated baselines."""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression

from dte_adj import (
    SimpleDistributionEstimator,
    SimpleStratifiedDistributionEstimator,
    AdjustedDistributionEstimator,
)


class TestKnownBaselines(unittest.TestCase):
    def test_qte_location_shift(self):
        # Y = 5 * D: every quantile shifts by exactly 5
        D = np.repeat([0, 1], 20)
        X = np.zeros((40, 1))
        Y = 5.0 * D
        est = SimpleDistributionEstimator().fit(X, D, Y)
        qte, _, _ = est.predict_qte(
            1,
            0,
            quantiles=np.array([0.1, 0.5, 0.9]),
            n_bootstrap=5,
            display_progress=False,
        )
        np.testing.assert_allclose(qte, [5.0, 5.0, 5.0], rtol=0, atol=1e-10)

    def test_qte_hand_computed(self):
        # control sorted: 1..10, treated sorted: 2,4,...,20 (a scale-by-2 effect)
        D = np.repeat([0, 1], 10)
        X = np.zeros((20, 1))
        Y = np.concatenate([np.arange(1, 11), 2 * np.arange(1, 11)]).astype(float)
        est = SimpleDistributionEstimator().fit(X, D, Y)
        qte, _, _ = est.predict_qte(
            1,
            0,
            quantiles=np.array([0.1, 0.5, 0.9]),
            n_bootstrap=5,
            display_progress=False,
        )
        # q-th quantile is the smallest y with F(y) >= q: control 1,5,9; treated 2,10,18
        np.testing.assert_allclose(qte, [1.0, 5.0, 9.0])

    def test_dte_pte_hand_computed(self):
        D = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.zeros((8, 1))
        Y = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0])
        est = SimpleDistributionEstimator().fit(X, D, Y)
        locations = np.array([0.0, 2.0, 4.0, 6.0])
        dte, _, _ = est.predict_dte(1, 0, locations, display_progress=False)
        # F_1 = [0, 0, .5, 1], F_0 = [0, .5, 1, 1]
        np.testing.assert_allclose(dte, [0.0, -0.5, -0.5, 0.0])
        pte, _, _ = est.predict_pte(1, 0, locations, display_progress=False)
        np.testing.assert_allclose(pte, [-0.5, 0.0, 0.5])

    def test_stratified_equals_simple_for_single_stratum(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        D = rng.binomial(1, 0.5, 200)
        Y = rng.normal(size=200) + D
        locations = np.linspace(-2, 3, 8)
        a = SimpleDistributionEstimator().fit(X, D, Y)
        b = SimpleStratifiedDistributionEstimator().fit(X, D, Y, np.zeros(200))
        np.testing.assert_allclose(
            a.predict_dte(1, 0, locations, display_progress=False)[0],
            b.predict_dte(1, 0, locations, display_progress=False)[0],
        )


class TestSimulatedDGP(unittest.TestCase):
    """Normal location-shift DGP with true DTE(y) = F_1(y) - F_0(y) = Phi((y - tau) / sd) - Phi(y / sd)."""

    @classmethod
    def setUpClass(cls):
        from scipy.stats import norm

        rng = np.random.default_rng(1)
        n, cls.tau = 4000, 1.0
        cls.X = rng.normal(size=(n, 3))
        cls.D = rng.binomial(1, 0.5, n)
        cls.Y = cls.X[:, 0] + cls.tau * cls.D + rng.normal(size=n)
        # Y | D=d ~ N(d * tau, 2)
        sd = np.sqrt(2.0)
        cls.locations = np.array([-1.0, 0.0, 1.0, 2.0])
        cls.true_dte = norm.cdf((cls.locations - cls.tau) / sd) - norm.cdf(
            cls.locations / sd
        )
        cls.norm = norm

    def _assert_covers(self, dte, lo, hi):
        # Pointwise 95% bands miss at each location with prob. ~5%, so check the truth
        # lies within a band widened to ~3 standard errors, and the estimate is accurate.
        half = (hi - lo) / 2
        widened = half * self.norm.ppf(0.9987) / self.norm.ppf(0.975)
        self.assertTrue(np.all(np.abs(dte - self.true_dte) <= widened))

    def test_simple_dte_covers_truth(self):
        est = SimpleDistributionEstimator().fit(self.X, self.D, self.Y)
        dte, lo, hi = est.predict_dte(1, 0, self.locations, display_progress=False)
        self._assert_covers(dte, lo, hi)

    def test_simple_qte_close_to_truth(self):
        est = SimpleDistributionEstimator().fit(self.X, self.D, self.Y)
        qte, _, _ = est.predict_qte(
            1,
            0,
            quantiles=np.array([0.25, 0.5, 0.75]),
            n_bootstrap=20,
            display_progress=False,
        )
        np.testing.assert_allclose(qte, [self.tau] * 3, atol=0.2)

    def test_adjusted_dte_covers_truth_with_tighter_interval(self):
        np.random.seed(0)
        simple = SimpleDistributionEstimator().fit(self.X, self.D, self.Y)
        adj = AdjustedDistributionEstimator(LogisticRegression(), folds=3).fit(
            self.X, self.D, self.Y
        )
        _, s_lo, s_hi = simple.predict_dte(1, 0, self.locations, display_progress=False)
        dte, a_lo, a_hi = adj.predict_dte(1, 0, self.locations, display_progress=False)
        self._assert_covers(dte, a_lo, a_hi)
        # Y depends on X[:, 0], so adjustment should reduce the CI width
        self.assertLess((a_hi - a_lo).mean(), (s_hi - s_lo).mean())


if __name__ == "__main__":
    unittest.main()
