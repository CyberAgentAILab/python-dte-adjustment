import unittest
import numpy as np
import pandas as pd
import polars as pl
from dte_adj.util import _convert_to_ndarray, _infer_default_locations


class TestConvertToNdarray(unittest.TestCase):
    """Test that _convert_to_ndarray correctly converts various array-like inputs."""

    def test_ndarray(self):
        data = np.array([1, 2, 3])
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_ndarray_2d(self):
        data = np.array([[1, 2], [3, 4]])
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_pandas_series(self):
        data = pd.Series([1, 2, 3])
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_pandas_dataframe(self):
        data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[1, 3], [2, 4]]))

    def test_polars_series(self):
        data = pl.Series([1, 2, 3])
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_polars_dataframe(self):
        data = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[1, 3], [2, 4]]))

    def test_list(self):
        data = [1, 2, 3]
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_tuple(self):
        data = (1, 2, 3)
        result = _convert_to_ndarray(data)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


class TestInferDefaultLocations(unittest.TestCase):
    """Test _infer_default_locations for generating default DTE/PTE locations."""

    def test_evenly_spaced_spanning_outcome_range(self):
        rng = np.random.default_rng(0)
        outcomes = rng.normal(size=500)
        result = _infer_default_locations(outcomes)
        self.assertAlmostEqual(result[0], float(outcomes.min()))
        self.assertAlmostEqual(result[-1], float(outcomes.max()))
        diffs = np.diff(result)
        np.testing.assert_allclose(diffs, diffs[0])

    def test_for_intervals_left_endpoint_below_min(self):
        outcomes = np.linspace(2.0, 5.0, 50)
        result = _infer_default_locations(outcomes, for_intervals=True)
        self.assertLess(result[0], outcomes.min())
        self.assertAlmostEqual(result[-1], 5.0)

    def test_auto_n_locations_matches_histogram_bin_edges(self):
        rng = np.random.default_rng(0)
        outcomes = rng.normal(size=1000)
        expected_n = len(np.histogram_bin_edges(outcomes, bins="auto"))
        result = _infer_default_locations(outcomes)
        self.assertEqual(result.shape, (expected_n,))

    def test_auto_scales_with_sample_size(self):
        rng = np.random.default_rng(1)
        small = _infer_default_locations(rng.normal(size=100))
        large = _infer_default_locations(rng.normal(size=10000))
        self.assertGreater(large.shape[0], small.shape[0])

    def test_for_intervals_constant_outcomes(self):
        outcomes = np.full(50, 3.0)
        result = _infer_default_locations(outcomes, for_intervals=True)
        self.assertLess(result[0], 3.0)
