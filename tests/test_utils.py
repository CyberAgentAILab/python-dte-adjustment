import unittest
import numpy as np
import pandas as pd
import polars as pl
from dte_adj.util import _convert_to_ndarray


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
