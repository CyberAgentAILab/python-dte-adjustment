import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from dte_adj.plot import plot


class TestPlot(unittest.TestCase):
    @patch("dte_adj.plot.plt")
    def test_plot(self, mock_plt):
        # Arrange
        x_values = np.array([1, 2, 3, 4, 5])
        means = np.array([1, 2, 3, 4, 5])
        upper_bands = np.array([2, 3, 4, 5, 6])
        lower_bands = np.array([0, 1, 2, 3, 4])
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        # Act
        result_ax = plot(
            x_values,
            means,
            lower_bands,
            upper_bands,
            title="Test Title",
            xlabel="X Axis",
            ylabel="Y Axis",
            chart_type="line",
        )

        # Assert
        self.assertEqual(result_ax, mock_ax)
        mock_plt.subplots.assert_called_once()
        plot_call = mock_ax.plot.call_args
        fill_between_call = mock_ax.fill_between.call_args
        plot_args, plot_kwargs = plot_call
        x_values_arg, y_values_arg = plot_args
        self.assertTrue(np.array_equal(x_values_arg, x_values))
        self.assertTrue(np.array_equal(y_values_arg, means))
        fill_between_args, fill_between_kwargs = fill_between_call
        x_fill, lower_fill, upper_fill = fill_between_args
        self.assertTrue(np.array_equal(x_fill, x_values_arg))
        self.assertTrue(np.array_equal(lower_fill, lower_bands))
        self.assertTrue(np.array_equal(upper_fill, upper_bands))
        self.assertEqual(fill_between_kwargs["color"], "green")
        self.assertAlmostEqual(fill_between_kwargs["alpha"], 0.3)
        self.assertEqual(fill_between_kwargs["label"], "Confidence Interval")
        mock_ax.set_title.assert_called_once_with("Test Title")
        mock_ax.set_xlabel.assert_called_once_with("X Axis")
        mock_ax.set_ylabel.assert_called_once_with("Y Axis")

    def test_plot_fail_unknown_chart_type(self):
        # Arrange
        x_values = np.array([1, 2, 3, 4, 5])
        means = np.array([1, 2, 3, 4, 5])
        upper_bands = np.array([2, 3, 4, 5, 6])
        lower_bands = np.array([0, 1, 2, 3, 4])

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            plot(
                x_values,
                means,
                lower_bands,
                upper_bands,
                title="Test Title",
                xlabel="X Axis",
                ylabel="Y Axis",
                chart_type="other",
            )
        self.assertEqual(
            str(cm.exception),
            "Chart type other is not supported",
        )


if __name__ == "__main__":
    unittest.main()
