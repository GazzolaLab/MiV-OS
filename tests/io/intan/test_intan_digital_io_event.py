import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

import miv
import miv.core.operator.cachable as cacher_module
import miv.io.intan.data as intan_module


class TestYourClass:
    @patch.object(intan_module.DataIntan, "__init__", lambda x: None)
    def test_load_digital_in_event(self):
        # Create an instance of your class
        instance = intan_module.DataIntan()

        with patch.object(instance, "_generator_by_channel_name") as mock_generator:
            # Mock the _generator_by_channel_name function
            # Configure the mock to return some data
            mock_generator.return_value = [
                Mock(
                    data=np.array([[0, 1], [1, 0]], dtype=np.bool_).T,
                    timestamps=np.array([0, 1]),
                ),
                Mock(
                    data=np.array([[1, 0, 1], [0, 1, 1]], dtype=np.bool_).T,
                    timestamps=np.array([2, 3, 4]),
                ),
            ]

            # Call the function you want to test
            result = instance._load_digital_event_common("test_str", 2)

            # Assertions
            np.testing.assert_array_equal(result.data, [[1, 2, 4], [0, 3, 4]])

            # Assertions for mock calls
            mock_generator.assert_called_once_with("test_str", progress_bar=False)
