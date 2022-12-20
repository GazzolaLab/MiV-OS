from typing import Tuple

import matplotlib.pyplot as plt
import pytest


def test_get_closest_node(create_mock_geometry):
    # Create a mock object that conforms to the MEAGeometryProtocol
    mock_geometry = create_mock_geometry()

    # Test the get_closest_node method with various input coordinates
    assert mock_geometry.get_closest_node(0, 0) == 0
    assert mock_geometry.get_closest_node(1, 1) == 2
    assert mock_geometry.get_closest_node(2, 2) == 4


def test_get_xy(create_mock_geometry):
    # Create a mock object that conforms to the MEAGeometryProtocol
    mock_geometry = create_mock_geometry()

    # Test the get_xy method with various node indices
    assert mock_geometry.get_xy(0) == (0, 0)
    assert mock_geometry.get_xy(1) == (1, 1)
    assert mock_geometry.get_xy(2) == (2, 2)


def test_save_and_load(create_mock_geometry):
    # Create a mock object that conforms to the MEAGeometryProtocol
    mock_geometry = create_mock_geometry()

    # Test the save and load methods
    mock_geometry.save("test.json")
    mock_geometry.load("test.json")


def test_view(create_mock_geometry):
    # Create a mock object that conforms to the MEAGeometryProtocol
    mock_geometry = create_mock_geometry()

    # Test the view method
    fig = mock_geometry.view()
    assert isinstance(fig, plt.Figure)


@pytest.fixture
def create_mock_geometry():
    # Create a mock object that conforms to the MEAGeometryProtocol
    class MockGeometry:
        def get_closest_node(self, x: float, y: float) -> int:
            return x + y

        def get_xy(self, idx: int) -> Tuple[float, float]:
            return (idx, idx)

        def save(self, path: str) -> None:
            pass

        def load(self, path: str) -> None:
            pass

        def view(self) -> plt.Figure:
            return plt.figure()

    return MockGeometry
