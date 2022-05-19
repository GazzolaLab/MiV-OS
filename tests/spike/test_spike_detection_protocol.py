from typing import Any, Protocol, runtime_checkable

import pytest

from miv.signal.spike import SpikeDetectionProtocol
from tests.spike.mock_spike_detection import (
    mock_nonspike_detection_list,
    mock_spike_detection_list,
)


@runtime_checkable
class RuntimeSpikeDetectionProtocol(SpikeDetectionProtocol, Protocol):
    # This only check the presence of required method, not their type signature.
    # Check @typing.runtime_checkable documentation for more detail.
    ...


@pytest.mark.parametrize("MockSpikeDetection", mock_spike_detection_list)
def test_protocol_abide(MockSpikeDetection: SpikeDetectionProtocol):
    spike_detection: SpikeDetectionProtocol = MockSpikeDetection()
    assert isinstance(spike_detection, RuntimeSpikeDetectionProtocol)


@pytest.mark.parametrize("NonSpikeDetection", mock_nonspike_detection_list)
def test_non_protocol_filter(NonSpikeDetection: Any):
    spike_detection: Any = NonSpikeDetection()
    with pytest.raises(TypeError):
        issubclass(spike_detection, RuntimeSpikeDetectionProtocol)
