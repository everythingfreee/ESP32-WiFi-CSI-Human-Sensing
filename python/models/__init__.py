"""Model implementations for the Wi-Fi CSI sensing system.

``baseline_model``
    Model A: scikit-learn candidates (KNN, Random Forest) with a uniform
    predict interface.
``neural_network``
    Model B: small PyTorch networks for presence classification and
    position regression (optional dependency).
"""

from .baseline_model import (
    BaselinePresenceModel,
    BaselinePositionModel,
    build_position_candidates,
    build_presence_candidates,
    load_sklearn_model,
    save_sklearn_model,
)

__all__ = [
    "BaselinePresenceModel",
    "BaselinePositionModel",
    "build_position_candidates",
    "build_presence_candidates",
    "load_sklearn_model",
    "save_sklearn_model",
]

