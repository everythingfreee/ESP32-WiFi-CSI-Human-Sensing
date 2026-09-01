"""Model A: baseline scikit-learn models (KNN + Random Forest).

These serve as the *baseline* the neural network has to beat.  Both tasks
use the same candidate families:

* presence classification (EMPTY=0 / PERSON=1) — ``predict`` returns the
  probability of a person being on the link;
* position regression (normalized 0.0 = ESP32-A .. 1.0 = ESP32-B) —
  ``predict`` returns the position, clipped to [0, 1].

Every candidate is an sklearn ``Pipeline``; KNN pipelines include a
``StandardScaler`` because KNN is distance-based, while tree ensembles do
not need scaling.  Selection happens in ``train.py`` based on validation
performance only.
"""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_presence_candidates(cfg: dict) -> Dict[str, Pipeline]:
    """Build the presence-classifier candidates from ``config["train"]``."""
    train = cfg.get("train", {})
    k = int(train.get("knn_neighbors", 7))
    n_trees = int(train.get("rf_estimators", 200))
    return {
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]),
        "random_forest": Pipeline([
            ("rf", RandomForestClassifier(
                n_estimators=n_trees,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=int(train.get("seed", 42)),
            )),
        ]),
    }


def build_position_candidates(cfg: dict) -> Dict[str, Pipeline]:
    """Build the position-regressor candidates from ``config["train"]``."""
    train = cfg.get("train", {})
    k = int(train.get("knn_neighbors", 7))
    n_trees = int(train.get("rf_estimators", 200))
    return {
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
        ]),
        "random_forest": Pipeline([
            ("rf", RandomForestRegressor(
                n_estimators=n_trees,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=int(train.get("seed", 42)),
            )),
        ]),
    }


def save_sklearn_model(model: Pipeline, path: str) -> None:
    """Persist an sklearn pipeline with joblib."""
    joblib.dump(model, path)


def load_sklearn_model(path: str) -> Pipeline:
    """Load an sklearn pipeline saved by :func:`save_sklearn_model`."""
    return joblib.load(path)



class BaselinePresenceModel:
    """Uniform wrapper around a presence-classification pipeline."""

    def __init__(self, pipeline: Pipeline, name: str = "baseline") -> None:
        self.pipeline = pipeline
        self.name = name
        self.kind = "sklearn"

    def predict(self, features: np.ndarray) -> float:
        """Return P(person) for one feature vector (1-D array)."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        proba = self.pipeline.predict_proba(x)[0]
        # Class 1 == person (class ordering fixed by the training labels).
        classes = list(self.pipeline.classes_) if hasattr(
            self.pipeline, "classes_") else None
        if classes is not None and 1 in classes:
            idx = classes.index(1)
            return float(proba[idx])
        return float(proba[-1])


class BaselinePositionModel:
    """Uniform wrapper around a position-regression pipeline."""

    def __init__(self, pipeline: Pipeline, name: str = "baseline") -> None:
        self.pipeline = pipeline
        self.name = name
        self.kind = "sklearn"

    def predict(self, features: np.ndarray) -> float:
        """Return the normalized position prediction clipped to [0, 1]."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        raw = float(self.pipeline.predict(x)[0])
        return min(max(raw, 0.0), 1.0)
