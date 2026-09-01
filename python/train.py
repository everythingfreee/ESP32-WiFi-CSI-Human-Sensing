"""Train presence and position models from the collected dataset.

Pipeline (see the project README for the full description):

    load dataset -> validate -> group by session -> session-based split
        -> normalize features (fit on train only) -> train candidates
           (KNN, Random Forest, small neural network)
        -> select best per task on VALIDATION data
        -> report honest TEST metrics -> save models + preprocessing config

Selection is never based on test results; the test set is only touched for
the final report.  If the neural network (Model B) does not beat the
baselines (Model A) on validation data, the baselines win.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler

from config import load_config, resolve_path, setup_logging
from dataset import Dataset
from models.baseline_model import (BaselinePositionModel,
                                   BaselinePresenceModel,
                                   build_position_candidates,
                                   build_presence_candidates,
                                   save_sklearn_model)

LOG = logging.getLogger("wifi_sensing.train")


def _xy(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[feature_columns].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, df


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }



def train_task(
    task: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    cfg: dict,
    scaler: StandardScaler,
    use_nn: bool,
) -> Tuple[object, Dict[str, Dict[str, float]], Dict[str, float], str]:
    """Train all candidates for one task; pick the best on validation.

    Returns (best_model, all_val_metrics, test_metrics_of_best, best_name).
    """
    if task == "presence":
        candidates = build_presence_candidates(cfg)
        fit_df = lambda d: (d[feature_columns].to_numpy(dtype=np.float64),
                            d["presence"].to_numpy())  # noqa: E731
        eval_fn = _classification_metrics
    elif task == "position":
        candidates = build_position_candidates(cfg)
        fit_df = lambda d: (d[feature_columns].to_numpy(dtype=np.float64),
                            d["position"].to_numpy(dtype=np.float64))  # noqa: E731
        eval_fn = _regression_metrics
        # Only samples WITH a person carry a position label.
        train_df = train_df[train_df["presence"] == 1]
        val_df = val_df[val_df["presence"] == 1]
        test_df = test_df[test_df["presence"] == 1]
        train_df = train_df[train_df["position"].notna()]
        val_df = val_df[val_df["position"].notna()]
        test_df = test_df[test_df["position"].notna()]
    else:
        raise ValueError(task)

    x_train_raw, y_train = fit_df(train_df)
    x_val_raw, y_val = fit_df(val_df)
    x_test_raw, y_test = fit_df(test_df)
    if len(x_train_raw) == 0 or len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"not enough labeled data to train the {task} model")
    if len(x_val_raw) == 0 or len(np.unique(y_val)) < 2:
        raise RuntimeError(
            f"validation split for {task} lacks class diversity - collect "
            "more sessions")

    # sklearn pipelines carry their own preprocessing (e.g. StandardScaler
    # inside the KNN pipeline), so they must be fit on RAW features — this
    # keeps the saved artifacts self-contained for inference.  The shared
    # scaler is only used for the torch models.
    x_train_nn = scaler.transform(x_train_raw)
    x_val_nn = scaler.transform(x_val_raw)
    x_test_nn = scaler.transform(x_test_raw)

    val_metrics: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, object] = {}

    # --- sklearn baselines (raw features) ---
    for name, pipeline in candidates.items():
        pipeline.fit(x_train_raw, y_train)
        if task == "presence":
            pred = (pipeline.predict_proba(x_val_raw)[:, 1] >= 0.5).astype(int)
        else:
            pred = pipeline.predict(x_val_raw)
        val_metrics[name] = eval_fn(y_val, pred)
        fitted[name] = pipeline
        LOG.info("%s %s validation: %s", task, name, val_metrics[name])

    # --- neural network (Model B, optional; scaled features) ---
    nn_name = "neural_network"
    if use_nn:
        try:
            from models import neural_network as nn_mod
            model = nn_mod.train_torch_model(task, x_train_nn, y_train,
                                             x_val_nn, y_val, cfg)
            fitted[nn_name] = model
            if task == "presence":
                probs = np.array([nn_mod.TorchPresenceModel(model).predict(row)
                                  for row in x_val_nn])
                pred = (probs >= 0.5).astype(int)
            else:
                pred = np.array([nn_mod.TorchPositionModel(model).predict(row)
                                 for row in x_val_nn])
            val_metrics[nn_name] = eval_fn(y_val, pred)
            LOG.info("%s %s validation: %s", task, nn_name, val_metrics[nn_name])
        except RuntimeError as exc:
            LOG.warning("skipping neural network (%s)", exc)

    # --- selection on validation only ---
    if task == "presence":
        best_name = max(val_metrics, key=lambda k: val_metrics[k]["f1"])
    else:
        best_name = max(val_metrics, key=lambda k: val_metrics[k]["r2"]
                        if not np.isnan(val_metrics[k]["r2"]) else -9)

    best = fitted[best_name]
    if best_name == nn_name:
        from models import neural_network as nn_mod
        if task == "presence":
            test_pred = np.array([
                nn_mod.TorchPresenceModel(best).predict(row)
                for row in x_test_nn])
            test_pred = (test_pred >= 0.5).astype(int)
        else:
            test_pred = np.array([
                nn_mod.TorchPositionModel(best).predict(row)
                for row in x_test_nn])
    else:
        if task == "presence":
            test_pred = (best.predict_proba(x_test_raw)[:, 1] >= 0.5).astype(int)
        else:
            test_pred = best.predict(x_test_raw)
    test_metrics = eval_fn(y_test, test_pred)

    LOG.info("%s: best=%s val=%s test=%s", task, best_name,
             val_metrics[best_name], test_metrics)
    return best, val_metrics, test_metrics, best_name


def save_artifacts(
    models_dir: Path,
    cfg: dict,
    feature_columns: List[str],
    presence_model: object,
    presence_name: str,
    presence_val: Dict[str, Dict[str, float]],
    presence_test: Dict[str, float],
    position_model: Optional[object],
    position_name: Optional[str],
    position_val: Dict[str, Dict[str, float]],
    position_test: Dict[str, float],
    scaler: StandardScaler,
    dataset_path: str,
    seed: int,
) -> None:
    """Persist every artifact inference needs (models are NOT retrained)."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler.pkl")
    (models_dir / "feature_config.json").write_text(
        json.dumps({"feature_names": feature_columns}, indent=2),
        encoding="utf-8")

    manifest: Dict[str, object] = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_path,
        "seed": seed,
        "feature_names": feature_columns,
        "scaler": "scaler.pkl",
    }

    if presence_name == "neural_network":
        from models import neural_network as nn_mod
        nn_mod.save_torch_model(
            presence_model, str(models_dir / "presence_model.pt"),
            {"task": "presence", "n_features": len(feature_columns)})
        manifest["presence"] = {"file": "presence_model.pt",
                                "type": "torch", "name": presence_name}
    else:
        save_sklearn_model(presence_model, str(models_dir / "presence_model.pkl"))
        manifest["presence"] = {"file": "presence_model.pkl",
                                "type": "sklearn", "name": presence_name}

    if position_model is not None:
        if position_name == "neural_network":
            from models import neural_network as nn_mod
            nn_mod.save_torch_model(
                position_model, str(models_dir / "position_model.pt"),
                {"task": "position", "n_features": len(feature_columns)})
            manifest["position"] = {"file": "position_model.pt",
                                    "type": "torch", "name": position_name}
        else:
            save_sklearn_model(position_model,
                               str(models_dir / "position_model.pkl"))
            manifest["position"] = {"file": "position_model.pkl",
                                    "type": "sklearn", "name": position_name}

    report = {
        "presence": {"validation": presence_val, "test": presence_test,
                     "selected": presence_name},
        "position": {"validation": position_val, "test": position_test,
                     "selected": position_name},
    }
    (models_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (models_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("artifacts saved to %s", models_dir)


def print_report(presence_val, presence_test, presence_name,
                 position_val, position_test, position_name) -> None:
    """Human-readable comparison + final test metrics (spec section 22)."""
    width = 60
    print()
    print("=" * width)
    print("Training Results")
    print("=" * width)

    print("\nPresence classifier (validation comparison)")
    print(f"{'model':<16}{'accuracy':>10}{'precision':>11}"
          f"{'recall':>9}{'f1':>8}")
    for name, m in presence_val.items():
        print(f"{name:<16}{m['accuracy']:>10.3f}{m['precision']:>11.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}")
    print(f"\nselected: {presence_name}")
    t = presence_test
    print(f"TEST  accuracy={t['accuracy'] * 100:.1f}%  "
          f"precision={t['precision'] * 100:.1f}%  "
          f"recall={t['recall'] * 100:.1f}%  f1={t['f1'] * 100:.1f}%")

    if position_val:
        print("\nPosition regression (validation comparison)")
        print(f"{'model':<16}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
        for name, m in position_val.items():
            print(f"{name:<16}{m['mae']:>9.3f}{m['rmse']:>9.3f}"
                  f"{m['r2']:>9.3f}")
        print(f"\nselected: {position_name}")
        t = position_test
        print(f"TEST  MAE={t['mae']:.3f}  RMSE={t['rmse']:.3f}  R2={t['r2']:.3f}")
    else:
        print("\nPosition regression: no usable position labels "
              "(collect person sessions at known positions)")
    print("=" * width)

def main(argv=None) -> int:
    """CLI: ``python train.py [--dataset ...] [--models-dir ...]``"""
    parser = argparse.ArgumentParser(
        description="Train presence + position models with session-based "
                    "splitting and honest test metrics")
    parser.add_argument("--dataset", default=None,
                        help="dataset CSV (default: config value)")
    parser.add_argument("--models-dir", default=None,
                        help="output directory (default: config model_path)")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: config value)")
    parser.add_argument("--no-nn", action="store_true",
                        help="skip the neural network (Model B)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    seed = args.seed if args.seed is not None \
        else int(cfg.get("train", {}).get("seed", 42))

    dataset_path = args.dataset or str(resolve_path(cfg, "dataset_path"))
    ds = Dataset.load_csv(dataset_path)
    problems = ds.validate()
    if problems:
        print("Dataset warnings:")
        for p in problems:
            print(f"  - {p}")
    fatal = [p for p in problems if p.startswith("missing column")]
    if fatal:
        print("\nCannot train with a broken dataset schema.", file=sys.stderr)
        return 1
    feature_columns = ds.feature_columns

    train_cfg = cfg.get("train", {})
    train_df, val_df, test_df = ds.session_split(
        test_fraction=float(train_cfg.get("test_fraction", 0.2)),
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        seed=seed)
    print(f"Session-based split: {train_df['session_id'].nunique()} train / "
          f"{val_df['session_id'].nunique()} val / "
          f"{test_df['session_id'].nunique()} test sessions "
          f"({len(train_df)}/{len(val_df)}/{len(test_df)} rows)")

    # The shared scaler is fit on TRAIN rows only (no leakage). sklearn
    # candidates carry their own scaler inside the pipeline; this shared
    # instance is saved for the torch models and for reference.
    x_all_train = train_df[feature_columns].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(np.nan_to_num(x_all_train))

    use_nn = not args.no_nn
    presence_model, presence_val, presence_test, presence_name = train_task(
        "presence", train_df, val_df, test_df, feature_columns, cfg,
        scaler, use_nn)
    try:
        position_model, position_val, position_test, position_name = train_task(
            "position", train_df, val_df, test_df, feature_columns, cfg,
            scaler, use_nn)
    except RuntimeError as exc:
        print(f"Position model skipped: {exc}")
        position_model = None
        position_name = None
        position_val, position_test = {}, {}

    models_dir = Path(args.models_dir) if args.models_dir \
        else resolve_path(cfg, "model_path")
    save_artifacts(models_dir, cfg, feature_columns, presence_model,
                   presence_name, presence_val, presence_test,
                   position_model, position_name, position_val,
                   position_test, scaler, dataset_path, seed)
    print_report(presence_val, presence_test, presence_name,
                 position_val, position_test, position_name)
    print(f"\nArtifacts: {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# __CHUNK_END__
