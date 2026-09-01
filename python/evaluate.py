"""Evaluate trained models on a dataset (honest, session-based).

Loads the artifacts saved by ``train.py`` (via ``model_manifest.json``),
re-derives a session-based test split with the same seed, and reports
metrics computed from actual predictions — nothing is copied from the
training report.

    python evaluate.py [--dataset data/processed/dataset.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

from config import load_config, resolve_path, setup_logging
from dataset import Dataset

LOG = logging.getLogger("wifi_sensing.evaluate")


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the saved models on a dataset")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    from inference import load_models
    models_dir = Path(args.models_dir) if args.models_dir \
        else resolve_path(cfg, "model_path")
    models = load_models(models_dir, cfg)
    if models is None:
        print("No saved models found - run train.py first.", file=sys.stderr)
        return 1

    dataset_path = args.dataset or str(resolve_path(cfg, "dataset_path"))
    ds = Dataset.load_csv(dataset_path)
    problems = ds.validate()
    if problems:
        print("Dataset warnings:")
        for p in problems:
            print(f"  - {p}")
    feature_columns = models.feature_names or ds.feature_columns

    seed = args.seed if args.seed is not None \
        else int(cfg.get("train", {}).get("seed", 42))
    _, _, test_df = ds.session_split(
        test_fraction=float(cfg.get("train", {}).get("test_fraction", 0.2)),
        val_fraction=float(cfg.get("train", {}).get("val_fraction", 0.2)),
        seed=seed)
    print(f"Test split: {test_df['session_id'].nunique()} sessions, "
          f"{len(test_df)} rows")

    width = 60
    print("=" * width)
    print("Evaluation on test sessions")
    print("=" * width)

    # --- presence ---
    x = np.nan_to_num(
        test_df[feature_columns].to_numpy(dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0)
    y = test_df["presence"].to_numpy(dtype=int)
    probs = np.array([models.presence.predict(row) for row in x])
    pred = (probs >= float(cfg.get("presence_threshold", 0.5))).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    print(f"\nPresence ({getattr(models.presence, 'name', '?')})")
    print(f"  accuracy : {acc * 100:.1f}%")
    print(f"  precision: {prec * 100:.1f}%")
    print(f"  recall   : {rec * 100:.1f}%")
    print(f"  f1       : {f1 * 100:.1f}%")

    # --- position ---
    if models.position_available:
        mask = (test_df["presence"] == 1) & test_df["position"].notna()
        xp = x[mask.to_numpy()]
        yp = test_df.loc[mask, "position"].to_numpy(dtype=np.float64)
        if len(xp) == 0:
            print("\nPosition: no person rows with position labels in test "
                  "split")
        else:
            preds = np.array([models.position.predict(row) for row in xp])
            mae = mean_absolute_error(yp, preds)
            rmse = float(np.sqrt(mean_squared_error(yp, preds)))
            r2 = r2_score(yp, preds)
            print(f"\nPosition ({getattr(models.position, 'name', '?')})")
            print(f"  MAE : {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R2  : {r2:.3f}")
            print("\n  MAE per labeled position:")
            for pos in sorted(set(yp.round(2))):
                sel = np.isclose(yp, pos, atol=0.005)
                if sel.any():
                    mae_p = float(np.mean(np.abs(preds[sel] - yp[sel])))
                    print(f"    {pos:.2f}: MAE={mae_p:.3f} "
                          f"(n={int(sel.sum())})")
    else:
        print("\nPosition: no position model saved")
    print("=" * width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
