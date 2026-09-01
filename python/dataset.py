"""Dataset loading, validation, statistics and session-based splitting.

The processed dataset is a CSV with one row per feature window:

    sample_id, session_id, timestamp, rssi, <feature columns...>,
    presence, movement, position

* ``presence``: 1 = person on the link, 0 = empty room.
* ``movement``: 1 = person moving, 0 = static/empty.
* ``position``: normalized 0.0 (ESP32-A) .. 1.0 (ESP32-B); empty for
  samples without a person.

Data-leakage rule: rows from one continuous session are temporally
correlated, so random row splits leak information into the test set and
inflate every metric.  All splitting here is *session-based*.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("wifi_sensing.dataset")

META_COLUMNS = ("sample_id", "session_id", "timestamp")
LABEL_COLUMNS = ("presence", "movement", "position")
#: ``rssi`` is a convenience metadata column in the dataset schema; the
#: feature vector already carries ``rssi_mean``, so rssi must not be treated
#: as an ML feature (it would duplicate rssi_mean and desync train/inference).
NON_FEATURE_COLUMNS = ("rssi",)


class Dataset:
    """Wrapper around the processed dataset with validation + splits."""

    def __init__(self, df: pd.DataFrame, source_path: Optional[str] = None):
        self.df = df
        self.source_path = source_path
        self.feature_columns: List[str] = []

    @classmethod
    def load_csv(cls, path: str) -> "Dataset":
        """Load the dataset CSV with a helpful error when it is missing."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                "Collect one first, e.g.:\n"
                "  python dataset_collector.py --label empty --duration 30\n"
                "  python dataset_collector.py --label person --position 0.5 "
                "--duration 30")
        df = pd.read_csv(csv_path)
        return cls(df, source_path=str(csv_path))

    def validate(self) -> List[str]:
        """Validate schema + values; returns a list of problems (empty=ok)."""
        problems: List[str] = []
        df = self.df
        for col in META_COLUMNS + LABEL_COLUMNS:
            if col not in df.columns:
                problems.append(f"missing column: {col}")
        if problems:
            return problems

        self.feature_columns = [
            c for c in df.columns
            if c not in META_COLUMNS and c not in LABEL_COLUMNS
            and c not in NON_FEATURE_COLUMNS
        ]
        if not self.feature_columns:
            problems.append("no feature columns found")
        if df["session_id"].nunique() < 2:
            problems.append(
                "only one session present - at least 2 (ideally 6+) sessions "
                "are required for a meaningful session-based split")
        if df["presence"].nunique() < 2:
            problems.append(
                "dataset contains a single presence class - collect both "
                "'empty' and 'person' sessions")
        if df.loc[df["presence"] == 1, "position"].isna().all():
            problems.append("person sessions carry no position labels")
        feats = df[self.feature_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(feats).all():
            problems.append("feature columns contain NaN or infinite values")
        n_pos = int((df["presence"] == 1).sum())
        n_neg = int((df["presence"] == 0).sum())
        if min(n_pos, n_neg) < 100:
            problems.append(
                f"class imbalance is severe ({n_neg} empty vs {n_pos} person "
                "rows) - collect more of the smaller class")
        return problems


    def session_split(
        self,
        test_fraction: float = 0.2,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split whole *sessions* into train/val/test sets.

        The split is stratified by each session's dominant class so that
        every split contains both 'empty' and 'person' data whenever the
        collected sessions allow it.  Raises RuntimeError when there are
        too few sessions overall.
        """
        sessions = sorted(self.df["session_id"].astype(str).unique())
        if len(sessions) < 3:
            raise RuntimeError(
                f"only {len(sessions)} session(s); need at least 3 for a "
                "session-based split. Collect more sessions.")
        df = self.df.copy()
        df["_sid"] = df["session_id"].astype(str)
        dom_class = df.groupby("_sid")["presence"].agg(
            lambda s: int(pd.Series(s).mode().iloc[0]))
        rng = np.random.default_rng(seed)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        warnings: List[str] = []
        for cls in sorted(dom_class.unique()):
            ids = sorted(sid for sid, c in dom_class.items() if c == cls)
            ids = list(rng.permutation(ids))
            n = len(ids)
            if n == 1:
                train_ids += ids
                warnings.append(
                    f"class {cls}: single session goes to train only")
                continue
            if n == 2:
                train_ids.append(ids[0])
                val_ids.append(ids[1])
                warnings.append(
                    f"class {cls}: only 2 sessions (train+val); test set "
                    "will not contain this class")
                continue
            n_test = max(1, int(round(n * test_fraction)))
            n_val = max(1, int(round(n * val_fraction)))
            n_test = min(n_test, n - 2)
            n_val = min(n_val, n - n_test - 1)
            test_ids += ids[:n_test]
            val_ids += ids[n_test:n_test + n_val]
            train_ids += ids[n_test + n_val:]
        for w in warnings:
            LOG.warning("session split: %s", w)

        by_id = df["_sid"]
        train = df[by_id.isin(train_ids)].drop(columns="_sid")
        val = df[by_id.isin(val_ids)].drop(columns="_sid")
        test = df[by_id.isin(test_ids)].drop(columns="_sid")
        return (train.reset_index(drop=True), val.reset_index(drop=True),
                test.reset_index(drop=True))

    def stats(self) -> Dict[str, object]:
        """Summary statistics used by ``python dataset.py --stats``."""
        df = self.df
        out: Dict[str, object] = {}
        out["total_samples"] = int(len(df))
        out["sessions"] = int(df["session_id"].nunique())
        out["empty_samples"] = int((df["presence"] == 0).sum())
        out["person_samples"] = int((df["presence"] == 1).sum())
        out["missing_values"] = int(df.isna().sum().sum())
        per_session = df.groupby("session_id").size().to_dict()
        out["samples_per_session"] = {
            str(k): int(v) for k, v in per_session.items()}
        person = df[df["presence"] == 1]
        if "position" in person.columns and len(person):
            per_pos = person.groupby(
                person["position"].round(2)).size().to_dict()
            out["samples_per_position"] = {
                str(float(k)): int(v) for k, v in sorted(per_pos.items())}
        else:
            out["samples_per_position"] = {}
        out["feature_columns"] = int(len(self.feature_columns))
        return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python dataset.py --stats [--dataset data/processed/dataset.csv]``."""
    import argparse

    from config import load_config, resolve_path, setup_logging

    parser = argparse.ArgumentParser(
        description="Inspect the processed Wi-Fi CSI dataset")
    parser.add_argument("--dataset", default=None,
                        help="path to dataset CSV (default: config value)")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--stats", action="store_true",
                        help="print dataset statistics")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    default_path = resolve_path(cfg, "dataset_path")
    path = args.dataset or str(default_path)
    try:
        ds = Dataset.load_csv(path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    ds.validate()  # populates feature columns; problems shown in stats
    s = ds.stats()
    width = 46
    print("=" * width)
    print("Dataset statistics")
    print("=" * width)
    print(f"file              : {s['source'] if 'source' in s else path}")
    print(f"total samples     : {s['total_samples']:,}")
    print(f"sessions          : {s['sessions']}")
    print(f"feature columns   : {s['feature_columns']}")
    print(f"empty samples     : {s['empty_samples']:,}")
    print(f"person samples    : {s['person_samples']:,}")
    print(f"missing values    : {s['missing_values']:,}")
    print("-" * width)
    print("samples per position (person rows):")
    per_pos: Dict[str, int] = s["samples_per_position"]  # type: ignore[assignment]
    if not per_pos:
        print("  (none)")
    for pos, count in per_pos.items():
        print(f"  position {pos:>5}: {count:,}")
    print("-" * width)
    print("samples per session:")
    per_sess: Dict[str, int] = s["samples_per_session"]  # type: ignore[assignment]
    for sid, count in per_sess.items():
        print(f"  {sid}: {count:,}")
    problems = ds.validate()
    if problems:
        print("-" * width)
        print("WARNINGS (dataset quality):")
        for p in problems:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
