"""Inference-with-models test — verifies the trained-artifact path.

Loads the models saved by train.py (data/models), streams a simulated
person session (generated with the exact same code as the synthetic
dataset) through the real InferenceEngine and asserts presence + position
are produced.  Synthetic CSI only, never real measurements.

    python tests/inference_model_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))


def main() -> int:
    import numpy as np
    import make_synthetic_dataset as syn
    from csi_parser import parse_line
    from config import load_config, resolve_path
    from inference import InferenceEngine, load_models

    cfg = load_config()
    models = load_models(resolve_path(cfg, "model_path"), cfg)
    if models is None:
        print("No models found - run train.py with a dataset first.")
        return 1

    engine = InferenceEngine(cfg, models)
    rng = np.random.default_rng(3)
    phase0 = rng.uniform(-np.pi, np.pi, len(syn.VALID_SUBCARRIERS_20MHZ))

    # 3 seconds of a person standing at position 0.25 (same generator as
    # make_synthetic_dataset, so the distribution matches training).
    for k in range(3 * syn.RATE):
        line = syn._packet(rng, phase0, pos=0.25, person=True, seq=k,
                           ts_us=int(k * 1e6 / syn.RATE))
        engine.process_sample(parse_line(line, time.time() + k / syn.RATE))

    s = engine.state
    print("presence:", s.presence_label, "| P(person)=",
          None if s.presence_probability is None
          else round(s.presence_probability, 3))
    print("movement:", s.movement_label,
          "| P(movement)=", None if s.movement_probability is None
          else round(s.movement_probability, 3))
    print("position_filtered:", s.position_filtered,
          "| position_raw:", s.position_raw, "| conf:", s.confidence)
    print("model:", s.model_name)
    print("latency_ms:", round(s.latency_ms, 3))
    if s.presence_label != "PERSON" or s.position_filtered is None:
        print("INFERENCE MODEL PATH FAILED")
        return 1
    if not 0.0 <= s.position_filtered <= 1.0:
        print("INFERENCE MODEL PATH FAILED: position out of range")
        return 1
    print("INFERENCE MODEL PATH OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())