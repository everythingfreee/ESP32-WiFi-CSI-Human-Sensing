import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from csi_parser import parse_line
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor


REAL_CSV = Path(__file__).resolve().parent.parent / "data/raw/20260901_164411_empty_static.csv"


def _real_packet_from_csv(path: Path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        row = next(reader)

    csi_len = int(row[22])
    csi_values = [int(v) for v in row[23:23 + csi_len]]
    obj = {
        "timestamp": int(float(row[1])),
        "seq": int(row[3]),
        "rssi": int(row[4]),
        "channel": int(row[5]),
        "secondary_channel": int(row[6]),
        "sig_mode": int(row[7]),
        "mcs": int(row[8]),
        "cwb": int(row[9]),
        "rate": int(row[10]),
        "aggregation": int(row[11]),
        "stbc": int(row[12]),
        "fec_coding": int(row[13]),
        "sgi": int(row[14]),
        "noise_floor": int(row[15]),
        "ampdu_cnt": int(row[16]),
        "sig_len": int(row[17]),
        "rx_state": int(row[18]),
        "ant": int(row[19]),
        "timestamp_wifi": int(row[2]),
        "mac": row[20],
        "first_word_invalid": int(row[21]),
        "csi_len": csi_len,
        "csi": csi_values,
    }
    return json.dumps(obj)


def test_real_packet_parses_and_passes_preprocessing_and_features():
    packet = _real_packet_from_csv(REAL_CSV)
    sample = parse_line(packet, 1788264851.344083)
    assert sample is not None
    assert sample.csi is not None
    assert len(sample.csi) > 0
    assert sample.first_word_invalid is True
    assert sample.csi_len == 256

    preproc_cfg = PreprocessConfig()
    preprocessor = Preprocessor(preproc_cfg)
    amp = preprocessor.process(sample)
    assert amp is not None, preprocessor.stats.as_dict()
    assert len(amp) == len(preproc_cfg.valid_subcarriers)

    extractor = FeatureExtractor(FeatureConfig(), preproc_cfg)
    fv = extractor.update(amp, sample.rssi, sample.ts_local)
    assert fv is not None, preprocessor.stats.as_dict()
    assert len(fv.values) == len(extractor.feature_names())


def test_real_packet_rejection_reason_is_not_corruption():
    packet = _real_packet_from_csv(REAL_CSV)
    sample = parse_line(packet, 1788264851.344083)
    preproc_cfg = PreprocessConfig()
    preprocessor = Preprocessor(preproc_cfg)
    amp = preprocessor.process(sample)
    assert amp is not None
    stats = preprocessor.stats.as_dict()
    assert stats["accepted"] >= 1
    assert stats["rejected_total"] == 0
