import importlib
import sys
from pathlib import Path

import numpy as np


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _load_aaindex_package():
    # This file is also named aaindex.py, so temporarily hide it from import.
    this_module = sys.modules.get(__name__)
    old_path = list(sys.path)
    script_dir = Path(__file__).resolve().parent

    if sys.modules.get("aaindex") is this_module:
        sys.modules.pop("aaindex")

    sys.path = [
        p for p in sys.path
        if Path(p or ".").resolve() != script_dir
    ]
    try:
        pkg = importlib.import_module("aaindex")
    finally:
        sys.path = old_path
        sys.modules[__name__] = this_module

    return pkg


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _build_aaindex1_table():
    pkg = _load_aaindex_package()
    record_ids = pkg.aaindex1.record_codes
    if callable(record_ids):
        record_ids = record_ids()
    record_ids = sorted(record_ids)

    columns = []
    used_ids = []
    for record_id in record_ids:
        record = pkg.aaindex1[record_id]
        record_values = record["values"] if "values" in record else record
        values = np.array([_as_float(record_values[aa]) for aa in AA_ORDER], dtype=np.float32)
        if np.isnan(values).all():
            continue
        mean = np.nanmean(values)
        values = np.where(np.isnan(values), mean, values)
        columns.append(values)
        used_ids.append(record_id)

    table = np.stack(columns, axis=1).astype(np.float32)
    std = table.std(axis=0)
    std[std == 0] = 1.0
    table = (table - table.mean(axis=0)) / std

    aa_table = {aa: table[i] for i, aa in enumerate(AA_ORDER)}
    aa_table["X"] = np.zeros(table.shape[1], dtype=np.float32)
    return aa_table, table.shape[1], used_ids


AAINDEX_TABLE, AAINDEX_DIM, AAINDEX_IDS = _build_aaindex1_table()


def encode_aaindex(seq):
    return np.stack(
        [AAINDEX_TABLE.get(str(aa).upper(), AAINDEX_TABLE["X"]) for aa in seq]
    ).astype(np.float32)
