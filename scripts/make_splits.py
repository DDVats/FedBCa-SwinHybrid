#!/usr/bin/env python3
"""Build patient-level train/validation/test splits.

Following Section 4.1, the split is stratified by MIBC/NMIBC class and
performed within each centre, so the class balance of every centre is
preserved and no patient contributes slices to more than one split. The
defaults reproduce the split of the recorded experiments (about 71 / 10 / 19).

Example::

    python scripts/make_splits.py \\
        --slice-index data/preprocessed/slice_index.csv \\
        --output-dir data/splits
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("make_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice-index",
        default="data/preprocessed/slice_index.csv",
        help="slice index written by scripts/preprocess.py",
    )
    parser.add_argument("--output-dir", default="data/splits", help="output directory")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_patients(patients: pd.DataFrame, args) -> dict:
    """Assign every patient to a split, stratified by centre and MIBC label.

    Patients are visited in ``patient_id`` order and each centre-and-class
    group is shuffled with one seeded generator. Per group, ``round(0.20 n)``
    patients (at least one) go to test and ``round(0.10 n)`` (at least one) go
    to validation; the rest train. With seed 42 this is the split of the
    recorded experiments (157 / 22 / 42 patients, about 71 / 10 / 19 %).
    """
    rng = np.random.default_rng(args.seed)
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    patients["strat"] = patients["center"].astype(str) + "_" + patients["label"].astype(str)

    assignment: dict[str, str] = {}
    for _, group in patients.groupby("strat"):
        ids = group["patient_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * args.test_frac))
        n_val = max(1, round(n * args.val_frac))
        n_train = n - n_test - n_val
        if n_train < 1:
            assignment.update({pid: "train" for pid in ids})
            continue
        assignment.update({pid: "train" for pid in ids[:n_train]})
        assignment.update({pid: "val" for pid in ids[n_train : n_train + n_val]})
        assignment.update({pid: "test" for pid in ids[n_train + n_val :]})
    return assignment


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    index_path = Path(args.slice_index)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Slice index not found: {index_path}. Run scripts/preprocess.py first."
        )

    slices = pd.read_csv(index_path)
    patients = (
        slices[["patient_id", "center", "label"]]
        .drop_duplicates(subset="patient_id")
        .reset_index(drop=True)
    )

    assignment = split_patients(patients, args)
    slices["split"] = slices["patient_id"].map(assignment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        subset = slices[slices["split"] == split].drop(columns="split")
        subset.to_csv(output_dir / f"{split}.csv", index=False)
        logger.info(
            "%-5s %5d slices / %3d patients",
            split,
            len(subset),
            subset["patient_id"].nunique(),
        )

    logger.info("Splits written to %s", output_dir)


if __name__ == "__main__":
    main()
