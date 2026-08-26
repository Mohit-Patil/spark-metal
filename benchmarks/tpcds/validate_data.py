#!/usr/bin/env python3

"""Validate row counts and schemas for generated TPC-DS Parquet tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        metavar="TABLE=ROWS",
        help="Expected row count; may be specified repeatedly",
    )
    return parser.parse_args()


def parse_expected(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        table, separator, rows = value.partition("=")
        if not separator or not table or not rows.isdigit():
            raise ValueError(f"Expected TABLE=ROWS, received: {value}")
        result[table] = int(rows)
    if not result:
        raise ValueError("At least one --expected TABLE=ROWS value is required")
    return result


def main() -> None:
    args = arguments()
    expected = parse_expected(args.expected)
    spark = SparkSession.builder.appName("spark-metal-tpcds-validate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    observations: dict[str, object] = {}
    try:
        for table, expected_rows in expected.items():
            path = args.parquet_dir / table
            if not (path / "_SUCCESS").is_file():
                raise FileNotFoundError(f"Incomplete Parquet table: {path}")
            frame = spark.read.parquet(str(path))
            actual_rows = frame.count()
            if actual_rows != expected_rows:
                raise RuntimeError(
                    f"{table} row-count mismatch: expected={expected_rows}, actual={actual_rows}"
                )
            observations[table] = {
                "rows": actual_rows,
                "schema": frame.schema.simpleString(),
            }
    finally:
        spark.stop()
    print(json.dumps(observations, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
