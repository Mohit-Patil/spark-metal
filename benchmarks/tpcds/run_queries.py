#!/usr/bin/env python3

"""Run pinned TPC-DS SQL with repeat timings and deterministic result hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--queries-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--queries", default="q1", help="Comma-separated query names or 'all'")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def query_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    number = "".join(character for character in stem[1:] if character.isdigit())
    return (int(number), stem)


def select_queries(directory: Path, selection: str) -> list[Path]:
    available = {path.stem: path for path in directory.glob("q*.sql")}
    names = sorted(available, key=lambda name: query_sort_key(available[name])) if selection == "all" else selection.split(",")
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(f"Unknown query name(s): {', '.join(missing)}")
    return [available[name] for name in names]


def execute(spark: SparkSession, sql_text: str) -> tuple[float, int, str, str, str]:
    started = time.perf_counter()
    frame = spark.sql(sql_text)
    rows = frame.collect()
    elapsed = time.perf_counter() - started
    serialized = [
        json.dumps(row.asDict(recursive=True), default=str, separators=(",", ":"), sort_keys=True)
        for row in rows
    ]
    canonical = "\n".join(sorted(serialized)).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    plan = frame._jdf.queryExecution().executedPlan().toString()
    return elapsed, len(rows), digest, frame.schema.json(), plan


def main() -> None:
    args = arguments()
    if args.warmups < 0 or args.runs < 1:
        raise ValueError("warmups must be non-negative and runs must be positive")
    table_paths = sorted(path for path in args.data_dir.iterdir() if path.is_dir())
    if not table_paths:
        raise FileNotFoundError(f"No Parquet table directories under {args.data_dir}")
    queries = select_queries(args.queries_dir, args.queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName("spark-metal-tpcds-cpu").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spark_version": spark.version,
        "spark_conf": dict(spark.sparkContext.getConf().getAll()),
        "data_dir": str(args.data_dir.resolve()),
        "queries_dir": str(args.queries_dir.resolve()),
        "warmups": args.warmups,
        "runs": args.runs,
        "queries": {},
    }
    try:
        for table_path in table_paths:
            spark.read.parquet(str(table_path)).createOrReplaceTempView(table_path.name)

        for query_path in queries:
            print(f"Running {query_path.stem}")
            sql_text = query_path.read_text(encoding="utf-8")
            for _ in range(args.warmups):
                execute(spark, sql_text)
            observations: list[float] = []
            expected: tuple[int, str, str] | None = None
            final_plan = ""
            for _ in range(args.runs):
                elapsed, row_count, digest, schema_json, final_plan = execute(spark, sql_text)
                current = (row_count, digest, schema_json)
                if expected is not None and current != expected:
                    raise RuntimeError(f"Non-deterministic result detected for {query_path.stem}")
                expected = current
                observations.append(elapsed)
            assert expected is not None
            (args.output_dir / f"{query_path.stem}-plan.txt").write_text(final_plan, encoding="utf-8")
            summary["queries"][query_path.stem] = {
                "seconds": observations,
                "median_seconds": statistics.median(observations),
                "row_count": expected[0],
                "sha256": expected[1],
                "schema_json": expected[2],
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
