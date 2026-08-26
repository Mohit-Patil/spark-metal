#!/usr/bin/env python3

"""Convert pipe-delimited TPC-DS generator output to typed Parquet tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def split_columns(body: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(body):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            pieces.append(body[start:index].strip())
            start = index + 1
    pieces.append(body[start:].strip())
    return pieces


def spark_type(type_name: str):
    normalized = type_name.lower()
    if normalized == "integer":
        return IntegerType()
    if normalized == "date":
        return DateType()
    if normalized.startswith(("char(", "varchar(", "time")):
        return StringType()
    decimal_match = re.fullmatch(r"decimal\((\d+),(\d+)\)", normalized)
    if decimal_match:
        return DecimalType(int(decimal_match.group(1)), int(decimal_match.group(2)))
    raise ValueError(f"Unsupported TPC-DS type: {type_name}")


def parse_schemas(ddl_path: Path) -> dict[str, StructType]:
    ddl = ddl_path.read_text(encoding="utf-8", errors="replace")
    schemas: dict[str, StructType] = {}
    pattern = re.compile(r"create\s+table\s+(\w+)\s*\((.*?)\);", re.I | re.S)
    for table_name, body in pattern.findall(ddl):
        fields: list[StructField] = []
        for definition in split_columns(body):
            if definition.lower().startswith("primary key"):
                continue
            match = re.match(
                r"^(\w+)\s+(integer|date|time|char\(\d+\)|varchar\(\d+\)|decimal\(\d+\s*,\s*\d+\))",
                definition,
                re.I,
            )
            if not match:
                raise ValueError(f"Cannot parse {table_name} column: {definition}")
            column_name, type_name = match.groups()
            fields.append(StructField(column_name, spark_type(type_name.replace(" ", "")), True))
        schemas[table_name] = StructType(fields)
    return schemas


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--ddl", required=True, type=Path)
    parser.add_argument("--tables", help="Comma-separated subset; default is every generated table")
    parser.add_argument("--validate-schema-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    schemas = parse_schemas(args.ddl)
    if args.validate_schema_only:
        print(f"Validated {len(schemas)} table schemas from {args.ddl}")
        return
    requested = set(args.tables.split(",")) if args.tables else None
    spark = SparkSession.builder.appName("spark-metal-tpcds-prepare").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        for table_name, schema in schemas.items():
            if table_name == "dbgen_version" or (requested and table_name not in requested):
                continue
            sources = sorted(args.raw_dir.glob(f"{table_name}*.dat"))
            if not sources:
                raise FileNotFoundError(f"No source files found for {table_name}")
            destination = args.parquet_dir / table_name
            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")
            print(f"Converting {table_name}: {len(sources)} source file(s)")
            frame = (
                spark.read.schema(schema)
                .option("sep", "|")
                .option("nullValue", "")
                .option("dateFormat", "yyyy-MM-dd")
                .csv([str(path) for path in sources])
            )
            frame.write.mode("errorifexists").parquet(str(destination))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
