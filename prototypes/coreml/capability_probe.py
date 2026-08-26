#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan
import numpy as np


THRESHOLD = 100
MULTIPLIER = 3
ADDEND = 7


def device_name(device):
    return type(device).__name__


def build_program(row_count):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(row_count,), dtype=types.int32)],
        opset_version=ct.target.macOS15,
    )
    def program(values):
        condition = mb.greater(x=values, y=np.int32(THRESHOLD))
        multiplied = mb.mul(x=values, y=np.int32(MULTIPLIER))
        projected = mb.add(x=multiplied, y=np.int32(ADDEND))
        selected = mb.select(cond=condition, a=projected, b=np.int32(0))
        return mb.reduce_sum(x=selected, axes=[0], keep_dims=False)

    return program


def int64_capability():
    try:
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(16,), dtype=types.int32)],
            opset_version=ct.target.macOS15,
        )
        def program(values):
            return mb.cast(x=values, dtype="int64")

        return {"supported": True, "error": None}
    except Exception as error:
        return {"supported": False, "error": str(error)}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4_194_304)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.rows <= 0:
        parser.error("rows must be positive")
    return arguments


def main():
    arguments = parse_arguments()
    program = build_program(arguments.rows)
    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
    )
    compute_plan = MLComputePlan.load_from_path(
        model.get_compiled_model_path(), ct.ComputeUnit.ALL)
    operations = []
    for operation in compute_plan.model_structure.program.functions["main"].block.operations:
        usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        operations.append({
            "operator": operation.operator_name,
            "preferredDevice": device_name(usage.preferred_compute_device),
            "supportedDevices": [device_name(device) for device in usage.supported_compute_devices],
        })

    values = (np.arange(arguments.rows, dtype=np.int64) % 2_001 - 1_000).astype(np.int32)
    projected = np.where(
        values > THRESHOLD,
        values.astype(np.int64) * MULTIPLIER + ADDEND,
        0,
    )
    expected = int(projected.sum(dtype=np.int64))
    prediction = model.predict({"values": values})
    actual = int(next(iter(prediction.values())).item())

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "coremltoolsVersion": ct.__version__,
        "rows": arguments.rows,
        "availableDevices": [device_name(device) for device in model.get_available_compute_devices()],
        "operations": operations,
        "int64Cast": int64_capability(),
        "sparkCompatibleExpectedSum": expected,
        "coreMLInt32Sum": actual,
        "exactMatch": actual == expected,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
        print(f"Wrote {arguments.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
