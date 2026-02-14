from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _validate_property(name: str, value: Any, prop_schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = prop_schema.get("type")
    expected_types: list[str] = expected if isinstance(expected, list) else [expected]
    if expected is not None and not any(_matches_type(value, t) for t in expected_types):
        errors.append(f"{name}: expected type {expected_types}, got {type(value).__name__}")
        return errors

    if "minimum" in prop_schema and isinstance(value, (int, float)) and value < prop_schema["minimum"]:
        errors.append(f"{name}: value {value} is less than minimum {prop_schema['minimum']}")
    if "maximum" in prop_schema and isinstance(value, (int, float)) and value > prop_schema["maximum"]:
        errors.append(f"{name}: value {value} is greater than maximum {prop_schema['maximum']}")
    if "enum" in prop_schema and value not in prop_schema["enum"]:
        errors.append(f"{name}: value {value!r} is not in enum {prop_schema['enum']}")
    return errors


def validate_trace_jsonl(*, trace_path: str | Path, schema_path: str | Path) -> list[str]:
    trace_file = Path(trace_path)
    schema_file = Path(schema_path)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    required = list(schema.get("required", []))
    properties = dict(schema.get("properties", {}))
    additional_properties = bool(schema.get("additionalProperties", True))

    errors: list[str] = []
    for line_no, raw in enumerate(trace_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue

        if not isinstance(obj, dict):
            errors.append(f"line {line_no}: expected JSON object")
            continue

        missing = [key for key in required if key not in obj]
        if missing:
            errors.append(f"line {line_no}: missing required keys: {missing}")
            continue

        if not additional_properties:
            unexpected = [key for key in obj if key not in properties]
            if unexpected:
                errors.append(f"line {line_no}: unexpected keys: {unexpected}")
                continue

        for key, prop_schema in properties.items():
            if key in obj and isinstance(prop_schema, dict):
                for err in _validate_property(key, obj[key], prop_schema):
                    errors.append(f"line {line_no}: {err}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EGA trace JSONL against a JSON schema.")
    parser.add_argument("trace_jsonl", help="Path to trace JSONL file.")
    parser.add_argument(
        "--schema",
        default="trace_schema.json",
        help="Path to schema file (default: trace_schema.json).",
    )
    args = parser.parse_args()

    errors = validate_trace_jsonl(trace_path=args.trace_jsonl, schema_path=args.schema)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
