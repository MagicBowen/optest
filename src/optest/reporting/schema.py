"""JSON schema definition for reporter output."""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

JSON_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "optest report",
    "type": "object",
    "required": ["schema_version", "generated_at", "summary", "cases"],
    "properties": {
        "schema_version": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "summary": {
            "type": "object",
            "required": ["total", "passed", "failed", "errors", "backend", "seed", "duration_s"],
            "properties": {
                "total": {"type": "integer"},
                "passed": {"type": "integer"},
                "failed": {"type": "integer"},
                "errors": {"type": "integer"},
                "backend": {"type": "string"},
                "chip": {"type": ["string", "null"]},
                "seed": {"type": "integer"},
                "duration_s": {"type": "number"},
            },
        },
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "operator",
                    "status",
                    "duration_ms",
                    "backend",
                    "dtypes",
                    "shapes",
                    "tolerance",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "operator": {"type": "string"},
                    "status": {"type": "string"},
                    "duration_ms": {"type": "number"},
                    "backend": {"type": "string"},
                    "chip": {"type": ["string", "null"]},
                    "dtypes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "shapes": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                    },
                    "tolerance": {
                        "type": "object",
                        "required": ["abs", "rel"],
                        "properties": {
                            "abs": {"type": "number"},
                            "rel": {"type": "number"},
                        },
                    },
                    "attributes": {"type": "object"},
                    "error": {"type": "string"},
                    "comparison": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "tensor_index",
                                "passed",
                                "max_abs_error",
                                "max_rel_error",
                                "mismatched",
                                "total",
                            ],
                            "properties": {
                                "tensor_index": {"type": "integer"},
                                "passed": {"type": "boolean"},
                                "max_abs_error": {"type": "number"},
                                "max_rel_error": {"type": "number"},
                                "mismatched": {"type": "integer"},
                                "total": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        },
    },
}
