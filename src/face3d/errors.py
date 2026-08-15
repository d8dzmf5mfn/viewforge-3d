from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, NoReturn


@dataclass(slots=True)
class Face3DError(Exception):
    code: str
    message: str
    stage: str
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 2

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "stage": self.stage,
                "details": self.details,
            },
        }


def fail(
    code: str,
    message: str,
    *,
    stage: str,
    details: dict[str, Any] | None = None,
    exit_code: int = 2,
) -> NoReturn:
    raise Face3DError(code, message, stage, details or {}, exit_code)


def error_json(error: Face3DError) -> str:
    return json.dumps(error.as_dict(), ensure_ascii=False, sort_keys=True)
