from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# Validation messages and context may embed the rejected password, transcript or
# parser input. Keep field locations and error codes, but use fixed public text.
_VALIDATION_MESSAGES = {
    "missing": "Field required",
    "json_invalid": "Invalid JSON",
    "string_type": "Input should be a string",
    "string_too_long": "String is too long",
    "string_too_short": "String is too short",
    "int_type": "Input should be an integer",
    "int_parsing": "Input should be an integer",
    "float_type": "Input should be a number",
    "float_parsing": "Input should be a number",
    "bool_type": "Input should be a boolean",
    "bool_parsing": "Input should be a boolean",
}


class AsrError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


async def asr_error_handler(_request: Request, exc: AsrError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.code, exc.message, exc.details),
    )


async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"type": error["type"], "loc": error["loc"],
         "msg": _VALIDATION_MESSAGES.get(error["type"], "Invalid value")}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content=error_payload("bad_request", "request validation failed", {"errors": errors}),
    )

