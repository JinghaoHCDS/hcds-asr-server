"""Admission before body parsing, plus the single framework parser boundary."""
from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from collections.abc import Callable, Coroutine
from time import perf_counter
from typing import Any, BinaryIO, cast

from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from python_multipart.multipart import parse_options_header
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from asr_server.audio.workspace import service_temporary_root
from asr_server.config import Settings
from asr_server.errors import AsrError, error_payload
from asr_server.observability import ExecutionContext, bind_context, current_context, emit


class _BodyRejected(BaseException):
    # FastAPI wraps ordinary exceptions during parsing as a generic HTTP 400.
    # Keep admission errors intact until our outer ASGI boundary.
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


class ControlledMultipartParser(MultiPartParser):
    """Starlette 1.x parser, with service-owned spools and bounded parts.

    The sole version-sensitive hook is kept here. Starlette's max_part_size
    limits non-file fields, not the uploaded file or entire request.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.metadata_limit = int(kwargs.pop("metadata_limit", 256 * 1024))
        self.file_limit = int(kwargs.pop("file_limit", 512 * 1024**2))
        self.metadata_bytes = self.file_bytes = 0
        super().__init__(*args, **kwargs)

    def _metadata(self, size: int) -> None:
        self.metadata_bytes += size
        if self.metadata_bytes > self.metadata_limit:
            raise _BodyRejected(413, "audio_too_large", "multipart metadata exceeds the server size limit")

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._metadata(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._metadata(end - start)
        super().on_header_value(data, start, end)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is None:
            self._metadata(end - start)
        else:
            self.file_bytes += end - start
            if self.file_bytes > self.file_limit:
                raise _BodyRejected(413, "audio_too_large", "audio upload exceeds the server size limit")
        super().on_part_data(data, start, end)

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._current_part.content_disposition)
        if b"name" not in options:
            raise MultiPartException("multipart field name is required")
        self._current_part.field_name = options[b"name"].decode(self._charset, errors="replace")
        if b"filename" not in options:
            self._current_fields += 1
            if self._current_fields > self.max_fields:
                raise MultiPartException("too many multipart fields")
            self._current_part.file = None
            return
        self._current_files += 1
        if self._current_files > self.max_files:
            raise MultiPartException("only one uploaded file is accepted")
        spool = tempfile.SpooledTemporaryFile(max_size=self.spool_max_size, dir=str(service_temporary_root()))
        self._files_to_close_on_error.append(spool)
        self._current_part.file = UploadFile(
            file=cast(BinaryIO, spool), size=0,
            filename=options[b"filename"].decode(self._charset, errors="replace"),
            headers=Headers(raw=self._current_part.item_headers),
        )

    async def parse(self) -> Any:
        try:
            return await super().parse()
        except BaseException:
            # Framework error handling only closes spools for its own parser
            # errors; include disconnect, timeout, admission and cancellation.
            for file in self._files_to_close_on_error:
                file.close()
            raise


class AdmissionRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            if request.headers.get("content-type", "").split(";", 1)[0].strip() != "multipart/form-data":
                return await original(request)
            parser = ControlledMultipartParser(request.headers, request.stream(), max_files=1,
                                               max_fields=32, max_part_size=request.app.state.settings.multipart_metadata_bytes,
                                               metadata_limit=request.app.state.settings.multipart_metadata_bytes,
                                               file_limit=request.app.state.settings.max_upload_mb * 1024**2)
            lease = request.scope.get("state", {}).get("body_lease")
            if lease is not None:
                original_stream = parser.stream
                async def accounted_stream() -> Any:
                    materialized = 0
                    async for block in original_stream:
                        yield block
                        written = sum(file.tell() for file in parser._files_to_close_on_error if getattr(file, "_rolled", False))
                        if written > materialized:
                            await lease.materialize(written - materialized)
                            materialized = written
                parser.stream = accounted_stream()
            try:
                request._form = await parser.parse()
                return await original(request)
            except MultiPartException as exc:
                raise AsrError(400, "bad_request", exc.message) from exc
            finally:
                for file in parser._files_to_close_on_error:
                    file.close()

        return handler


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        context = ExecutionContext.new()
        started, status = perf_counter(), 500
        scope.setdefault("state", {})["execution_context"] = context

        async def traced_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []),
                                      (b"x-request-id", (context.request_id or "").encode()),
                                      (b"x-trace-id", context.trace_id.encode())]
            await send(message)

        with bind_context(context):
            try:
                await self.app(scope, receive, traced_send)
            finally:
                route = scope.get("route")
                emit("request.rejected" if status >= 400 else "request.completed",
                     level=logging.DEBUG if scope["method"] == "GET" and status < 400 else logging.INFO,
                     route=getattr(route, "path", "unmatched"), status_code=status,
                     duration_ms=(perf_counter() - started) * 1000,
                     input_bytes=scope.get("state", {}).get("body_bytes", 0))


class AdmissionMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app, self.settings = app, settings
        self.receiving = 0

    def limit(self, path: str) -> int:
        if path in {"/v1/audio/transcriptions", "/v1/audio/transcription-jobs"}:
            return self.settings.max_upload_mb * 1024**2 + self.settings.multipart_metadata_bytes
        if re.fullmatch(r"/api/uploads/[^/]+", path):
            return 8 * 1024**2
        if path == "/api/admin/glossary":
            return 1024**2
        return 256 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith(("/api/", "/v1/")):
            await self.app(scope, receive, send)
            return
        leases: list[Any] = []
        admitted = False
        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            # Authenticate personal requests before JSON or multipart parsing.
            if path.startswith("/api/") and path not in {"/api/auth/login", "/api/auth/logout"}:
                service = getattr(scope["app"].state, "workbench", None)
                if service is not None:
                    from asr_server.workbench.security import COOKIE_NAME, check_same_origin
                    request = Request(scope)
                    check_same_origin(request)
                    user = await service.store.run(service.store.authenticate_session, request.cookies.get(COOKIE_NAME, ""))
                    if path.startswith("/api/admin/") and not user["is_admin"]:
                        raise AsrError(403, "forbidden", "administrator access required")
                    if scope["method"] == "PATCH" and re.fullmatch(r"/api/uploads/[^/]+", path):
                        await service.store.run(service.upload, user["id"], path.rsplit("/", 1)[1])
            maximum = self.limit(path)
            content_length = Headers(scope=scope).get("content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError:
                    raise _BodyRejected(400, "bad_request", "invalid Content-Length") from None
                if length < 0:
                    raise _BodyRejected(400, "bad_request", "invalid Content-Length")
                if length > maximum:
                    raise _BodyRejected(413, "audio_too_large", "request body exceeds the server size limit")
            if self.receiving >= self.settings.receive_slots:
                raise _BodyRejected(429, "receive_capacity_exceeded", "too many incoming request bodies")
            self.receiving += 1
            admitted = True
            total = 0
            manager = scope["app"].state.workspace_manager
            lease = await manager.reserve_resource(0, service_temporary_root())
            leases.append(lease)
            scope.setdefault("state", {})["body_lease"] = lease

            async def limited_receive() -> Message:
                nonlocal total, admitted
                try:
                    message = await asyncio.wait_for(receive(), self.settings.receive_timeout_seconds)
                except TimeoutError:
                    raise _BodyRejected(408, "request_timeout", "request body receive timed out") from None
                if message["type"] == "http.request":
                    block = message.get("body", b"")
                    total += len(block)
                    scope.setdefault("state", {})["body_bytes"] = total
                    if total > maximum:
                        raise _BodyRejected(413, "audio_too_large", "request body exceeds the server size limit")
                    if block:
                        try:
                            await lease.grow(len(block))
                        except AsrError as exc:
                            raise _BodyRejected(exc.status_code, exc.code, exc.message) from None
                    if not message.get("more_body", False) and admitted:
                        self.receiving -= 1
                        admitted = False
                return message

            await self.app(scope, limited_receive, tracked_send)
        except (_BodyRejected, AsrError) as exc:
            if response_started:
                raise
            status = exc.status if isinstance(exc, _BodyRejected) else exc.status_code
            emit("resource.rejected", error_code=exc.code)
            await JSONResponse(error_payload(exc.code, exc.message), status_code=status)(scope, receive, send)
        finally:
            for lease in leases:
                await lease.release()
            if admitted:
                self.receiving -= 1
