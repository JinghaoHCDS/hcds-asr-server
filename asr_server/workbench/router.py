import secrets
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, Header, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.observability import log_stats
from asr_server.workbench.security import COOKIE_NAME, LoginFailures, check_same_origin, password_executor, password_hash, verify_password
from asr_server.workbench.service import CHUNK_SIZE, WorkbenchService, export_result


class Login(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=1024)


class NewUpload(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class CompleteUpload(BaseModel):
    domain: str = "general"
    hotwords: str = Field(default="", max_length=20000)
    language: str = "auto"
    idempotency_key: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=160)
    quality_profile: str | None = Field(default=None, max_length=80)


class EditSegment(BaseModel):
    text: str = Field(max_length=20000)
    base_version: int = Field(ge=1)


class NewUser(Login):
    is_admin: bool = False


class UpdateUser(BaseModel):
    password: str | None = Field(default=None, max_length=1024)
    disabled: bool | None = None


class GlossaryEntry(BaseModel):
    canonical: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=50)
    domain: str = "general"
    auto_normalize: bool = False


class UpdateGlossary(BaseModel):
    entries: list[GlossaryEntry] = Field(max_length=2000)
    base_version: int | None = None


def install_workbench(app: FastAPI, settings: Settings) -> WorkbenchService:
    service = WorkbenchService(app, settings)
    app.state.workbench = service
    router = APIRouter(prefix="/api", tags=["workbench"], dependencies=[Depends(check_same_origin)])
    failed_logins = LoginFailures()

    async def current_user(request: Request) -> dict[str, Any]:
        return await service.store.run(service.store.authenticate_session, request.cookies.get(COOKIE_NAME, ""))

    def admin(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
        if not user["is_admin"]:
            raise AsrError(403, "forbidden", "administrator access required")
        return user

    User = Annotated[dict[str, Any], Depends(current_user)]
    Admin = Annotated[dict[str, Any], Depends(admin)]

    @router.post("/auth/login")
    async def login(data: Login, request: Request, response: Response) -> dict[str, Any]:
        address = request.client.host if request.client else "unknown"
        recent = failed_logins.recent(address)
        if len(recent) >= 10:
            raise AsrError(429, "login_rate_limited", "too many sign-in attempts; try again in one minute")
        row = await service.store.run(service.store.user_for_login, data.username)
        valid = await password_executor.run(verify_password, data.password, row["password_hash"] if row is not None else "scrypt$00000000000000000000000000000000$" + "00" * 64)
        if row is None or row["disabled"] or not valid:
            failed_logins.failed(address)
            raise AsrError(401, "unauthorized", "incorrect username or password")
        failed_logins.clear(address)
        token = secrets.token_urlsafe(32)
        await service.store.run(service.store.session, token, row["id"], row["password_hash"])
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=7 * 86400, path="/api")
        response.headers["Cache-Control"] = "no-store"
        return {"user": service.store.public_user(row)}

    @router.post("/auth/logout", status_code=204)
    async def logout(request: Request, response: Response) -> None:
        await service.store.run(service.store.session, request.cookies.get(COOKIE_NAME, ""))
        response.delete_cookie(COOKIE_NAME, path="/api")

    @router.get("/auth/me")
    async def me(user: User) -> dict[str, Any]:
        return {"user": user}

    @router.get("/capabilities")
    async def capabilities(user: User) -> dict[str, Any]:
        default_model, default_profile = service.default_selection()
        return {
            "models": app.state.manager.list_models(),
            "default_model": default_model, "default_quality_profile": default_profile,
            "domains": [{"id": "general", "label": "通用"}, {"id": "ai", "label": "AI／技术内容"}],
            "quality_profiles": [{"id": "legacy", "label": "标准转录"}, {"id": "quality_v1", "label": "质量优先", "validated": False}],
            "max_upload_bytes": settings.workbench_max_upload_bytes, "chunk_size": CHUNK_SIZE,
            "max_duration_seconds": 21600, "retention_days": settings.workbench_retention_days,
            "storage": {"used_bytes": await service.store.run(service.store.usage, user["id"]), "quota_bytes": settings.workbench_user_quota_bytes},
        }

    @router.post("/uploads", status_code=201)
    async def create_upload(data: NewUpload, user: User) -> dict[str, Any]:
        return await service.create_upload(user["id"], data.filename, data.size_bytes)

    @router.get("/uploads/{upload_id}")
    async def get_upload(upload_id: str, user: User) -> dict[str, Any]:
        return await service.store.run(service.upload, user["id"], upload_id)

    @router.patch("/uploads/{upload_id}")
    async def append_upload(upload_id: str, request: Request, user: User,
                            upload_offset: Annotated[int, Header(ge=0)]) -> dict[str, Any]:
        # Authorization happens before consuming a potentially large body.
        await service.store.run(service.upload, user["id"], upload_id)
        data = bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data) > CHUNK_SIZE:
                raise AsrError(413, "upload_chunk_too_large", "upload chunks must not exceed eight MiB")
        return await service.append_upload(user["id"], upload_id, upload_offset, bytes(data))

    @router.post("/uploads/{upload_id}/complete", status_code=202)
    async def complete_upload(upload_id: str, data: CompleteUpload, user: User) -> dict[str, Any]:
        return await service.complete_upload(user["id"], upload_id, data.model_dump())

    @router.delete("/uploads/{upload_id}", status_code=204)
    async def delete_upload(upload_id: str, user: User) -> None:
        def remove() -> None:
            upload = service.upload(user["id"], upload_id)
            if upload["job_id"]:
                raise AsrError(409, "upload_completed", "delete the submitted task instead")
            service.upload_path(upload_id).unlink(missing_ok=True)
            with service.store.connection() as db:
                db.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
            service.store.set_physical_usage(upload_id, user["id"], 0)
        await service.store.run(service.upload, user["id"], upload_id)
        async with service.upload_lock(upload_id):
            async with service.lock:
                await service.store.run(remove)
                await service.sync_disk_reservations()

    @router.get("/jobs")
    async def jobs(user: User, view: str = "legacy", limit: int = 50, before: int | None = None,
                   active_only: bool = False, ids: str | None = None) -> dict[str, Any]:
        if view == "summary":
            result = await service.store.run(service.store.job_summaries, user["id"], limit=limit, before=before, active_only=active_only,
                                             job_ids=ids.split(",") if ids is not None else None)
            for item in result["jobs"]:
                item["queue_position"] = app.state.job_manager.external_queue_position(item["id"])
            return result
        if view != "legacy":
            raise AsrError(400, "bad_request", "unknown jobs view")
        values = await service.store.run(service.store.jobs, user["id"])
        return {"jobs": [service.public_job(job, include_result=False) for job in values]}

    @router.get("/jobs/{job_id}")
    async def job(job_id: str, request: Request, response: Response, user: User) -> Any:
        revision = await service.store.run(service.store.job_revision, job_id, user["id"])
        etag = f'"{job_id}:{revision}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "private, no-cache"})
        payload = await service.store.run(service.store.job, job_id, user["id"])
        response.headers["ETag"] = f'"{job_id}:{payload["revision"]}"'
        response.headers["Cache-Control"] = "private, no-cache"
        return service.public_job(payload)

    @router.post("/jobs/{job_id}/cancel")
    async def cancel(job_id: str, user: User) -> dict[str, Any]:
        return await service.cancel(user["id"], job_id)

    @router.post("/jobs/{job_id}/retry", status_code=202)
    async def retry(job_id: str, user: User) -> dict[str, Any]:
        return await service.retry(user["id"], job_id)

    @router.delete("/jobs/{job_id}")
    async def delete(job_id: str, user: User) -> Response:
        deleted = await service.delete(user["id"], job_id)
        return Response(status_code=204 if deleted else 202)

    @router.get("/jobs/{job_id}/audio")
    async def audio(job_id: str, user: User) -> FileResponse:
        await service.store.run(service.store.job_revision, job_id, user["id"])
        path = service.job_dir(job_id) / "audio.wav"
        if not path.exists():
            raise AsrError(409, "audio_not_ready", "audio extraction has not completed")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

    @router.patch("/jobs/{job_id}/segments/{segment_id}")
    async def edit(job_id: str, segment_id: str, data: EditSegment, user: User) -> dict[str, Any]:
        return await service.edit(user["id"], job_id, segment_id, data.text, data.base_version)

    @router.get("/jobs/{job_id}/export")
    async def export(job_id: str, user: User, format: str = "txt", version: int | None = None) -> Response:
        task = await service.store.run(service.store.job, job_id, user["id"])
        definition = app.state.manager.runtime_for(task["request"]["model"]).definition
        if format in {"srt", "vtt"} and not (definition.capabilities.forced_alignment or definition.capabilities.segment_timestamps):
            raise AsrError(422, "capability_not_supported", "selected model does not provide subtitle timing")
        if not task.get("result"):
            raise AsrError(409, "result_not_ready", "task has no complete transcript")
        result = await service.store.run(service.store.version, job_id, version) if version is not None else task["result"]
        text, media_type = await service.store.run(export_result, result, format,
            segment_timestamps=definition.capabilities.segment_timestamps)
        filename = quote(f"{task['filename']}.v{result['version']}.{format}")
        return Response(text, media_type=media_type, headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "Cache-Control": "private, no-store", "X-Transcript-Version": str(result["version"]),
        })

    @router.get("/admin/users")
    async def users(_user: Admin) -> dict[str, Any]:
        return {"users": await service.store.run(service.store.users)}

    @router.post("/admin/users", status_code=201)
    async def create_user(data: NewUser, _user: Admin) -> dict[str, Any]:
        return {"user": await password_executor.run(service.store.create_user, data.username, data.password, data.is_admin)}

    @router.patch("/admin/users/{user_id}")
    async def update_user(user_id: str, data: UpdateUser, user: Admin) -> dict[str, Any]:
        if user_id == user["id"] and data.disabled:
            raise AsrError(400, "bad_request", "cannot disable your current administrator account")
        hashed = await password_executor.run(password_hash, data.password) if data.password is not None else None
        return {"user": await service.store.run(service.store.update_user, user_id, hashed, data.disabled)}

    @router.get("/admin/glossary")
    async def glossary(_user: Admin) -> dict[str, Any]:
        return await service.store.run(service.store.glossary)

    @router.put("/admin/glossary")
    async def update_glossary(data: UpdateGlossary, _user: Admin) -> dict[str, Any]:
        entries = []
        for entry in data.entries:
            if entry.domain not in {"general", "ai"} or not entry.canonical.strip():
                raise AsrError(400, "bad_request", "invalid glossary entry")
            if any(not alias.strip() or len(alias) > 120 for alias in entry.aliases):
                raise AsrError(400, "bad_request", "aliases must contain 1 to 120 characters")
            entries.append({**entry.model_dump(), "canonical": entry.canonical.strip(), "aliases": [a.strip() for a in entry.aliases]})
        async with service.lock:
            updated = await service.store.run(service.store.update_glossary, entries, data.base_version)
        return updated

    @router.get("/admin/status")
    async def status(_user: Admin) -> dict[str, Any]:
        values = await service.store.run(service.store.job_summaries, active_only=True, limit=100)
        queue = []
        for task in reversed(values["jobs"]):
            progress = task.get("progress") or {}
            queue.append({**{key: task.get(key) for key in ("id", "status", "model", "created_at", "started_at")},
                "queue_position": app.state.job_manager.external_queue_position(task["id"]),
                "completed_chunks": progress.get("completed_chunks"), "total_chunks": progress.get("total_chunks"),
                "error_code": (task.get("error") or {}).get("code")})
        return {"storage": {"used_bytes": await service.store.run(service.store.usage), "quota_bytes": settings.workbench_total_quota_bytes},
                "queue": queue, "queue_has_more": values["next_cursor"] is not None,
                "users_count": len(await service.store.run(service.store.users)),
                "logging": log_stats()}

    app.include_router(router)
    return service
