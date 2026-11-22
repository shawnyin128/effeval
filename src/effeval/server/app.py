"""Lab server: HTTP wrapper over api.service plus authentication. Never imports execution code."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.gzip import GZipMiddleware

import effeval
from effeval.api import service
from effeval.api.service import Actor, ServiceError
from effeval.core import bundle as bundle_io
from effeval.core.catalog import ModelEntry
from effeval.core.listing import TaskListing
from effeval.core.manifest import MethodManifest
from effeval.core.types import Artifact, Visibility, utc_now
from effeval.server.events import change_stream
from effeval.store.sqlite import SqliteStore

SESSION_COOKIE = "effeval_session"
SESSION_DAYS = 14
TOKEN_DAYS = 90
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("password must have at least 8 characters")
    return _hasher.hash(password)


class LoginBody(BaseModel):
    username: str
    password: str


class PasswordTokenBody(LoginBody):
    label: str = "cli"
    can_publish: bool = True


USERNAME_PATTERN = r"^[a-z][a-z0-9_.-]{1,31}$"
ROLE_PATTERN = "^(admin|member)$"


class NewUserBody(BaseModel):
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str
    role: str = Field(default="member", pattern=ROLE_PATTERN)


class UserUpdateBody(BaseModel):
    username: str | None = Field(default=None, pattern=USERNAME_PATTERN)
    role: str | None = Field(default=None, pattern=ROLE_PATTERN)


class PasswordBody(BaseModel):
    password: str


class LocaleBody(BaseModel):
    locale: str = Field(pattern="^(en|zh)$")


class SpecBody(BaseModel):
    spec: dict[str, Any]
    shards: int = 1
    to: str = "private"


class PathBody(BaseModel):
    path: str = Field(min_length=1)


class TitleBody(BaseModel):
    title: str = Field(max_length=200)


class RedoBody(BaseModel):
    to: str = "private"


class ProjectBody(BaseModel):
    name: str
    description: str = ""


class ProjectUpdateBody(BaseModel):
    name: str | None = None
    description: str | None = None


class MemberBody(BaseModel):
    username: str


class CellsBody(BaseModel):
    run_keys: list[str] = Field(min_length=1)
    force: bool = False


class LinkBody(BaseModel):
    record_ids: list[str] = Field(min_length=1)


class AnnotationBody(BaseModel):
    tags: list[str] = []
    note: str = ""
    status: str = "unreviewed"


class HeartbeatBody(BaseModel):
    plan_id: str
    job_id: str
    status: str = Field(pattern="^(running|done|failed)$")
    message: str = ""


def create_app(data_dir: Path, web_dist: Path | None = None, public_url: str | None = None) -> FastAPI:
    store = SqliteStore(data_dir)
    service.backfill_all(store)
    app = FastAPI(title="effeval", version=effeval.__version__)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    def verify(username: str, password: str) -> Any:
        user = store.user(username)
        if user is None or user["disabled"]:
            raise HTTPException(401, "invalid username or password")
        try:
            _hasher.verify(user["password_hash"], password)
        except VerifyMismatchError as exc:
            raise HTTPException(401, "invalid username or password") from exc
        return user

    def current(request: Request) -> Actor:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            now = utc_now()
            owner = store.token_owner(auth[len("Bearer ") :], now)
            if owner is None:
                raise HTTPException(401, "invalid or revoked login token; run `effeval login` again")
            if owner["expires_at"] is not None and owner["expires_at"] <= now:
                raise HTTPException(401, "login token expired; run `effeval login` again")
            return Actor(
                username=owner["username"], role=owner["role"], can_publish=bool(owner["can_publish"])
            )
        cookie = request.cookies.get(SESSION_COOKIE)
        if cookie:
            owner = store.session_owner(cookie, utc_now())
            if owner is not None:
                return Actor(username=owner["username"], role=owner["role"], can_publish=True)
        raise HTTPException(401, "login required")

    def admin(actor: Actor = Depends(current)) -> Actor:
        if actor.role != "admin":
            raise HTTPException(403, "admin only")
        return actor

    def server_url(request: Request) -> str:
        return public_url or str(request.base_url).rstrip("/")

    # health and auth ----------------------------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": effeval.__version__}

    @app.post("/api/auth/login")
    def login(body: LoginBody, response: Response) -> dict[str, str]:
        user = verify(body.username, body.password)
        expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=SESSION_DAYS)
        secret = store.create_session(user["username"], expires.isoformat(timespec="seconds"))
        response.set_cookie(
            SESSION_COOKIE, secret, httponly=True, samesite="lax", max_age=SESSION_DAYS * 86400
        )
        return {"username": user["username"], "role": user["role"]}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        if cookie := request.cookies.get(SESSION_COOKIE):
            store.delete_session(cookie)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @app.post("/api/auth/token")
    def token_with_password(body: PasswordTokenBody) -> dict[str, str]:
        """Exchange a password for this machine's login token (`effeval login`), valid TOKEN_DAYS days."""
        user = verify(body.username, body.password)
        expires = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=TOKEN_DAYS)).isoformat(
            timespec="seconds"
        )
        token_id, secret = store.create_token(user["username"], body.label, body.can_publish, expires)
        return {"token_id": token_id, "token": secret, "username": user["username"], "expires_at": expires}

    @app.get("/api/events")
    def events(request: Request, actor: Actor = Depends(current)) -> StreamingResponse:
        """Server-sent events naming the kinds of data that changed (topics only, never the data)."""
        return StreamingResponse(
            change_stream(store, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/me")
    def me(actor: Actor = Depends(current)) -> dict[str, Any]:
        return {**actor.model_dump(), "locale": store.user(actor.username)["locale"]}

    @app.put("/api/me/locale")
    def set_locale(body: LocaleBody, actor: Actor = Depends(current)) -> dict[str, str]:
        """Change the UI language of the calling user only."""
        store.update_user(actor.username, locale=body.locale)
        return {"locale": body.locale}

    @app.put("/api/me/password")
    def change_password(body: PasswordBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        try:
            store.update_user(actor.username, password_hash=hash_password(body.password))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    @app.get("/api/admin/tokens")
    def list_tokens(_admin: Actor = Depends(admin)) -> list[dict[str, Any]]:
        return store.active_tokens()

    @app.delete("/api/admin/tokens/{token_id}")
    def revoke_token(token_id: str, _admin: Actor = Depends(admin)) -> dict[str, bool]:
        if not store.revoke_token(token_id):
            raise HTTPException(404, "no such active token")
        return {"ok": True}

    @app.get("/api/admin/users")
    def list_users(_admin: Actor = Depends(admin)) -> list[dict[str, Any]]:
        return store.users()

    @app.post("/api/admin/users")
    def add_user(body: NewUserBody, _admin: Actor = Depends(admin)) -> dict[str, str]:
        if store.user(body.username) is not None:
            raise HTTPException(409, "user exists")
        try:
            store.create_user(body.username, hash_password(body.password), body.role)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"username": body.username, "role": body.role}

    @app.patch("/api/admin/users/{username}")
    def update_user(username: str, body: UserUpdateBody, actor: Actor = Depends(admin)) -> dict[str, str]:
        """Change a user's role and/or username. Admins cannot change their own role."""
        user = store.user(username)
        if user is None:
            raise HTTPException(404, "no such user")
        renaming = body.username is not None and body.username != username
        if renaming and store.user(body.username) is not None:
            raise HTTPException(409, "user exists")
        role = body.role or user["role"]
        if role != user["role"]:
            if username == actor.username:
                raise HTTPException(400, "admins cannot change their own role")
            store.update_user(username, role=role)
        if renaming:
            try:
                store.rename_user(username, body.username)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            username = body.username
        return {"username": username, "role": role}

    @app.delete("/api/admin/users/{username}")
    def delete_user(username: str, actor: Actor = Depends(admin)) -> dict[str, bool]:
        """Delete a user who has no data; users with data must be disabled instead."""
        if username == actor.username:
            raise HTTPException(400, "admins cannot delete themselves")
        if store.user(username) is None:
            raise HTTPException(404, "no such user")
        owned = store.user_data(username)
        if owned:
            listing = ", ".join(f"{table}: {n}" for table, n in owned.items())
            raise HTTPException(409, f"{username} has data ({listing}); disable the account instead")
        try:
            store.delete_user(username)
        except ValueError as exc:
            raise HTTPException(409, f"{exc}; disable the account instead") from exc
        return {"ok": True}

    @app.post("/api/admin/users/{username}/disable")
    def disable_user(username: str, actor: Actor = Depends(admin)) -> dict[str, bool]:
        if username == actor.username:
            raise HTTPException(400, "admins cannot disable themselves")
        store.update_user(username, disabled=1)
        return {"ok": True}

    @app.post("/api/admin/users/{username}/enable")
    def enable_user(username: str, _admin: Actor = Depends(admin)) -> dict[str, bool]:
        store.update_user(username, disabled=0)
        return {"ok": True}

    @app.post("/api/admin/users/{username}/password")
    def reset_password(username: str, body: PasswordBody, _admin: Actor = Depends(admin)) -> dict[str, bool]:
        try:
            store.update_user(username, password_hash=hash_password(body.password))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    # catalog, validation, rendering -------------------------------------------------------------------------

    @app.get("/api/describe")
    def describe(actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.describe(store, actor.username)

    @app.post("/api/validate")
    def validate(body: SpecBody, actor: Actor = Depends(current)) -> dict[str, Any]:
        return {
            "issues": [
                i.model_dump(mode="json") for i in service.validate_spec(store, actor.username, body.spec)
            ]
        }

    @app.post("/api/render")
    def render(body: SpecBody, request: Request, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.render(store, actor, body.spec, server_url(request), body.shards, body.to).model_dump(
            mode="json"
        )

    @app.post("/api/reuse-preview")
    def reuse_preview(body: SpecBody, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.reuse_preview(store, actor, body.spec)

    @app.get("/api/reuse")
    def reuse(
        run_key: str, need: str = "", hardware: str | None = None, actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        artifacts = frozenset(Artifact(a) for a in need.split(",") if a)
        record = store.find_reusable(run_key, viewer=actor.username, need=artifacts, hardware=hardware)
        return {"record": None if record is None else record.model_dump(mode="json")}

    @app.get("/api/runs")
    def runs(actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return service.runs(store, actor)

    @app.patch("/api/runs/{plan_id}")
    def rename_run(plan_id: str, body: TitleBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.rename_run(store, actor, plan_id, body.title)
        return {"ok": True}

    @app.delete("/api/runs/{plan_id}")
    def delete_run(plan_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.delete_run(store, actor, plan_id)
        return {"ok": True}

    @app.post("/api/runs/{plan_id}/command")
    def run_command(
        plan_id: str, request: Request, body: RedoBody | None = None, actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        to = body.to if body else "private"
        return service.run_command(store, actor, plan_id, server_url(request), to).model_dump()

    @app.post("/api/runs/{plan_id}/resume")
    def resume(
        plan_id: str, request: Request, body: RedoBody | None = None, actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        to = body.to if body else "private"
        return service.resume_run(store, actor, plan_id, server_url(request), to).model_dump(mode="json")

    @app.get("/api/settings/storage")
    def storage(actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.storage(store, actor)

    @app.put("/api/settings/storage/shared")
    def set_shared_storage(body: PathBody, actor: Actor = Depends(admin)) -> dict[str, int]:
        return {"moved": service.set_storage(store, actor, shared=True, path=body.path)}

    @app.put("/api/me/storage")
    def set_my_storage(body: PathBody, actor: Actor = Depends(current)) -> dict[str, int]:
        return {"moved": service.set_storage(store, actor, shared=False, path=body.path)}

    @app.post("/api/heartbeat")
    def heartbeat(body: HeartbeatBody, _actor: Actor = Depends(current)) -> dict[str, bool]:
        store.heartbeat(body.plan_id, body.job_id, body.status, body.message)
        return {"ok": True}

    # results ------------------------------------------------------------------------------------------------

    @app.post("/api/artifacts")
    def upload_artifact(file: UploadFile, actor: Actor = Depends(current)) -> dict[str, str]:
        try:
            return {"uri": store.save_artifact(file.file.read(), actor.username)}
        except (ValueError, OSError, ValidationError) as exc:
            raise HTTPException(400, f"not a per-sample artifact: {exc}") from exc

    @app.post("/api/bundles")
    def import_bundle(
        text: str = Body(..., media_type="text/plain"),
        via: str = "paste",
        to: str = "private",
        actor: Actor = Depends(current),
    ) -> dict[str, Any]:
        if via not in {"cli", "paste", "mcp", "api"}:
            raise HTTPException(400, "via must be cli, paste, mcp or api")
        try:
            parsed = bundle_io.loads(text)
        except bundle_io.BundleError as exc:
            raise HTTPException(400, str(exc)) from exc
        return service.import_bundle(store, actor, parsed, via, to).model_dump()

    @app.post("/api/packages")
    def import_package(
        file: UploadFile, to: str = "private", actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        return service.import_package(store, actor, file.file.read(), to).model_dump()

    @app.post("/api/packages/preview")
    def preview_package(file: UploadFile, actor: Actor = Depends(current)) -> dict[str, Any]:
        """Rows of a results package and how many carry their per-sample file; stores nothing."""
        try:
            parsed, files = bundle_io.read_package(file.file.read())
        except bundle_io.BundleError as exc:
            raise HTTPException(400, str(exc)) from exc
        catalog = service.catalog_for(store, None)
        return {"rows": [service.row(store, catalog, r) for r in parsed.records], "with_samples": len(files)}

    @app.post("/api/records/{record_id}/evidence")
    def attach_evidence(record_id: str, file: UploadFile, actor: Actor = Depends(current)) -> dict[str, Any]:
        try:
            uri = store.save_artifact(file.file.read(), actor.username)
        except (ValueError, OSError, ValidationError) as exc:
            raise HTTPException(400, f"not a per-sample artifact: {exc}") from exc
        return {"comparisons_updated": service.attach_evidence(store, actor, record_id, uri)}

    @app.post("/api/bundles/preview")
    def preview_bundle(
        text: str = Body(..., media_type="text/plain"), actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        """Parse and verify a pasted bundle without storing it, for the Import page's table preview.

        Each row's `replaces` lists the records marked for redo that importing it would replace.
        """
        try:
            parsed = bundle_io.loads(text)
        except bundle_io.BundleError as exc:
            raise HTTPException(400, str(exc)) from exc
        catalog = service.catalog_for(store, None)
        rows = []
        for record in parsed.records:
            pending = store.redo_pending(record.run_key, actor.username)
            rows.append({**service.row(store, catalog, record), "replaces": [r.record_id for r in pending]})
        return {"rows": rows}

    @app.get("/api/records")
    def records(
        request: Request, scope: str = "mine", actor: Actor = Depends(current)
    ) -> list[dict[str, Any]]:
        filters = {k: v for k, v in request.query_params.items() if k != "scope"}
        return service.query(store, actor.username, scope, filters)

    @app.get("/api/records/{record_id}")
    def record(record_id: str, actor: Actor = Depends(current)) -> dict[str, Any]:
        found = store.record(record_id)
        if found is None or not store.can_view(actor.username, record_id):
            raise HTTPException(404, "no such record")
        return service.row(store, service.catalog_for(store, actor.username), found, actor.username)

    @app.get("/api/records/{record_id}/samples")
    def samples(
        record_id: str,
        against: str | None = None,
        flip: str = "changed",
        subtask: str | None = None,
        limit: int = 100,
        offset: int = 0,
        actor: Actor = Depends(current),
    ) -> dict[str, Any]:
        return service.sample_diff(
            store, actor.username, record_id, against, flip=flip, subtask=subtask, limit=limit, offset=offset
        )

    @app.get("/api/records/{record_id}/dependents")
    def dependents(record_id: str, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.dependents(store, actor, record_id)

    @app.delete("/api/records/{record_id}")
    def trash(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.trash(store, actor, record_id)
        return {"ok": True}

    @app.post("/api/records/{record_id}/restore")
    def restore(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.restore(store, actor, record_id)
        return {"ok": True}

    @app.post("/api/records/{record_id}/redo")
    def redo(
        record_id: str, request: Request, body: RedoBody | None = None, actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        to = body.to if body else "private"
        return service.redo(store, actor, record_id, server_url(request), to).model_dump(mode="json")

    @app.delete("/api/records/{record_id}/redo")
    def cancel_redo(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.cancel_redo(store, actor, record_id)
        return {"ok": True}

    @app.get("/api/trash")
    def trash_list(actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return service.trashed(store, actor)

    @app.delete("/api/trash/{record_id}")
    def purge(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.purge(store, actor, record_id)
        return {"ok": True}

    # projects -----------------------------------------------------------------------------------------------

    @app.get("/api/projects")
    def projects(actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return service.projects(store, actor)

    @app.post("/api/projects")
    def create_project(body: ProjectBody, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.create_project(store, actor, body.name, body.description)

    @app.get("/api/projects/{name}")
    def project(name: str, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.project(store, actor, name)

    @app.patch("/api/projects/{name}")
    def update_project(name: str, body: ProjectUpdateBody, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.update_project(store, actor, name, body.name, body.description)

    @app.delete("/api/projects/{name}")
    def delete_project(name: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.delete_project(store, actor, name)
        return {"ok": True}

    @app.post("/api/projects/{name}/members")
    def add_member(name: str, body: MemberBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.add_member(store, actor, name, body.username)
        return {"ok": True}

    @app.delete("/api/projects/{name}/members/{username}")
    def remove_member(name: str, username: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.remove_member(store, actor, name, username)
        return {"ok": True}

    @app.post("/api/projects/{name}/records")
    def link(name: str, body: LinkBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.link(store, actor, name, body.record_ids)
        return {"ok": True}

    @app.get("/api/projects/{name}/tables")
    def tables(name: str, actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return service.tables(store, actor, name)

    @app.post("/api/projects/{name}/tables")
    def create_table(name: str, body: dict[str, Any], actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.save_table(store, actor, name, None, body)

    @app.put("/api/projects/{name}/tables/{table_id}")
    def update_table(
        name: str, table_id: str, body: dict[str, Any], actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        return service.save_table(store, actor, name, table_id, body)

    @app.delete("/api/projects/{name}/tables/{table_id}")
    def delete_table(name: str, table_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.delete_table(store, actor, name, table_id)
        return {"ok": True}

    @app.post("/api/projects/{name}/tables/{table_id}/command")
    def table_command(
        name: str, table_id: str, body: CellsBody, request: Request, actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        rendered = service.table_command(
            store, actor, name, table_id, body.run_keys, server_url(request), force=body.force
        )
        return rendered.model_dump(mode="json")

    @app.post("/api/projects/{name}/tables/{table_id}/snapshots")
    def freeze_table(name: str, table_id: str, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.freeze_table(store, actor, name, table_id)

    @app.get("/api/projects/{name}/snapshots")
    def snapshots(name: str, actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return service.snapshots(store, actor, name)

    @app.get("/api/projects/{name}/snapshots/{snapshot_id}")
    def snapshot(name: str, snapshot_id: str, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.snapshot(store, actor, name, snapshot_id)

    @app.patch("/api/projects/{name}/snapshots/{snapshot_id}")
    def rename_snapshot(
        name: str, snapshot_id: str, body: TitleBody, actor: Actor = Depends(current)
    ) -> dict[str, bool]:
        service.rename_snapshot(store, actor, name, snapshot_id, body.title)
        return {"ok": True}

    @app.delete("/api/projects/{name}/snapshots/{snapshot_id}")
    def delete_snapshot(name: str, snapshot_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.delete_snapshot(store, actor, name, snapshot_id)
        return {"ok": True}

    @app.post("/api/projects/{name}/claims")
    def claim(name: str, body: CellsBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.claim(store, actor, name, body.run_keys)
        return {"ok": True}

    @app.delete("/api/projects/{name}/claims/{run_key}")
    def release(name: str, run_key: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.release(store, actor, name, run_key)
        return {"ok": True}

    @app.delete("/api/projects/{name}/records/{record_id}")
    def unlink(name: str, record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.unlink(store, actor, name, record_id)
        return {"ok": True}

    @app.patch("/api/records/{record_id}/annotation")
    def annotate(record_id: str, body: AnnotationBody, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.annotate(store, actor, record_id, body.tags, body.note, body.status)
        return {"ok": True}

    @app.post("/api/records/{record_id}/publish")
    def publish(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.publish(store, actor, record_id)
        return {"ok": True}

    @app.post("/api/records/{record_id}/retract")
    def retract(record_id: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        service.retract(store, actor, record_id)
        return {"ok": True}

    # models -------------------------------------------------------------------------------------------------

    @app.get("/api/models")
    def models(_actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        catalog = service.catalog_for(store, None)
        registered = {entry.code: by for entry, by in store.models()}
        return [
            {"model": m.model_dump(mode="json"), "registered_by": registered.get(code, "builtin")}
            for code, m in catalog.models.items()
        ]

    @app.get("/api/models/lookup")
    def lookup_model(
        repo: str, revision: str | None = None, _actor: Actor = Depends(current)
    ) -> dict[str, Any]:
        return service.model_from_hub(repo, revision or None).model_dump(mode="json")

    @app.post("/api/models")
    def add_model(entry: ModelEntry, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.register_model(store, actor, entry).model_dump(mode="json")

    # methods ------------------------------------------------------------------------------------------------

    @app.get("/api/methods")
    def methods(actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return [
            {"manifest": m.model_dump(mode="json"), "owner": owner, "visibility": vis}
            for m, owner, vis in store.manifests(actor.username)
        ]

    @app.post("/api/methods")
    def register(manifest: MethodManifest, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.register_manifest(store, actor, manifest).model_dump(mode="json")

    @app.post("/api/methods/{name}/publish")
    def publish_method(name: str, actor: Actor = Depends(current)) -> dict[str, bool]:
        if not store.set_manifest_visibility(name, actor.username, Visibility.PUBLIC):
            raise HTTPException(404, "no such method owned by you")
        return {"ok": True}

    # task catalogs ------------------------------------------------------------------------------------------

    @app.get("/api/task-catalogs")
    def task_catalogs(actor: Actor = Depends(current)) -> list[dict[str, Any]]:
        return store.task_listing_summaries()

    @app.post("/api/task-catalogs")
    def register_task_catalog(listing: TaskListing, actor: Actor = Depends(current)) -> dict[str, Any]:
        return service.register_task_listing(store, actor, listing)

    # web UI -------------------------------------------------------------------------------------------------

    if web_dist is not None:
        if not (web_dist / "index.html").exists():
            raise FileNotFoundError(f"{web_dist} has no index.html; build the web UI first (npm run build)")
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = web_dist / path
            if path and candidate.is_file() and candidate.resolve().is_relative_to(web_dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    app.state.store = store
    return app


def init_admin(data_dir: Path, username: str, password: str) -> None:
    """Create the first admin. Raises RuntimeError if the server already has users."""
    store = SqliteStore(data_dir)
    if store.count_users() > 0:
        raise RuntimeError("this server already has users; add more from the Admin page or `effeval admin`")
    store.create_user(username, hash_password(password), "admin")
