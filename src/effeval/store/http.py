"""HTTP client for the lab server, used by the CLI and compute nodes. Implements the ReuseLookup protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from effeval.core.bundle import Bundle, dumps
from effeval.core.catalog import ModelEntry
from effeval.core.listing import TaskListing
from effeval.core.manifest import MethodManifest
from effeval.core.types import Artifact, Record


class ClientError(RuntimeError):
    """The server answered with an error, or could not be reached."""


def credentials_file() -> Path:
    """Where `effeval login` keeps the server URL and login token: `$EFFEVAL_HOME/credentials.json`."""
    return Path(os.environ.get("EFFEVAL_HOME", Path.home() / ".config" / "effeval")) / "credentials.json"


def saved_login(server: str | None = None) -> tuple[str | None, str | None]:
    """(server, token) from `server`, else EFFEVAL_SERVER / EFFEVAL_TOKEN, else the saved credentials."""
    path = credentials_file()
    saved = json.loads(path.read_text()) if path.exists() else {}
    return (
        server or os.environ.get("EFFEVAL_SERVER") or saved.get("server"),
        os.environ.get("EFFEVAL_TOKEN") or saved.get("token"),
    )


class HttpClient:
    def __init__(
        self, server: str, token: str | None, timeout: float = 30.0, http: httpx.Client | None = None
    ):
        """`http`, when given, is used as-is (it must already carry the base URL and auth header)."""
        self.server = server.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = http or httpx.Client(base_url=self.server, headers=headers, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ClientError(f"cannot reach {self.server}: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ClientError(f"{method} {path} -> {response.status_code}: {detail}")
        return response.json()

    def health(self, timeout: float) -> dict[str, Any]:
        try:
            response = self._http.get("/api/health", timeout=timeout)
        except httpx.HTTPError as exc:
            raise ClientError(f"cannot reach {self.server}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise ClientError(f"{self.server}/api/health answered {response.status_code}")
        return response.json()

    def password_token(
        self, username: str, password: str, label: str, can_publish: bool = True
    ) -> dict[str, str]:
        return self._call(
            "POST",
            "/api/auth/token",
            json={"username": username, "password": password, "label": label, "can_publish": can_publish},
        )

    def me(self) -> dict[str, Any]:
        return self._call("GET", "/api/me")

    def find_reusable(
        self, run_key: str, *, viewer: str | None, need: frozenset[Artifact], hardware: str | None
    ) -> Record | None:
        params = {"run_key": run_key, "need": ",".join(sorted(a.value for a in need))}
        if hardware is not None:
            params["hardware"] = hardware
        found = self._call("GET", "/api/reuse", params=params)["record"]
        return None if found is None else Record.model_validate(found)

    def render(self, spec: dict[str, Any], shards: int, to: str = "private") -> dict[str, Any]:
        return self._call("POST", "/api/render", json={"spec": spec, "shards": shards, "to": to})

    def describe(self) -> dict[str, Any]:
        return self._call("GET", "/api/describe")

    def validate(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return self._call("POST", "/api/validate", json={"spec": spec})["issues"]

    def runs(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/runs")

    def record(self, record_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/records/{record_id}")

    def projects(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/projects")

    def create_project(self, name: str, description: str) -> dict[str, Any]:
        return self._call("POST", "/api/projects", json={"name": name, "description": description})

    def tables(self, project: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/api/projects/{project}/tables")

    def save_table(self, project: str, table: dict[str, Any], table_id: str | None) -> dict[str, Any]:
        if table_id is None:
            return self._call("POST", f"/api/projects/{project}/tables", json=table)
        return self._call("PUT", f"/api/projects/{project}/tables/{table_id}", json=table)

    def table_command(self, project: str, table_id: str, run_keys: list[str], force: bool) -> dict[str, Any]:
        return self._call(
            "POST",
            f"/api/projects/{project}/tables/{table_id}/command",
            json={"run_keys": run_keys, "force": force},
        )

    def models(self) -> list[ModelEntry]:
        return [ModelEntry.model_validate(m["model"]) for m in self._call("GET", "/api/models")]

    def register_model(self, entry: ModelEntry) -> dict[str, Any]:
        return self._call("POST", "/api/models", json=entry.model_dump(mode="json"))

    def manifests(self) -> list[MethodManifest]:
        return [MethodManifest.model_validate(m["manifest"]) for m in self._call("GET", "/api/methods")]

    def register_manifest(self, manifest: MethodManifest) -> dict[str, Any]:
        return self._call("POST", "/api/methods", json=manifest.model_dump(mode="json"))

    def register_task_listing(self, listing: TaskListing) -> dict[str, Any]:
        return self._call("POST", "/api/task-catalogs", json=listing.model_dump(mode="json"))

    def task_listings(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/task-catalogs")

    def upload_artifact(self, path: Path) -> str:
        with path.open("rb") as handle:
            return self._call(
                "POST", "/api/artifacts", files={"file": (path.name, handle, "application/gzip")}
            )["uri"]

    def push(self, bundle: Bundle, via: str = "cli", to: str = "private") -> dict[str, Any]:
        return self._call(
            "POST",
            "/api/bundles",
            params={"via": via, "to": to},
            content=dumps(bundle),
            headers={"Content-Type": "text/plain"},
        )

    def heartbeat(self, plan_id: str, job_id: str, status: str, message: str = "") -> None:
        self._call(
            "POST",
            "/api/heartbeat",
            json={"plan_id": plan_id, "job_id": job_id, "status": status, "message": message},
        )

    def records(self, scope: str, filters: dict[str, str]) -> list[dict[str, Any]]:
        return self._call("GET", "/api/records", params={"scope": scope, **filters})

    def annotate(self, record_id: str, tags: list[str], note: str, status: str) -> None:
        self._call(
            "PATCH",
            f"/api/records/{record_id}/annotation",
            json={"tags": tags, "note": note, "status": status},
        )

    def publish(self, record_id: str) -> None:
        self._call("POST", f"/api/records/{record_id}/publish")

    def retract(self, record_id: str) -> str:
        """Take a record out of the lab pool; returns "private" or "retracted" (others depend on it)."""
        return self._call("POST", f"/api/records/{record_id}/retract")["unshared"]

    def admin(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._call(method, f"/api/admin{path}", json=body)
