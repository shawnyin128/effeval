"""SQLite store for the lab server: users, tokens, records (append-only), annotations, comparisons, manifests.

Per-sample artifacts are gzip JSONL files under `<data_dir>/artifacts/`; records reference them as
`server:artifacts/<name>`. Only server-held artifacts count as available for reuse and comparisons.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from effeval.core.canonical import sha256_hex
from effeval.core.catalog import ModelEntry
from effeval.core.listing import TaskListing
from effeval.core.manifest import MethodManifest
from effeval.core.types import Artifact, Record, Sample, Visibility, utc_now

SERVER_ARTIFACT_PREFIX = "server:artifacts/"
BUSY_TIMEOUT_S = 10.0
# Kinds of data whose changes are pushed to open pages (see SqliteStore.change_versions).
TOPICS = ("records", "projects", "runs", "catalog", "users")
# Columns added after release, created on databases that predate them.
ADDED_COLUMNS = (("tokens", "expires_at"), ("tokens", "last_used"), ("table_snapshots", "title"))

_TRASHED = "record_id IN (SELECT record_id FROM record_states WHERE state = 'trashed')"
_IN_MY_PROJECTS = (
    "record_id IN (SELECT pr.record_id FROM project_records pr"
    " JOIN project_members pm ON pm.project = pr.project WHERE pm.username = ?)"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'member')),
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    locale TEXT NOT NULL DEFAULT 'en' CHECK (locale IN ('en', 'zh')));
CREATE TABLE IF NOT EXISTS tokens (
    token_id TEXT PRIMARY KEY,
    username TEXT NOT NULL REFERENCES users(username),
    token_hash TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    can_publish INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
    last_used TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    session_hash TEXT PRIMARY KEY,
    username TEXT NOT NULL REFERENCES users(username),
    expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    run_key TEXT NOT NULL,
    reference_key TEXT,
    role TEXT NOT NULL,
    owner TEXT,
    visibility TEXT NOT NULL,
    reusable INTEGER NOT NULL,
    probe INTEGER NOT NULL,
    gpu TEXT,
    model TEXT,
    method_name TEXT,
    task_name TEXT,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS records_run_key ON records(run_key);
CREATE INDEX IF NOT EXISTS records_reference_key ON records(reference_key);
CREATE TABLE IF NOT EXISTS metrics (
    record_id TEXT NOT NULL REFERENCES records(record_id),
    name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    value_real REAL,
    PRIMARY KEY (record_id, name));
CREATE TABLE IF NOT EXISTS annotations (
    record_id TEXT PRIMARY KEY REFERENCES records(record_id),
    tags TEXT NOT NULL,
    note TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS comparisons (
    record_id TEXT NOT NULL REFERENCES records(record_id),
    ref_record_id TEXT NOT NULL REFERENCES records(record_id),
    name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    PRIMARY KEY (record_id, name));
CREATE TABLE IF NOT EXISTS manifests (
    name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    visibility TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS task_catalogs (
    evaluator TEXT PRIMARY KEY,
    library_version TEXT,
    sha256 TEXT NOT NULL,
    uploaded_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS models (
    code TEXT PRIMARY KEY,
    registered_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    owner TEXT,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS record_states (
    record_id TEXT PRIMARY KEY REFERENCES records(record_id),
    state TEXT NOT NULL CHECK (state IN ('redo', 'trashed')),
    changed_by TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    superseded_by TEXT);
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES users(username),
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS project_members (
    project TEXT NOT NULL REFERENCES projects(name),
    username TEXT NOT NULL REFERENCES users(username),
    added_at TEXT NOT NULL,
    PRIMARY KEY (project, username));
CREATE TABLE IF NOT EXISTS project_records (
    project TEXT NOT NULL REFERENCES projects(name),
    record_id TEXT NOT NULL REFERENCES records(record_id),
    added_by TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (project, record_id));
CREATE TABLE IF NOT EXISTS project_tables (
    table_id TEXT PRIMARY KEY,
    project TEXT NOT NULL REFERENCES projects(name),
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS table_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project TEXT NOT NULL REFERENCES projects(name),
    table_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL,
    title TEXT);
CREATE TABLE IF NOT EXISTS changes (
    topic TEXT PRIMARY KEY,
    version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS claims (
    project TEXT NOT NULL REFERENCES projects(name),
    run_key TEXT NOT NULL,
    claimed_by TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    PRIMARY KEY (project, run_key));
CREATE TABLE IF NOT EXISTS derived_metrics (
    record_id TEXT NOT NULL REFERENCES records(record_id),
    name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (record_id, name));
CREATE TABLE IF NOT EXISTS plan_titles (
    plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id),
    title TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS storage (
    owner TEXT PRIMARY KEY,
    path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS heartbeats (
    plan_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    at TEXT NOT NULL,
    PRIMARY KEY (plan_id, job_id));
"""


def _real(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


class SqliteStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.artifact_dir = self.data_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.data_dir / "effeval.sqlite3"
        self._write_lock = threading.Lock()
        self._writer = self._connect()
        self._writer.execute("PRAGMA journal_mode=WAL")
        self._writer.executescript(SCHEMA)
        with self._write() as db:
            for table, column in ADDED_COLUMNS:
                if column not in {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self._readers = threading.local()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transactions are opened explicitly by _read and _write.
        db = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
        )
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def _write(self, *topics: str) -> Iterator[sqlite3.Connection]:
        """One write transaction, holding SQLite's write lock from its start; commits, or rolls back on error.

        Each of `topics` (a kind of data users see, one of TOPICS) has its change counter bumped in the same
        transaction. Raises sqlite3.OperationalError ("database is locked") if another process keeps the write
        lock longer than BUSY_TIMEOUT_S.
        """
        unknown = set(topics) - set(TOPICS)
        if unknown:
            raise ValueError(f"unknown change topics {sorted(unknown)}")
        with self._write_lock:
            self._writer.execute("BEGIN IMMEDIATE")
            try:
                self._writer.executemany(
                    "INSERT INTO changes VALUES (?, 1)"
                    " ON CONFLICT(topic) DO UPDATE SET version = version + 1",
                    [(t,) for t in topics],
                )
                yield self._writer
            except BaseException:
                if self._writer.in_transaction:
                    self._writer.execute("ROLLBACK")
                raise
            self._writer.execute("COMMIT")

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """One read transaction (a consistent snapshot) on this thread's own connection, alongside writes."""
        db = getattr(self._readers, "db", None)
        if db is None:
            db = self._readers.db = self._connect()
        db.execute("BEGIN")
        try:
            yield db
        finally:
            if db.in_transaction:
                db.execute("ROLLBACK")

    def change_versions(self) -> dict[str, int]:
        """Change counter of every topic that has changed at least once."""
        with self._read() as db:
            return {r["topic"]: r["version"] for r in db.execute("SELECT topic, version FROM changes")}

    # users, tokens and sessions -----------------------------------------------------------------------------

    def create_user(self, username: str, password_hash: str, role: str) -> None:
        with self._write("users") as db:
            db.execute(
                "INSERT INTO users (username, password_hash, role, disabled, created_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (username, password_hash, role, utc_now()),
            )

    def user(self, username: str) -> sqlite3.Row | None:
        with self._read() as db:
            return db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def users(self) -> list[dict[str, Any]]:
        with self._read() as db:
            rows = db.execute(
                "SELECT username, role, disabled, created_at FROM users ORDER BY username"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_user(self, username: str, **fields: Any) -> None:
        allowed = {"password_hash", "role", "disabled", "locale"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"can only update {sorted(allowed)}")
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._write("users") as db:
            db.execute(f"UPDATE users SET {assignments} WHERE username = ?", (*fields.values(), username))

    def count_users(self) -> int:
        with self._read() as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # (table, column) pairs naming the user as author or owner of lab data.
    USER_DATA = (
        ("records", "owner"),
        ("manifests", "owner"),
        ("plans", "owner"),
        ("projects", "created_by"),
        ("project_records", "added_by"),
        ("project_tables", "created_by"),
        ("table_snapshots", "created_by"),
        ("annotations", "updated_by"),
        ("record_states", "changed_by"),
        ("claims", "claimed_by"),
        ("models", "registered_by"),
        ("task_catalogs", "uploaded_by"),
    )
    # (table, column) pairs that only tie the user to logins, project memberships or a storage setting.
    USER_LINKS = (
        ("tokens", "username"),
        ("sessions", "username"),
        ("project_members", "username"),
        ("storage", "owner"),
    )

    def user_data(self, username: str) -> dict[str, int]:
        """Row counts per USER_DATA table attributed to `username`, listing only tables with rows."""
        with self._read() as db:
            counts = {
                table: db.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (username,)).fetchone()[
                    0
                ]
                for table, column in self.USER_DATA
            }
        return {table: n for table, n in counts.items() if n}

    def delete_user(self, username: str) -> None:
        """Delete a user and their USER_LINKS rows. Raises ValueError if any USER_DATA row names them."""
        with self._write("users", "projects") as db:
            for table, column in self.USER_DATA:
                if db.execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (username,)).fetchone():
                    raise ValueError(f"{username} still has {table}")
            for table, column in self.USER_LINKS:
                db.execute(f"DELETE FROM {table} WHERE {column} = ?", (username,))
            db.execute("DELETE FROM users WHERE username = ?", (username,))

    def rename_user(self, old: str, new: str) -> None:
        """Rename a user in every USER_DATA and USER_LINKS column and in record and plan bodies.

        Raises ValueError if `old` does not exist or `new` is taken.
        """
        with self._write("users", "records", "projects", "runs", "catalog") as db:
            if db.execute("SELECT 1 FROM users WHERE username = ?", (new,)).fetchone():
                raise ValueError(f"user {new} exists")
            # Parent and child rows change in separate statements; check foreign keys at commit.
            db.execute("PRAGMA defer_foreign_keys = ON")
            if db.execute("UPDATE users SET username = ? WHERE username = ?", (new, old)).rowcount == 0:
                raise ValueError(f"no user {old}")
            for table in ("records", "plans"):
                db.execute(
                    f"UPDATE {table} SET body = json_set(body, '$.owner', ?) WHERE owner = ?", (new, old)
                )
            for table, column in (*self.USER_DATA, *self.USER_LINKS):
                db.execute(f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (new, old))

    def create_token(self, username: str, label: str, can_publish: bool, expires_at: str) -> tuple[str, str]:
        """Issue a login token for one machine (`label`); the user's previous token for it stops working.

        Returns (token_id, secret). Only the hash of the secret is stored.
        """
        secret = "eft_" + secrets.token_urlsafe(32)
        token_id = "tok_" + uuid.uuid4().hex[:12]
        with self._write() as db:
            db.execute(
                "UPDATE tokens SET revoked = 1 WHERE username = ? AND label = ? AND revoked = 0",
                (username, label),
            )
            db.execute(
                "INSERT INTO tokens (token_id, username, token_hash, label, can_publish, created_at, revoked,"
                " expires_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (token_id, username, sha256_hex(secret), label, int(can_publish), utc_now(), expires_at),
            )
        return token_id, secret

    def token_owner(self, secret: str, now: str) -> sqlite3.Row | None:
        """The unrevoked token row for `secret` with its user's role, recording the use.

        Expired tokens are returned too (callers check `expires_at`); revoked and disabled users' are not.
        """
        with self._write() as db:
            row = db.execute(
                "SELECT t.token_id, t.username, t.can_publish, t.expires_at, u.role FROM tokens t "
                "JOIN users u USING (username) WHERE t.token_hash = ? AND t.revoked = 0 AND u.disabled = 0",
                (sha256_hex(secret),),
            ).fetchone()
            if row is not None:
                db.execute("UPDATE tokens SET last_used = ? WHERE token_id = ?", (now, row["token_id"]))
        return row

    def active_tokens(self) -> list[dict[str, Any]]:
        """Every user's unrevoked login tokens, newest first."""
        with self._read() as db:
            rows = db.execute(
                "SELECT token_id, username, label, can_publish, created_at, expires_at, last_used"
                " FROM tokens WHERE revoked = 0 ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_token(self, token_id: str) -> bool:
        with self._write() as db:
            cur = db.execute("UPDATE tokens SET revoked = 1 WHERE token_id = ? AND revoked = 0", (token_id,))
        return cur.rowcount == 1

    def create_session(self, username: str, expires_at: str) -> str:
        secret = secrets.token_urlsafe(32)
        with self._write() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (sha256_hex(secret), username, expires_at))
        return secret

    def session_owner(self, secret: str, now: str) -> sqlite3.Row | None:
        with self._read() as db:
            return db.execute(
                "SELECT s.username, u.role FROM sessions s JOIN users u USING (username) "
                "WHERE s.session_hash = ? AND s.expires_at > ? AND u.disabled = 0",
                (sha256_hex(secret), now),
            ).fetchone()

    def delete_session(self, secret: str) -> None:
        with self._write() as db:
            db.execute("DELETE FROM sessions WHERE session_hash = ?", (sha256_hex(secret),))

    # artifacts ----------------------------------------------------------------------------------------------

    # Per-sample files live in the lab's shared directory (published records) or their owner's personal one
    # (everything else). The URI names the file only; it is found by searching the configured directories.

    SHARED = ""

    def storage_dir(self, owner: str) -> Path:
        """Directory of `owner`'s files (shared ones when `owner` is SHARED); defaults to <data>/artifacts."""
        with self._read() as db:
            row = db.execute("SELECT path FROM storage WHERE owner = ?", (owner,)).fetchone()
        return Path(row["path"]) if row else self.artifact_dir

    def _artifact_dirs(self) -> list[Path]:
        with self._read() as db:
            paths = [Path(r["path"]) for r in db.execute("SELECT path FROM storage").fetchall()]
        return list(dict.fromkeys([*paths, self.artifact_dir]))

    def artifact_path(self, uri: str) -> Path | None:
        """Where the server holds artifact `uri`; None for non-server artifacts or missing files."""
        if not uri.startswith(SERVER_ARTIFACT_PREFIX):
            return None
        name = uri[len(SERVER_ARTIFACT_PREFIX) :]
        if "/" in name or name.startswith("."):
            return None
        return next((d / name for d in self._artifact_dirs() if (d / name).exists()), None)

    def save_artifact(self, payload: bytes, owner: str) -> str:
        """Store a gzip JSONL artifact in `owner`'s directory once it parses as samples; returns its URI."""
        samples = [Sample.model_validate_json(line) for line in gzip.decompress(payload).splitlines() if line]
        if not samples:
            raise ValueError("artifact contains no samples")
        name = "art_" + hashlib.sha256(payload).hexdigest()[:32] + ".jsonl.gz"
        uri = SERVER_ARTIFACT_PREFIX + name
        if self.artifact_path(uri) is None:
            target = self.storage_dir(owner)
            target.mkdir(parents=True, exist_ok=True)
            partial = target / f".{name}.{uuid.uuid4().hex}.part"  # hidden from artifact_path until renamed
            partial.write_bytes(payload)
            os.replace(partial, target / name)
        return uri

    def _category_uris(self, owner: str) -> list[str]:
        """Artifact URIs of published records (owner SHARED) or of `owner`'s unpublished records."""
        if owner == self.SHARED:
            sql, params = "SELECT body FROM records WHERE visibility IN ('public', 'retracted')", ()
        else:
            sql, params = "SELECT body FROM records WHERE owner = ? AND visibility = 'private'", (owner,)
        with self._read() as db:
            rows = db.execute(sql, params).fetchall()
        uris = (Record.model_validate_json(r["body"]).artifacts.get("per_sample", "") for r in rows)
        return sorted({u for u in uris if u.startswith(SERVER_ARTIFACT_PREFIX)})

    def storage_usage(self, owner: str) -> dict[str, Any]:
        """{path, files, bytes} of the shared (owner SHARED) or a user's storage."""
        paths = [p for p in map(self.artifact_path, self._category_uris(owner)) if p is not None]
        return {
            "path": str(self.storage_dir(owner)),
            "files": len(paths),
            "bytes": sum(p.stat().st_size for p in paths),
        }

    def set_storage(self, owner: str, path: str) -> int:
        """Point shared (owner SHARED) or a user's storage at `path` and move its existing files there.

        Returns how many files moved. ValueError if `path` is not absolute or the server cannot write there.
        """
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f"storage path must be absolute, got {path!r}")
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".effeval-write-test"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise ValueError(f"the server cannot write to {path}: {exc}") from exc
        moved = 0
        for uri in self._category_uris(owner):
            current = self.artifact_path(uri)
            if current is not None and current.parent.resolve() != target.resolve():
                shutil.move(str(current), target / current.name)
                moved += 1
        with self._write() as db:
            db.execute("INSERT OR REPLACE INTO storage VALUES (?, ?)", (owner, str(target)))
        return moved

    def load_samples(self, uri: str) -> list[Sample] | None:
        path = self.artifact_path(uri)
        if path is None:
            return None
        return [
            Sample.model_validate_json(line)
            for line in gzip.decompress(path.read_bytes()).splitlines()
            if line
        ]

    @staticmethod
    def available_artifacts(record: Record) -> frozenset[Artifact]:
        """Artifacts usable on the server: produced by the evaluator and held in server storage."""
        if not record.artifacts.get("per_sample", "").startswith(SERVER_ARTIFACT_PREFIX):
            return frozenset()
        return frozenset(record.produces)

    # records ------------------------------------------------------------------------------------------------

    def insert_record(self, record: Record) -> bool:
        """Append a record. Returns False (and changes nothing) if its job_id is already stored."""
        env_gpu = record.env.gpu if record.env else None
        spec = record.spec
        with self._write("records") as db:
            if db.execute("SELECT 1 FROM records WHERE job_id = ?", (record.job_id,)).fetchone():
                return False
            db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    record.job_id,
                    record.run_key,
                    record.reference_key,
                    record.role,
                    record.owner,
                    record.visibility.value,
                    int(record.reusable),
                    int(record.probe),
                    env_gpu,
                    spec.get("model", {}).get("name"),
                    spec.get("method", {}).get("name"),
                    spec.get("task", {}).get("name"),
                    record.created_at,
                    record.model_dump_json(),
                ),
            )
            db.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                [(record.record_id, k, v, _real(v)) for k, v in record.metrics.items()],
            )
        return True

    def record(self, record_id: str) -> Record | None:
        with self._read() as db:
            row = db.execute(
                "SELECT body, visibility FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
        return None if row is None else self._hydrate(row)

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> Record:
        record = Record.model_validate_json(row["body"])
        return record.model_copy(update={"visibility": Visibility(row["visibility"])})

    def set_visibility(self, record_id: str, visibility: Visibility) -> None:
        """Change a record's visibility; publishing moves its per-sample file into the shared directory."""
        with self._write("records") as db:
            db.execute("UPDATE records SET visibility = ? WHERE record_id = ?", (visibility.value, record_id))
            row = db.execute("SELECT body FROM records WHERE record_id = ?", (record_id,)).fetchone()
        if visibility == Visibility.PUBLIC:
            current = self.artifact_path(
                Record.model_validate_json(row["body"]).artifacts.get("per_sample", "")
            )
            shared = self.storage_dir(self.SHARED)
            if current is not None and current.parent.resolve() != shared.resolve():
                shared.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), shared / current.name)

    def visible_records(self, viewer: str | None, *, where: str = "", params: tuple = ()) -> list[Record]:
        """Records the viewer may see and that are not in the trash.

        Visible means: the viewer's own (any visibility), public and retracted ones, and records linked to a
        project the viewer is a member of.
        """
        clause = (
            f"(owner = ? OR visibility IN ('public', 'retracted') OR {_IN_MY_PROJECTS}) AND NOT {_TRASHED}"
        )
        sql = (
            f"SELECT body, visibility FROM records WHERE {clause}{' AND ' + where if where else ''} "
            "ORDER BY created_at DESC"
        )
        with self._read() as db:
            rows = db.execute(sql, (viewer, viewer, *params)).fetchall()
        return [self._hydrate(r) for r in rows]

    def can_view(self, viewer: str | None, record_id: str) -> bool:
        """Whether `record_id` is visible to `viewer` by the rule of `visible_records`, trash aside."""
        with self._read() as db:
            row = db.execute(
                "SELECT 1 FROM records WHERE record_id = ?"
                f" AND (owner = ? OR visibility IN ('public', 'retracted')"
                f" OR {_IN_MY_PROJECTS})",
                (record_id, viewer, viewer),
            ).fetchone()
        return row is not None

    def set_artifact(self, record_id: str, uri: str) -> None:
        """Point a stored record's per-sample artifact at `uri`; the measured values stay untouched."""
        with self._write("records") as db:
            row = db.execute("SELECT body FROM records WHERE record_id = ?", (record_id,)).fetchone()
            record = Record.model_validate_json(row["body"])
            updated = record.model_copy(update={"artifacts": {**record.artifacts, "per_sample": uri}})
            db.execute(
                "UPDATE records SET body = ? WHERE record_id = ?", (updated.model_dump_json(), record_id)
            )

    def artifact_bytes(self, uri: str) -> bytes:
        """Raw bytes of a server-held artifact. ValueError for URIs that are not server-held."""
        path = self.artifact_path(uri)
        if path is None:
            raise ValueError(f"{uri!r} is not a server artifact held here")
        return path.read_bytes()

    def record_id_for_job(self, job_id: str) -> str | None:
        with self._read() as db:
            row = db.execute("SELECT record_id FROM records WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else row["record_id"]

    def all_records(self) -> list[Record]:
        """Every stored record, trashed ones included."""
        with self._read() as db:
            rows = db.execute("SELECT body, visibility FROM records").fetchall()
        return [self._hydrate(r) for r in rows]

    def derived(self, record_id: str) -> dict[str, dict[str, str]]:
        """Metrics the server computed for a record after storing it: name -> {value, computed_at}."""
        with self._read() as db:
            rows = db.execute(
                "SELECT name, value_text, computed_at FROM derived_metrics WHERE record_id = ?", (record_id,)
            ).fetchall()
        return {r["name"]: {"value": r["value_text"], "computed_at": r["computed_at"]} for r in rows}

    def put_derived(self, record_id: str, values: dict[str, str]) -> None:
        with self._write("records") as db:
            db.executemany(
                "INSERT OR REPLACE INTO derived_metrics VALUES (?, ?, ?, ?)",
                [(record_id, name, value, utc_now()) for name, value in values.items()],
            )

    # record lifecycle: redo marks and trash -----------------------------------------------------------------

    def state(self, record_id: str) -> dict[str, Any] | None:
        """The record's redo/trash state row, or None for an active record."""
        with self._read() as db:
            row = db.execute("SELECT * FROM record_states WHERE record_id = ?", (record_id,)).fetchone()
        return None if row is None else dict(row)

    def set_state(self, record_id: str, state: str, by: str, superseded_by: str | None = None) -> None:
        with self._write("records") as db:
            db.execute(
                "INSERT OR REPLACE INTO record_states VALUES (?, ?, ?, ?, ?)",
                (record_id, state, by, utc_now(), superseded_by),
            )

    def clear_state(self, record_id: str) -> None:
        with self._write("records") as db:
            db.execute("DELETE FROM record_states WHERE record_id = ?", (record_id,))

    def redo_pending(self, run_key: str, by: str) -> list[Record]:
        """Records with this RunKey marked for redo that `by` owns or marked."""
        with self._read() as db:
            rows = db.execute(
                "SELECT r.body, r.visibility FROM records r JOIN record_states s ON s.record_id = r.record_id"
                " WHERE s.state = 'redo' AND r.run_key = ? AND (r.owner = ? OR s.changed_by = ?)",
                (run_key, by, by),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def trashed(self, owner: str | None) -> list[tuple[Record, dict[str, Any]]]:
        """Trashed records with their state row; only `owner`'s unless `owner` is None (every record)."""
        sql = (
            "SELECT r.body, r.visibility, s.* FROM records r"
            " JOIN record_states s ON s.record_id = r.record_id"
            " WHERE s.state = 'trashed'"
        )
        params: tuple = ()
        if owner is not None:
            sql += " AND r.owner = ?"
            params = (owner,)
        with self._read() as db:
            rows = db.execute(sql + " ORDER BY s.changed_at DESC", params).fetchall()
        keys = ("record_id", "state", "changed_by", "changed_at", "superseded_by")
        return [(self._hydrate(r), {k: r[k] for k in keys}) for r in rows]

    def comparing_against(self, record_id: str) -> list[Record]:
        """Untrashed records whose stored comparisons use `record_id` as their reference."""
        with self._read() as db:
            rows = db.execute(
                "SELECT body, visibility FROM records WHERE record_id IN"
                f" (SELECT record_id FROM comparisons WHERE ref_record_id = ?) AND NOT {_TRASHED}",
                (record_id,),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def purge(self, record_id: str) -> None:
        """Delete a record and everything attached to it.

        Its artifact file is removed too once no other record refers to it.
        """
        with self._write("records") as db:
            row = db.execute("SELECT body FROM records WHERE record_id = ?", (record_id,)).fetchone()
            if row is None:
                return
            uri = Record.model_validate_json(row["body"]).artifacts.get("per_sample", "")
            db.execute(
                "DELETE FROM comparisons WHERE record_id = ? OR ref_record_id = ?", (record_id, record_id)
            )
            for table in ("metrics", "derived_metrics", "annotations", "project_records", "record_states"):
                db.execute(f"DELETE FROM {table} WHERE record_id = ?", (record_id,))
            db.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            shared = (
                uri
                and db.execute(
                    "SELECT 1 FROM records WHERE instr(body, ?) > 0 LIMIT 1", (json.dumps(uri),)
                ).fetchone()
            )
        path = self.artifact_path(uri)
        if path is not None and not shared:
            path.unlink(missing_ok=True)

    # projects -----------------------------------------------------------------------------------------------

    def create_project(self, name: str, description: str, by: str) -> None:
        """Create a project with `by` as its first member. ValueError if the name is taken."""
        with self._write("projects") as db:
            if db.execute("SELECT 1 FROM projects WHERE name = ?", (name,)).fetchone():
                raise ValueError(f"project {name!r} already exists")
            now = utc_now()
            db.execute("INSERT INTO projects VALUES (?, ?, ?, ?)", (name, description, by, now))
            db.execute("INSERT INTO project_members VALUES (?, ?, ?)", (name, by, now))

    def project(self, name: str) -> dict[str, Any] | None:
        """Project row with its member usernames, or None."""
        with self._read() as db:
            row = db.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
            if row is None:
                return None
            members = db.execute(
                "SELECT username FROM project_members WHERE project = ? ORDER BY added_at", (name,)
            ).fetchall()
            count = db.execute("SELECT COUNT(*) FROM project_records WHERE project = ?", (name,)).fetchone()[
                0
            ]
        return {**dict(row), "members": [m["username"] for m in members], "records": count}

    def projects(self, member: str | None) -> list[dict[str, Any]]:
        """Projects `member` belongs to, or every project when `member` is None."""
        with self._read() as db:
            if member is None:
                names = db.execute("SELECT name FROM projects ORDER BY name").fetchall()
            else:
                names = db.execute(
                    "SELECT project AS name FROM project_members WHERE username = ? ORDER BY project",
                    (member,),
                ).fetchall()
        return [p for p in (self.project(n["name"]) for n in names) if p is not None]

    def add_member(self, project: str, username: str) -> None:
        with self._write("projects") as db:
            db.execute(
                "INSERT OR IGNORE INTO project_members VALUES (?, ?, ?)", (project, username, utc_now())
            )

    def remove_member(self, project: str, username: str) -> None:
        with self._write("projects") as db:
            db.execute("DELETE FROM project_members WHERE project = ? AND username = ?", (project, username))

    # Tables whose `project` column refers to projects(name).
    PROJECT_TABLES = ("project_members", "project_records", "project_tables", "table_snapshots", "claims")

    def set_project_description(self, name: str, description: str) -> None:
        with self._write("projects") as db:
            db.execute("UPDATE projects SET description = ? WHERE name = ?", (description, name))

    def rename_project(self, old: str, new: str) -> None:
        """Rename a project in every table that refers to it. ValueError if `old` is missing or `new` is
        taken."""
        with self._write("projects", "records") as db:
            if db.execute("SELECT 1 FROM projects WHERE name = ?", (new,)).fetchone():
                raise ValueError(f"project {new!r} already exists")
            # The project row and the rows pointing at it change in separate statements; check keys at commit.
            db.execute("PRAGMA defer_foreign_keys = ON")
            if db.execute("UPDATE projects SET name = ? WHERE name = ?", (new, old)).rowcount == 0:
                raise ValueError(f"no project {old!r}")
            for table in self.PROJECT_TABLES:
                db.execute(f"UPDATE {table} SET project = ? WHERE project = ?", (new, old))

    def delete_project(self, name: str) -> None:
        """Delete a project with its members, record links, tables, snapshots and claims; the records stay."""
        with self._write("projects", "records") as db:
            for table in self.PROJECT_TABLES:
                db.execute(f"DELETE FROM {table} WHERE project = ?", (name,))
            db.execute("DELETE FROM projects WHERE name = ?", (name,))

    def link_records(self, project: str, record_ids: list[str], by: str) -> None:
        with self._write("projects", "records") as db:
            db.executemany(
                "INSERT OR IGNORE INTO project_records VALUES (?, ?, ?, ?)",
                [(project, rid, by, utc_now()) for rid in record_ids],
            )

    def link_records_from(self, source: str, target: str, by: str) -> None:
        """Link `target` to every project `source` is linked to."""
        with self._write("projects", "records") as db:
            db.execute(
                "INSERT OR IGNORE INTO project_records"
                " SELECT project, ?, ?, ? FROM project_records WHERE record_id = ?",
                (target, by, utc_now(), source),
            )

    def unlink_record(self, project: str, record_id: str) -> None:
        with self._write("projects", "records") as db:
            db.execute(
                "DELETE FROM project_records WHERE project = ? AND record_id = ?", (project, record_id)
            )

    def project_records(self, project: str) -> list[Record]:
        """Every record linked to the project, trashed ones included, newest first."""
        with self._read() as db:
            rows = db.execute(
                "SELECT r.body, r.visibility FROM records r"
                " JOIN project_records pr ON pr.record_id = r.record_id"
                " WHERE pr.project = ? ORDER BY r.created_at DESC",
                (project,),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def projects_of(self, record_id: str) -> list[str]:
        with self._read() as db:
            rows = db.execute(
                "SELECT project FROM project_records WHERE record_id = ? ORDER BY project", (record_id,)
            ).fetchall()
        return [r["project"] for r in rows]

    def find_reusable(
        self, run_key: str, *, viewer: str | None, need: frozenset[Artifact], hardware: str | None
    ) -> Record | None:
        where = (
            "run_key = ? AND visibility != 'retracted' AND probe = 0 AND reusable = 1"
            " AND record_id NOT IN (SELECT record_id FROM record_states)"
        )
        params: tuple = (run_key,)
        if hardware is not None:
            where += " AND gpu = ?"
            params += (hardware,)
        for record in self.visible_records(viewer, where=where, params=params):
            if need <= self.available_artifacts(record):
                return record
        return None

    # annotations and comparisons ----------------------------------------------------------------------------

    def annotation(self, record_id: str) -> dict[str, Any] | None:
        with self._read() as db:
            row = db.execute("SELECT * FROM annotations WHERE record_id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["tags"] = json.loads(out["tags"])
        return out

    def set_annotation(self, record_id: str, tags: list[str], note: str, status: str, by: str) -> None:
        with self._write("records") as db:
            db.execute(
                "INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, json.dumps(tags), note, status, by, utc_now()),
            )

    def comparisons(self, record_id: str) -> dict[str, Any]:
        with self._read() as db:
            rows = db.execute(
                "SELECT ref_record_id, name, value_text FROM comparisons WHERE record_id = ?", (record_id,)
            ).fetchall()
        return {
            "reference": rows[0]["ref_record_id"] if rows else None,
            "values": {r["name"]: r["value_text"] for r in rows},
        }

    def replace_comparisons(self, record_id: str, ref_record_id: str, values: dict[str, str]) -> None:
        with self._write("records") as db:
            db.execute("DELETE FROM comparisons WHERE record_id = ?", (record_id,))
            db.executemany(
                "INSERT INTO comparisons VALUES (?, ?, ?, ?)",
                [(record_id, ref_record_id, k, v) for k, v in values.items()],
            )

    def records_referencing(self, run_key: str) -> list[Record]:
        with self._read() as db:
            rows = db.execute(
                "SELECT body, visibility FROM records WHERE reference_key = ?", (run_key,)
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    # experiment tables, claims, activity --------------------------------------------------------------------

    def put_table(self, project: str, table_id: str | None, body: str, by: str) -> str:
        """Create (table_id None) or replace a project's experiment table; returns its id.

        KeyError if `table_id` does not name a table of `project`.
        """
        now = utc_now()
        with self._write("projects") as db:
            if table_id is None:
                table_id = "tbl_" + uuid.uuid4().hex[:12]
                db.execute(
                    "INSERT INTO project_tables VALUES (?, ?, ?, ?, ?, ?)",
                    (table_id, project, by, now, now, body),
                )
            else:
                cur = db.execute(
                    "UPDATE project_tables SET body = ?, updated_at = ? WHERE table_id = ? AND project = ?",
                    (body, now, table_id, project),
                )
                if cur.rowcount != 1:
                    raise KeyError(table_id)
        return table_id

    def tables(self, project: str) -> list[dict[str, Any]]:
        with self._read() as db:
            rows = db.execute(
                "SELECT * FROM project_tables WHERE project = ? ORDER BY created_at", (project,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_table(self, project: str, table_id: str) -> bool:
        with self._write("projects") as db:
            cur = db.execute(
                "DELETE FROM project_tables WHERE table_id = ? AND project = ?", (table_id, project)
            )
        return cur.rowcount == 1

    def put_snapshot(self, project: str, table_id: str, body: str, by: str) -> str:
        snapshot_id = "snap_" + uuid.uuid4().hex[:12]
        with self._write("projects") as db:
            db.execute(
                "INSERT INTO table_snapshots VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (snapshot_id, project, table_id, by, utc_now(), body),
            )
        return snapshot_id

    def snapshots(self, project: str) -> list[dict[str, Any]]:
        """Snapshots of a project's tables, newest first, without their bodies."""
        with self._read() as db:
            rows = db.execute(
                "SELECT snapshot_id, table_id, created_by, created_at, title FROM table_snapshots"
                " WHERE project = ?"
                " ORDER BY created_at DESC",
                (project,),
            ).fetchall()
        return [dict(r) for r in rows]

    def snapshot(self, project: str, snapshot_id: str) -> dict[str, Any] | None:
        with self._read() as db:
            row = db.execute(
                "SELECT * FROM table_snapshots WHERE project = ? AND snapshot_id = ?", (project, snapshot_id)
            ).fetchone()
        return None if row is None else {**dict(row), "body": json.loads(row["body"])}

    def set_snapshot_title(self, project: str, snapshot_id: str, title: str | None) -> bool:
        """Name a snapshot (None clears the name); False if the project has no such snapshot."""
        with self._write("projects") as db:
            cur = db.execute(
                "UPDATE table_snapshots SET title = ? WHERE project = ? AND snapshot_id = ?",
                (title, project, snapshot_id),
            )
        return cur.rowcount == 1

    def delete_snapshot(self, project: str, snapshot_id: str) -> bool:
        """Delete a snapshot; False if the project has no such snapshot."""
        with self._write("projects") as db:
            cur = db.execute(
                "DELETE FROM table_snapshots WHERE project = ? AND snapshot_id = ?", (project, snapshot_id)
            )
        return cur.rowcount == 1

    def claims(self, run_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Claims on these RunKeys in any project: run_key -> [{project, claimed_by, claimed_at}]."""
        out: dict[str, list[dict[str, Any]]] = {}
        with self._read() as db:
            for key in run_keys:
                rows = db.execute("SELECT * FROM claims WHERE run_key = ?", (key,)).fetchall()
                if rows:
                    out[key] = [dict(r) for r in rows]
        return out

    def claim(self, project: str, run_keys: list[str], by: str) -> None:
        """Claim RunKeys in a project. ValueError naming the holder if one is claimed by someone else."""
        with self._write("projects") as db:
            for key in run_keys:
                row = db.execute(
                    "SELECT claimed_by FROM claims WHERE project = ? AND run_key = ?", (project, key)
                ).fetchone()
                if row is not None and row["claimed_by"] != by:
                    raise ValueError(f"{key} is already claimed by {row['claimed_by']}")
            db.executemany(
                "INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?)",
                [(project, k, by, utc_now()) for k in run_keys],
            )

    def release(self, project: str, run_key: str) -> dict[str, Any] | None:
        """Drop a claim; returns the removed claim row, or None if there was none."""
        with self._write("projects") as db:
            row = db.execute(
                "SELECT * FROM claims WHERE project = ? AND run_key = ?", (project, run_key)
            ).fetchone()
            db.execute("DELETE FROM claims WHERE project = ? AND run_key = ?", (project, run_key))
        return None if row is None else dict(row)

    def activity(self) -> dict[str, dict[str, Any]]:
        """Latest heartbeat per RunKey across all plans: {status, owner, plan_id, job_id, at, message}."""
        with self._read() as db:
            plans = db.execute("SELECT plan_id, owner, body FROM plans").fetchall()
            beats = db.execute("SELECT * FROM heartbeats").fetchall()
        jobs: dict[tuple[str, str], tuple[str, str | None]] = {}
        for plan in plans:
            for job in json.loads(plan["body"]).get("jobs", []):
                jobs[(plan["plan_id"], job["job_id"])] = (job["run_key"], plan["owner"])
        latest: dict[str, dict[str, Any]] = {}
        for beat in beats:
            found = jobs.get((beat["plan_id"], beat["job_id"]))
            if found is None:
                continue
            key, owner = found
            if key not in latest or beat["at"] > latest[key]["at"]:
                latest[key] = {**dict(beat), "owner": owner}
        return latest

    # manifests, plans, heartbeats ---------------------------------------------------------------------------

    def put_manifest(self, manifest: MethodManifest, owner: str) -> None:
        with self._write("catalog") as db:
            existing = db.execute("SELECT owner FROM manifests WHERE name = ?", (manifest.name,)).fetchone()
            if existing is not None and existing["owner"] != owner:
                raise PermissionError(f"method name {manifest.name!r} is owned by {existing['owner']}")
            visibility = (
                "private"
                if existing is None
                else db.execute(
                    "SELECT visibility FROM manifests WHERE name = ?", (manifest.name,)
                ).fetchone()["visibility"]
            )
            db.execute(
                "INSERT OR REPLACE INTO manifests VALUES (?, ?, ?, ?, ?)",
                (manifest.name, owner, visibility, utc_now(), manifest.model_dump_json()),
            )

    def manifests(self, viewer: str | None) -> list[tuple[MethodManifest, str, str]]:
        """(manifest, owner, visibility) for manifests owned by the viewer or public."""
        with self._read() as db:
            rows = db.execute(
                "SELECT body, owner, visibility FROM manifests WHERE owner = ? OR visibility = 'public'"
                " ORDER BY name",
                (viewer,),
            ).fetchall()
        return [(MethodManifest.model_validate_json(r["body"]), r["owner"], r["visibility"]) for r in rows]

    def set_manifest_visibility(self, name: str, owner: str, visibility: Visibility) -> bool:
        with self._write("catalog") as db:
            cur = db.execute(
                "UPDATE manifests SET visibility = ? WHERE name = ? AND owner = ?",
                (visibility.value, name, owner),
            )
        return cur.rowcount == 1

    def put_task_listing(self, listing: TaskListing, by: str) -> None:
        """Store `listing` as the lab-wide task catalog of its evaluator, replacing the previous one."""
        with self._write("catalog") as db:
            db.execute(
                "INSERT OR REPLACE INTO task_catalogs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    listing.evaluator.name,
                    listing.evaluator.library_version,
                    listing.digest(),
                    by,
                    utc_now(),
                    listing.model_dump_json(),
                ),
            )

    def task_listing_digests(self) -> dict[str, str]:
        """Evaluator name -> digest of its stored listing."""
        with self._read() as db:
            return {
                r["evaluator"]: r["sha256"] for r in db.execute("SELECT evaluator, sha256 FROM task_catalogs")
            }

    def task_listing(self, evaluator: str) -> TaskListing:
        """The stored listing of `evaluator`. Raises KeyError if none was uploaded."""
        with self._read() as db:
            row = db.execute("SELECT body FROM task_catalogs WHERE evaluator = ?", (evaluator,)).fetchone()
        if row is None:
            raise KeyError(evaluator)
        return TaskListing.model_validate_json(row["body"])

    def task_listing_summaries(self) -> list[dict[str, Any]]:
        """One summary per stored listing: evaluator, library_version, tasks, uploaded_by, created_at."""
        with self._read() as db:
            rows = db.execute(
                "SELECT evaluator, library_version, uploaded_by, created_at, body FROM task_catalogs"
                " ORDER BY evaluator"
            ).fetchall()
        return [
            {
                "evaluator": r["evaluator"],
                "library_version": r["library_version"],
                "tasks": len(json.loads(r["body"])["tasks"]),
                "uploaded_by": r["uploaded_by"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def add_model(self, entry: ModelEntry, by: str) -> bool:
        """Register a model code. False if the identical entry exists; ValueError on a clash."""
        with self._write("catalog") as db:
            row = db.execute("SELECT body FROM models WHERE code = ?", (entry.code,)).fetchone()
            if row is not None:
                if ModelEntry.model_validate_json(row["body"]) == entry:
                    return False
                raise ValueError(f"model code {entry.code!r} is already registered for another checkpoint")
            db.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?)", (entry.code, by, utc_now(), entry.model_dump_json())
            )
        return True

    def models(self) -> list[tuple[ModelEntry, str]]:
        """Every registered model with who registered it; models are lab-wide."""
        with self._read() as db:
            rows = db.execute("SELECT body, registered_by FROM models ORDER BY code").fetchall()
        return [(ModelEntry.model_validate_json(r["body"]), r["registered_by"]) for r in rows]

    def save_plan(self, plan_id: str, owner: str | None, body: str) -> None:
        with self._write("runs") as db:
            db.execute("INSERT OR REPLACE INTO plans VALUES (?, ?, ?, ?)", (plan_id, owner, utc_now(), body))

    def plan(self, plan_id: str) -> tuple[str | None, str] | None:
        """(owner, plan JSON) of a stored plan, or None."""
        with self._read() as db:
            row = db.execute("SELECT owner, body FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
        return None if row is None else (row["owner"], row["body"])

    def plans(self, owner: str) -> list[dict[str, Any]]:
        with self._read() as db:
            plans = db.execute(
                "SELECT plan_id, created_at, body FROM plans WHERE owner = ? ORDER BY created_at DESC",
                (owner,),
            ).fetchall()
            beats = db.execute(
                "SELECT * FROM heartbeats WHERE plan_id IN (SELECT plan_id FROM plans WHERE owner = ?)",
                (owner,),
            ).fetchall()
            titles = dict(db.execute("SELECT plan_id, title FROM plan_titles").fetchall())
        by_plan: dict[str, dict[str, Any]] = {}
        for beat in beats:
            by_plan.setdefault(beat["plan_id"], {})[beat["job_id"]] = dict(beat)
        return [
            {
                "plan_id": p["plan_id"],
                "created_at": p["created_at"],
                "title": titles.get(p["plan_id"]),
                "plan": json.loads(p["body"]),
                "heartbeats": by_plan.get(p["plan_id"], {}),
            }
            for p in plans
        ]

    def set_plan_title(self, plan_id: str, title: str | None) -> None:
        with self._write("runs") as db:
            db.execute("DELETE FROM plan_titles WHERE plan_id = ?", (plan_id,))
            if title:
                db.execute("INSERT INTO plan_titles VALUES (?, ?)", (plan_id, title))

    def delete_plan(self, plan_id: str) -> None:
        """Forget a plan and its progress; records it produced stay."""
        with self._write("runs") as db:
            for table in ("plan_titles", "heartbeats", "plans"):
                db.execute(f"DELETE FROM {table} WHERE plan_id = ?", (plan_id,))

    def heartbeat(self, plan_id: str, job_id: str, status: str, message: str) -> None:
        with self._write("runs") as db:
            db.execute(
                "INSERT OR REPLACE INTO heartbeats VALUES (?, ?, ?, ?, ?)",
                (plan_id, job_id, status, message, utc_now()),
            )
