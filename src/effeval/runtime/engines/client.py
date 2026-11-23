"""Client of the engine worker subprocess (stdlib only); the protocol is defined by worker.py."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

WORKER_PATH = Path(__file__).with_name("worker.py")
READY_TIMEOUT_S = 1800.0
REQUEST_TIMEOUT_S = 900.0
EXIT_TIMEOUT_S = 120.0
TERM_GRACE_S = 30.0
REAP_TIMEOUT_S = 10.0
LOG_TAIL_LINES = 50


class EngineError(RuntimeError):
    """The engine worker failed, timed out, or violated the protocol."""


@dataclass(frozen=True)
class EngineReply:
    output_ids: list[int]
    seconds: float
    rounds: int | None
    proposed: int | None
    accepted: int | None
    raw_accepted: int | None = None


def tail(log: Path, n: int = LOG_TAIL_LINES) -> str:
    """Return the last `n` lines of `log` ("" when the file does not exist)."""
    if not log.is_file():
        return ""
    return "\n".join(log.read_text(errors="replace").splitlines()[-n:])


def _public_version(text: str) -> str:
    """Strip the distribution name, the `==` pin and any local `+suffix`: "vllm==0.32.0+cu129" -> "0.32.0"."""
    return text.rsplit("==", 1)[-1].split("+")[0]


class EngineClient:
    """One engine worker subprocess speaking JSON lines over stdin/stdout."""

    ready: dict

    def __init__(self, proc: subprocess.Popen, lines: queue.Queue, log: Path, config: dict):
        self._proc = proc
        self._lines = lines
        self._log = log
        self._config = config
        self._engine = config["engine"]
        self._stop_ids = frozenset(config["stop_ids"])
        self._raw = {"requests": 0, "rounds": 0, "proposed": 0, "accepted": 0, "discarded_accepted": 0}
        self._next_id = 0
        self._closed = False
        self.ready = {}

    @classmethod
    def start(cls, python: str, config: dict, workdir: Path, *, expect_library: str) -> EngineClient:
        """Spawn the worker with `python` and `config`, wait for its ready line and check the library version.

        Raises EngineError if the worker fails, is not ready in READY_TIMEOUT_S, speaks out of protocol, or
        reports a library version other than `expect_library` ("dist==x.y.z").
        """
        if os.name != "posix":
            raise EngineError("engine workers run on POSIX hosts only")
        workdir.mkdir(parents=True, exist_ok=True)
        config_path = workdir / "engine-config.json"
        config_path.write_text(json.dumps(config))
        log = workdir / "engine.log"
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        with log.open("ab") as log_file:
            proc = subprocess.Popen(
                [python, "-I", str(WORKER_PATH), "--config", str(config_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                cwd=workdir,
                start_new_session=True,
                env=env,
                text=True,
                bufsize=1,
            )
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
        client = cls(proc, lines, log, config)
        try:
            client.ready = client._read_ready()
            installed = client.ready.get("library", "")
            if _public_version(installed) != _public_version(expect_library):
                raise EngineError(
                    f"the {client._engine} worker reports library {installed!r}, "
                    f"but {expect_library!r} is required"
                )
        except BaseException:
            client.close()
            raise
        return client

    def _read_ready(self) -> dict:
        line = self._next_line(READY_TIMEOUT_S, "ready")
        if line is None:
            raise EngineError(self._died("before it was ready"))
        message = self._parse(line)
        if "error" in message:
            self._reap()
            raise EngineError(
                f"{message['error']}\nlast {LOG_TAIL_LINES} lines of {self._log}:\n{tail(self._log)}"
            )
        if message.get("ready") is not True:
            raise EngineError(f"unexpected line from the engine worker: {line[:200]!r}")
        return message

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> EngineReply:
        """Run one greedy generation in the worker and return its tokens, engine-side seconds and counters.

        `accepted` is the engine's count minus the drafts accepted beyond the budget when the output ends by
        length, so that 1 + sum over rounds of (accepted + 1) == len(output_ids); `raw_accepted` is the
        engine's own count. Nothing is subtracted when the output ends in a stop id.

        Raises EngineError on a worker error, crash or timeout, on any prefix-cache hit, when the stats
        presence does not match whether the config has a draft model, and when the output is shorter than
        `max_new_tokens` without ending in a stop id.
        """
        if self._closed:
            raise EngineError("the engine client is closed")
        self._next_id += 1
        request = {"id": self._next_id, "prompt_ids": list(prompt_ids), "max_new_tokens": max_new_tokens}
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise EngineError(self._died("while receiving a request")) from None
        line = self._next_line(REQUEST_TIMEOUT_S, "reply")
        if line is None:
            raise EngineError(self._died("during a request"))
        reply = self._parse(line)
        if reply.get("id") != self._next_id:
            self._kill_group()
            raise EngineError(
                f"unexpected reply from the engine worker (wanted id {self._next_id}): {line[:200]!r}"
            )
        if "error" in reply:
            raise EngineError(f"the {self._engine} worker failed a request: {reply['error']}")
        if reply["cache_hits"] != 0:
            raise EngineError(
                f"the engine reused {reply['cache_hits']} cached prompt tokens; prefix caching must be off"
            )
        stats = reply["stats"]
        speculative = self._config.get("draft") is not None
        if speculative != (stats is not None):
            raise EngineError(
                f"the engine reply has {'no' if stats is None else 'some'} speculative stats but the config "
                f"{'has' if speculative else 'has no'} draft model"
            )
        output = list(reply["output_ids"])
        stopped = bool(output) and output[-1] in self._stop_ids
        if len(output) < max_new_tokens and not stopped:
            raise EngineError(
                f"the {self._engine} worker returned {len(output)} tokens for max_new_tokens "
                f"{max_new_tokens} without ending in a stop id: the engine truncated the generation"
            )
        self._raw["requests"] += 1
        if stats is None:
            return EngineReply(output, float(reply["seconds"]), None, None, None)
        rounds, proposed, raw_accepted = int(stats["rounds"]), int(stats["proposed"]), int(stats["accepted"])
        accepted = raw_accepted if stopped else min(raw_accepted, max(len(output) - 1 - rounds, 0))
        self._raw["rounds"] += rounds
        self._raw["proposed"] += proposed
        self._raw["accepted"] += raw_accepted
        self._raw["discarded_accepted"] += raw_accepted - accepted
        return EngineReply(output, float(reply["seconds"]), rounds, proposed, accepted, raw_accepted)

    def raw_counters(self) -> dict[str, int]:
        """Totals over every request so far: `requests`, the engine's own `rounds`, `proposed` and `accepted`
        before normalization, and `discarded_accepted`, the accepted drafts that normalization removed."""
        return dict(self._raw)

    def close(self) -> None:
        """Ask the worker to exit, then terminate its whole process group if it lingers. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._proc.returncode is None:
            with contextlib.suppress(OSError):
                self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self._proc.stdin.flush()
            self._exited(EXIT_TIMEOUT_S)
        self._kill_group()
        with contextlib.suppress(OSError):
            self._proc.stdin.close()

    def _next_line(self, timeout: float, what: str) -> str | None:
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            self._kill_group()
            raise EngineError(
                f"the {self._engine} worker gave no {what} (not ready or not answering after {timeout:g} s); "
                f"last {LOG_TAIL_LINES} lines of {self._log}:\n{tail(self._log)}"
            ) from None

    def _parse(self, line: str) -> dict:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            message = None
        if not isinstance(message, dict):
            self._kill_group()
            raise EngineError(f"unexpected line from the engine worker: {line[:200]!r}")
        return message

    def _reap(self) -> int | None:
        """Wait up to REAP_TIMEOUT_S for the worker to exit, clean up its group, return the exit code."""
        self._exited(REAP_TIMEOUT_S)
        self._kill_group()
        return self._proc.returncode

    def _exited(self, timeout: float) -> bool:
        """Wait up to `timeout` s for the worker to exit without reaping it; True if it has exited."""
        if self._proc.returncode is not None:
            return True
        deadline = time.monotonic() + timeout
        while os.waitid(os.P_PID, self._proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def _died(self, when: str) -> str:
        code = self._reap()
        return (
            f"{self._engine} worker exited with code {code} {when}; "
            f"last {LOG_TAIL_LINES} lines of {self._log}:\n{tail(self._log)}"
        )

    def _kill_group(self) -> None:
        """SIGTERM, wait TERM_GRACE_S for the worker to exit, SIGKILL its process group, then reap the worker.

        The group (the worker's own session) is signalled only while the worker is not reaped, so its id
        cannot belong to another process. Raises EngineError if the worker is still not gone REAP_TIMEOUT_S
        after SIGKILL.
        """
        if self._proc.returncode is None:
            pgid = self._proc.pid
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
            self._exited(TERM_GRACE_S)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        try:
            self._proc.wait(timeout=REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise EngineError(
                f"the {self._engine} worker (pid {self._proc.pid}) is still running {REAP_TIMEOUT_S:g} s "
                "after SIGKILL"
            ) from None


def _pump(stream, lines: queue.Queue) -> None:
    with stream:
        for line in stream:
            lines.put(line.rstrip("\n"))
    lines.put(None)
