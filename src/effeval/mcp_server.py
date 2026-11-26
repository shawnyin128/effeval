"""MCP server for agents: lab catalog, spec validation, runs on this machine's free GPUs, results, tables.

Lab-server calls go through HttpClient with the saved login (`effeval login`). A run executes the command
the server renders for a spec or for table cells, in the background on one free GPU, so results upload and
fill tables exactly as a run started by hand would.
"""

from __future__ import annotations

import difflib
import functools
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from effeval.core.task_search import search_tasks as rank_tasks
from effeval.store.http import ClientError, HttpClient, credentials_file, saved_login

GUIDE = """effeval: evaluation infra for efficiency AI research. Typical flow:
1. describe: see models, methods (with paper presets), evaluators and the common tasks. The full task list
   is too large to list: find any other task with search_tasks and read its settings with get_task.
2. Draft a RunSpec: {"model": {"name": <model code>}, "method": {"name": <method>, "preset": <preset>},
   "tasks": [{"name": <task>}, ...]}. Task names are namespaced: lm_eval/<task>, evalscope/<bench>[:<subset>],
   native/<task>, kernel/<task>. Baselines start from their paper preset;
   leaving "tasks" out runs the preset's. A task takes "params" (see get_task's params_schema); "limit"
   evaluates a subset: {"limit": 200} the first 200 samples, {"limit": 0.1} 10% of them, -1 or absent all.
3. validate_spec until it returns no issue with severity "error".
4. run_spec starts the run on a free GPU of this machine ("hardware": {"count": N} takes N free GPUs at once,
   for models too big for one); poll run_status. Results upload to the lab server.
5. query_records / get_record read results.
Experiment tables: create_project (once), save_table with {"name", "field", "models", "tasks",
"rows": [{"label", "method": {...}}], "shared": {...}}, then run_table to run every pending or failed cell;
finished runs fill the cells.
Repeated runs (robustness): a RunSpec, or a table's "shared", takes "repeats" in one of two forms. {"count": N}
runs every job N times with the seed "determinism": {"seed": S} (the same seed each time). {"seeds": [3, 5, 7]}
runs every job once per listed seed, so the number of runs is the length of the list (no duplicates;
"determinism.seed" is then unused; a "count", if given, must equal the list length). A run is identified by its
seed, so a stored result with the same settings and seed is reused, also across different seed lists. Each
run is its own run. A table cell then shows the mean over its runs, with the sample std and n in the cell's
"spread"; its "repeats" lists the per-run cells. Likelihood-scored tasks and greedy generation are
deterministic, so their runs agree exactly; only sampled generation varies.
模型前处理 (model preprocessing): rotation and/or scaling applied to the model before the quantizer. A RunSpec
or table row takes "preprocess": a preset name from describe's preprocess_presets, or
{"preset"?, "steps": [...]}. A step has a kind (rotation | scaling; both = two steps, order matters) and
entries; an entry has a position (an id from get_architecture(model): residual, norm.<module>.out,
linear.<proj>.input, attn.v_head, attn.qk_pre_rope, attn.qk_rope), a mode (offline = folded into the weights;
online = a runtime op) and a source ({"type": ..., options}, see preprocess_sources). get_architecture also
says which kind x mode is valid at each position and why not.
The quantizer stays the method; a preprocess preset lists its paired method presets (describe: "quantizers")
and needs one of them set explicitly. Example "QuaRot + GPTQ W4A4KV4": save_table rows
[{"label": "QuaRot + GPTQ W4A4KV4", "preprocess": "quarot",
"method": {"name": "gptq", "preset": "quarot_w4a4kv4"}}].
Speculative decoding: use field "speculative_decoding". A RunSpec takes "decoding": {"mode": "speculative",
"draft": {"model": <model code>}, "num_speculative_tokens": k} (the target is the top-level "model"; the draft
and k, 1 to 16, are required, and the draft must share the target's vocabulary) and "runtime": {"name": "hf" |
"vllm" | "sglang"} for the provider (hf = transformers). describe's "runtimes" says which providers this host
can run (vllm and sglang need their own Python configured); those two run only speed tasks
(speed/<prompt set>: speed/spec_bench, or a user JSONL prompt set as speed/<name>; find them with
search_tasks). Timed (speed) tasks need "hardware": {"gpu": ...} and "generation": {"batch_size": 1}.
Each job is compared with plain
autoregressive decoding of the same target on the same provider, which runs as its reference. A table row
takes "decoding" and "runtime" like a RunSpec; the draft model, k and provider are config columns of rows and
records ("draft", "decoding", "runtime"), not part of the method label.
Speed tasks may sample under speculative decoding (generation.do_sample, temperature, top_p, top_k);
every other task decodes greedily. For independent sampled runs use repeats.seeds.
"""

DESCRIPTION_CHARS = 160
"""Task descriptions in search results are cut to this many characters."""

FREE_MEMORY_MIB = 1000
"""A GPU with less memory in use than this, and no compute process on it, counts as free."""


def parse_free_gpus(gpus_csv: str, apps_csv: str, busy: set[int]) -> list[int]:
    """Free GPU indices, leaving out `busy` ones.

    `gpus_csv` and `apps_csv` are the outputs of `nvidia-smi --query-gpu=index,uuid,memory.used` and
    `--query-compute-apps=gpu_uuid` with `--format=csv,noheader,nounits`."""
    in_use = {line.strip() for line in apps_csv.splitlines() if line.strip()}
    free = []
    for line in gpus_csv.splitlines():
        if not line.strip():
            continue
        index, uuid, memory = (part.strip() for part in line.split(","))
        if int(memory) < FREE_MEMORY_MIB and uuid not in in_use and int(index) not in busy:
            free.append(int(index))
    return free


def query_free_gpus(busy: set[int]) -> list[int]:
    """Free GPU indices on this machine according to nvidia-smi, leaving out `busy` ones."""

    def smi(query: str) -> str:
        return subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True
        ).stdout

    return parse_free_gpus(
        smi("--query-gpu=index,uuid,memory.used"), smi("--query-compute-apps=gpu_uuid"), busy
    )


def compact_task(task: dict[str, Any]) -> dict[str, Any]:
    """One-line view of a task entry from describe: identity, modes, default few-shot, primary metric,
    shortened description, and whether it is a group (with its direct-member count)."""
    description = " ".join(task["description"].split())
    if len(description) > DESCRIPTION_CHARS:
        description = description[: DESCRIPTION_CHARS - 1] + "…"
    primary = task["metrics"][0]
    return {
        "name": task["name"],
        "evaluator": task["evaluator"],
        "modes": task["modes"],
        "category": task["category"],
        "num_fewshot": task["num_fewshot"],
        "primary_metric": primary["name"],
        "higher_is_better": primary["higher_is_better"],
        "description": description,
        "is_group": bool(task["subtasks"]),
        "subtasks": len(task["subtasks"]),
    }


def search_task_entries(
    tasks: dict[str, dict[str, Any]],
    query: str,
    evaluator: str | None,
    limit: int,
    category: str | None = None,
    mode: str | None = None,
    titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Tasks matching every whitespace-separated term of `query` (case-insensitive), best matches first:
    {"total", "returned", "tasks": [compact_task, ...]}.

    Ranking is the web task picker's (see effeval.core.task_search.search_tasks): names, tags, metric names,
    category titles (`titles`: category to title) and descriptions, whole words before substrings, with
    aliases such as ppl for perplexity. `category` (e.g. language_modeling, math, code) and `mode`
    (loglikelihood or generation) keep only tasks in that category or able to run in that mode. Raises
    ValueError for an unknown `evaluator`, `category` or `mode`, or a `limit` below 1."""
    evaluators = sorted({t["evaluator"] for t in tasks.values()})
    if evaluator is not None and evaluator not in evaluators:
        raise ValueError(f"no evaluator {evaluator!r}; evaluators: {evaluators}")
    categories = sorted({t["category"] for t in tasks.values()})
    if category is not None and category not in categories:
        raise ValueError(f"no category {category!r}; categories: {categories}")
    if mode is not None and mode not in {m for t in tasks.values() for m in t["modes"]}:
        raise ValueError(f"no mode {mode!r}; modes: loglikelihood, generation")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates = {
        name: task
        for name, task in tasks.items()
        if (evaluator is None or task["evaluator"] == evaluator)
        and (category is None or task["category"] == category)
        and (mode is None or mode in task["modes"])
    }
    ranked = rank_tasks(candidates, query, titles)
    return {
        "total": len(ranked),
        "returned": min(limit, len(ranked)),
        "tasks": [compact_task(tasks[name]) for name in ranked[:limit]],
    }


def compact_preprocess(described: dict[str, Any]) -> dict[str, Any]:
    """describe's preprocess part in short form: per preset its title, description, paper title and the
    position, mode and source type of every entry; per source type its version, flags and option defaults."""
    presets = {
        name: {
            "title": p["title"],
            "description": p["description"],
            "paper": p["paper"]["title"],
            "steps": [
                {
                    "kind": step["kind"],
                    "entries": [
                        {"position": e["position"], "mode": e["mode"], "source": e["source"]["type"]}
                        for e in step["entries"]
                    ],
                }
                for step in p["steps"]
            ],
        }
        for name, p in described["preprocess_presets"].items()
    }
    sources = {
        kind: {
            name: {
                "version": s["version"],
                "needs_data": s["needs_data"],
                "implemented": s["implemented"],
                "positions": s["positions"],
                "no_paper": s["no_paper"],
                "options": {k: v.get("default") for k, v in s["schema"]["properties"].items() if k != "type"},
            }
            for name, s in by_type.items()
        }
        for kind, by_type in described["preprocess_sources"].items()
    }
    return {"preprocess_presets": presets, "preprocess_sources": sources}


def with_preprocess(spec: dict[str, Any]) -> dict[str, Any]:
    """`spec` with a preset-name `preprocess` expanded to the {"preset": name} a RunSpec takes."""
    if not isinstance(spec.get("preprocess"), str):
        return spec
    return {**spec, "preprocess": {"preset": spec["preprocess"]}}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class LocalRunner:
    """Runs rendered commands in the background, `hardware.count` GPUs each; state in `root/<plan_id>/`."""

    def __init__(
        self,
        root: Path,
        free_gpus: Callable[[set[int]], list[int]] = query_free_gpus,
        spawn: Callable[..., Any] = subprocess.Popen,
    ):
        self.root = root
        self.free_gpus = free_gpus
        self.spawn = spawn

    def _meta(self, plan_id: str) -> dict[str, Any]:
        return json.loads((self.root / plan_id / "meta.json").read_text())

    def status(self, plan_id: str) -> dict[str, Any]:
        """{plan_id, gpu, pid, started_at, state: running | finished | failed | lost, exit_code, log_tail}."""
        run_dir = self.root / plan_id
        if not (run_dir / "meta.json").exists():
            raise ValueError(f"no run {plan_id} on this machine")
        meta = self._meta(plan_id)
        meta.setdefault("gpus", [meta["gpu"]])
        exit_file = run_dir / "exit_code"
        exit_code = int(exit_file.read_text()) if exit_file.exists() else None
        if exit_code is not None:
            state = "finished" if exit_code == 0 else "failed"
        else:
            state = "running" if _alive(meta["pid"]) else "lost"
        log = run_dir / "run.log"
        tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
        return {**meta, "state": state, "exit_code": exit_code, "log_tail": "\n".join(tail)}

    def runs(self) -> list[dict[str, Any]]:
        """Status of every run started here, newest first, without log tails."""
        if not self.root.exists():
            return []
        found = [self.status(d.name) for d in self.root.iterdir() if (d / "meta.json").exists()]
        newest_first = sorted(found, key=lambda r: -r["started_at"])
        return [{k: v for k, v in r.items() if k != "log_tail"} for r in newest_first]

    def busy_gpus(self) -> set[int]:
        """GPUs held by runs started here that are still running."""
        return {g for r in self.runs() if r["state"] == "running" for g in r["gpus"]}

    def start(self, plan_id: str, command: dict[str, str], num_gpus: int = 1) -> dict[str, Any]:
        """Run one rendered command (its `write` and `run` steps) on `num_gpus` free GPUs.

        Raises RuntimeError if fewer than `num_gpus` GPUs are free (nothing is started) or the plan already
        has an unfinished run here."""
        run_dir = self.root / plan_id
        if (run_dir / "meta.json").exists() and self.status(plan_id)["state"] in ("running", "finished"):
            raise RuntimeError(f"plan {plan_id} already has a run on this machine")
        free = self.free_gpus(self.busy_gpus())
        if len(free) < num_gpus:
            raise RuntimeError(
                f"{num_gpus} free GPU(s) needed on this machine, {len(free)} free right now; try again later"
                if num_gpus > 1
                else "no free GPU on this machine right now; try again later"
            )
        gpus = free[:num_gpus]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "exit_code").unlink(missing_ok=True)
        (run_dir / "run.sh").write_text(f"set -e\n{command['write']}\n{command['run'].strip()} --yes\n")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        process = self.spawn(
            ["bash", "-c", "bash run.sh > run.log 2>&1; echo $? > exit_code"],
            cwd=run_dir,
            env=env,
            start_new_session=True,
        )
        meta = {
            "plan_id": plan_id,
            "gpu": gpus[0],
            "gpus": gpus,
            "pid": process.pid,
            "started_at": time.time(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta))
        return meta


def build_server(client: HttpClient, runner: LocalRunner) -> MCPServer:
    """The MCP server with every effeval tool bound to `client` and `runner`."""
    server = MCPServer("effeval", instructions=GUIDE)
    catalog: dict[str, Any] = {}

    def fetch_catalog(refresh: bool = False) -> dict[str, Any]:
        """The server's full describe payload, fetched once per process unless `refresh`."""
        if refresh or not catalog:
            catalog.clear()
            catalog.update(client.describe())
        return catalog

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register `fn`; lab-server errors and run failures reach the agent as messages it can act on."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except (ClientError, RuntimeError, ValueError) as exc:
                raise ToolError(str(exc)) from exc

        return server.tool()(wrapper)

    def launch(rendered: dict[str, Any]) -> dict[str, Any]:
        errors = [i for i in rendered["issues"] if i["severity"] == "error"]
        plan = rendered["plan"]
        if errors or plan is None:
            return {"started": False, "issues": rendered["issues"]}
        summary = {
            "plan_id": plan["plan_id"],
            "jobs": len(plan["jobs"]),
            "reused": len(plan["reused"]),
            "estimate": rendered["estimate"],
            "issues": rendered["issues"],
        }
        if not rendered["commands"]:
            return {"started": False, **summary, "note": "every job is reused from stored results"}
        num_gpus = max(job["spec"]["hardware"]["count"] for job in plan["jobs"])
        return {
            "started": True,
            **summary,
            **runner.start(plan["plan_id"], rendered["commands"][0], num_gpus),
        }

    @tool
    def describe() -> dict[str, Any]:
        """Models, methods (manifests, paper presets), evaluators, fields and GPUs the server knows.

        Tasks (thousands) are not listed: `common_tasks` holds the curated ones and `task_counts` the number
        of tasks per evaluator. Use search_tasks to find any other task, get_task for one task's settings.
        Model preprocessing: `preprocess_presets` (title, description, paper, steps in short form) and
        `preprocess_sources` (per kind and source type: version, needs_data, implemented, option defaults).
        Architecture graphs and per-model position tables are left out: use get_architecture(model)."""
        full = fetch_catalog(refresh=True)
        tasks = full["tasks"]
        brief = {k: v for k, v in full.items() if k != "tasks"}
        brief.update(compact_preprocess(full))
        brief["architectures"] = {n: {**a, "graph": None} for n, a in full["architectures"].items()}
        brief["models"] = {
            c: {k: v for k, v in m.items() if k != "positions"} for c, m in full["models"].items()
        }
        brief["common_tasks"] = [compact_task(tasks[n]) for n in full["common_tasks"]]
        brief["task_counts"] = {
            e: sum(t["evaluator"] == e for t in tasks.values()) for e in full["evaluators"]
        }
        return brief

    @tool
    def get_architecture(name: str) -> dict[str, Any]:
        """A model's architecture for choosing preprocessing positions: {model, architecture, shape, graph,
        positions}. `name` is a model code (e.g. qwen3-0.6b) or an architecture class name (then `model`,
        `shape` and `positions` are null). `graph` is one decoder layer (nodes, edges, globals); `positions`
        maps each position id to {"rotation" | "scaling": {"offline" | "online": "ok" or the reason it is
        not valid}}."""
        full = fetch_catalog()
        model = full["models"].get(name)
        architecture = full["architectures"].get(name if model is None else model["architecture"])
        if architecture is None:
            known = sorted(set(full["models"]) | set(full["architectures"]))
            raise ValueError(f"no model or architecture {name!r}; known: {known}")
        return {
            "model": None if model is None else model["code"],
            "architecture": architecture["name"],
            "shape": None if model is None else model["shape"],
            "graph": architecture["graph"],
            "positions": None if model is None else model["positions"],
        }

    @tool
    def search_tasks(
        query: str = "",
        evaluator: str | None = None,
        limit: int = 30,
        category: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Find evaluation tasks by name, tags, metric, category or description (case-insensitive; every word
        of `query` must match; best matches first, whole words before substrings, aliases such as ppl for
        perplexity; subtasks of a matching group are left out, open the group with get_task). `evaluator` (lm_eval, evalscope, native, kernel), `category` (language_modeling =
        perplexity tasks, qa, commonsense, reasoning, math, code, ...) and `mode` (loglikelihood or
        generation) narrow the search.

        Returns {total, returned, tasks}: `total` counts all matches, `tasks` holds at most `limit` compact
        entries (name, evaluator, modes, category, num_fewshot, primary_metric, higher_is_better, description,
        is_group, subtasks). Use `name` as a RunSpec task name; get_task gives the full entry."""
        catalog = fetch_catalog()
        titles = {n: c.get("title") or c["label"] for n, c in catalog["categories"].items()}
        return search_task_entries(catalog["tasks"], query, evaluator, limit, category, mode, titles)

    @tool
    def get_task(name: str) -> dict[str, Any]:
        """One task's full entry by namespaced name (e.g. lm_eval/mmlu): modes, default few-shot, the settings
        it accepts (`params_schema`, `fewshot_allowed`), metrics with higher_is_better, and its subtasks."""
        tasks = fetch_catalog()["tasks"]
        if name not in tasks:
            close = difflib.get_close_matches(name, tasks, n=5, cutoff=0.6)
            hint = f"; did you mean {close}" if close else "; use search_tasks to find it"
            raise ValueError(f"no task {name!r}{hint}")
        return tasks[name]

    @tool
    def validate_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Check a RunSpec; returns issues (severity error / warning / info, code, paths, message, hint)."""
        return client.validate(with_preprocess(spec))

    @tool
    def run_spec(spec: dict[str, Any], to: str = "private") -> dict[str, Any]:
        """Freeze a RunSpec and run it on a free GPU of this machine. `to`: private, pool, or project:<name>.
        Returns started=False with the issues when the spec has errors or everything is reused.
        `spec.repeats` = {"count": N} runs each job N times with the same seed; {"seeds": [3, 5]} runs it once per
        listed seed (see the guide). `spec.preprocess` is a preset name or {"preset"?, "steps"?}."""
        return launch(client.render(with_preprocess(spec), 1, to))

    @tool
    def run_status(plan_id: str) -> dict[str, Any]:
        """State of a run started here (running / finished / failed / lost) and its log tail.

        `records` maps each job_id to the record it has uploaded so far."""
        status = runner.status(plan_id)
        lab = next((r for r in client.runs() if r["plan_id"] == plan_id), None)
        return {**status, "records": lab["records"] if lab else {}}

    @tool
    def list_runs() -> list[dict[str, Any]]:
        """Runs started on this machine through this server, newest first."""
        return runner.runs()

    @tool
    def free_gpus() -> list[int]:
        """GPUs on this machine that are free now (no other process, almost no memory in use)."""
        return runner.free_gpus(runner.busy_gpus())

    @tool
    def query_records(scope: str = "mine", filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Result rows. `scope`: mine, pool, or project:<name>.

        `filters`: field=value pairs, e.g. {"model": "qwen3-0.6b"}."""
        return client.records(scope, filters or {})

    @tool
    def get_record(record_id: str) -> dict[str, Any]:
        """One result with its metrics, comparisons, spec and environment."""
        return client.record(record_id)

    @tool
    def list_projects() -> list[dict[str, Any]]:
        """Projects the user belongs to."""
        return client.projects()

    @tool
    def create_project(name: str, description: str = "") -> dict[str, Any]:
        """Create a project (lower-case name, e.g. "paper-a"); members share results and experiment tables."""
        return client.create_project(name, description)

    @tool
    def list_tables(project: str) -> list[dict[str, Any]]:
        """A project's experiment tables, each cell with its status.

        Status is one of done, reusable, running, failed, pending."""
        return client.tables(project)

    @tool
    def save_table(project: str, table: dict[str, Any], table_id: str | None = None) -> dict[str, Any]:
        """Create (no table_id) or replace an experiment table.

        `table`: {"name", "field", "models": [...], "tasks": [...],
        "rows": [{"label", "method": {"name", "preset"?, "config"?}, "preprocess"?}],
        "shared": {RunSpec settings for every row}}.

        Put "repeats": {"count": N} (same seed each time) or {"seeds": [3, 5, 7]} (one run per seed) in `shared`
        to run every cell several times; a cell then reports the mean over its runs and `spread`
        ({metric: {"std", "n"}}, sample std).

        A row's (or `shared`) "preprocess" is a preset name or {"preset"?, "steps"?}. A row's `title` is
        "<preprocess title> + <method title>", e.g. "QuaRot + GPTQ"; a row without a label is named so."""
        return client.save_table(project, table, table_id)

    @tool
    def run_table(
        project: str, table_id: str, run_keys: list[str] | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Run table cells on a free GPU: the given run_keys, or every pending or failed cell.

        Results go into the project and fill the cells. `force` reruns cells that already have a result."""
        if run_keys is None:
            table = next((t for t in client.tables(project) if t["table_id"] == table_id), None)
            if table is None:
                raise ValueError(f"no table {table_id} in project {project!r}")
            run_keys = sorted(
                {
                    part["run_key"]
                    for block in table["blocks"]
                    for row in block["rows"]
                    for cell in row["cells"].values()
                    for part in cell.get("repeats", [cell])
                    if part["status"] in ("pending", "failed")
                }
            )
            if not run_keys:
                return {"started": False, "note": "no pending or failed cell", "counts": table["counts"]}
        return launch(client.table_command(project, table_id, run_keys, force))

    @tool
    def annotate(
        record_id: str, tags: list[str], note: str = "", status: str = "unreviewed"
    ) -> dict[str, bool]:
        """Replace a record's tags, note and review status."""
        client.annotate(record_id, tags, note, status)
        return {"ok": True}

    return server


def main(server_url: str | None = None) -> None:
    """Serve over stdio with the saved login; runs live under `$EFFEVAL_HOME/runs`."""
    server, token = saved_login(server_url)
    if server is None or token is None:
        raise SystemExit(
            "not logged in: run `effeval login --server URL` (or set EFFEVAL_SERVER/EFFEVAL_TOKEN)"
        )
    build_server(HttpClient(server, token), LocalRunner(credentials_file().parent / "runs")).run("stdio")
