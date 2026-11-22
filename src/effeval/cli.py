"""`effeval` command line. Exit codes: 0 success, 1 runtime failure, 2 validation or node-check errors."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from effeval.api import service
from effeval.core.catalog import builtin_catalog
from effeval.core.types import Issue, Severity
from effeval.store.http import ClientError, HttpClient, credentials_file, saved_login


class CliError(Exception):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


DESTINATION_HELP = "where results go: private (default), pool, or project:<name>"


def _server_and_token(args: argparse.Namespace) -> tuple[str | None, str | None]:
    return saved_login(getattr(args, "server", None))


def _client(args: argparse.Namespace) -> HttpClient:
    server, token = _server_and_token(args)
    if server is None or token is None:
        raise CliError(
            "not logged in: run `effeval login --server URL` (or set EFFEVAL_SERVER/EFFEVAL_TOKEN)"
        )
    return HttpClient(server, token)


def _print_issues(issues: list[Issue], show_info: bool = True) -> None:
    for issue in issues:
        if issue.severity == Severity.INFO and not show_info:
            continue
        where = f" {', '.join(issue.paths)}" if issue.paths else ""
        print(f"{issue.severity.value}[{issue.code}]{where}: {issue.message}", file=sys.stderr)
        if issue.hint:
            print(f"  hint: {issue.hint}", file=sys.stderr)


def _has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == Severity.ERROR for i in issues)


def _document(args: argparse.Namespace) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise CliError(f"{args.config} does not contain a mapping", 2)
        doc = loaded
    if getattr(args, "baseline", None):
        if "plan_version" in doc:
            raise CliError("--baseline cannot be combined with a frozen plan", 2)
        method = doc.setdefault("method", {})
        if method.get("name") not in (None, args.baseline):
            raise CliError(
                f"--baseline {args.baseline} conflicts with method.name {method['name']!r} in the config", 2
            )
        method["name"] = args.baseline
        manifest = builtin_catalog().manifests.get(args.baseline)
        if manifest is not None and manifest.field != "any":
            doc.setdefault("field", manifest.field)
    if not doc:
        raise CliError("give --config FILE (and optionally --baseline NAME)", 2)
    return doc


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False) if getattr(args, "json", False) else text)


# commands ---------------------------------------------------------------------------------------------------


def cmd_login(args: argparse.Namespace) -> int:
    username = args.username or input("username: ")
    password = getpass.getpass("password: ")
    try:
        token = HttpClient(args.server, None).password_token(
            username,
            password,
            label=args.label or f"cli@{os.uname().nodename}",
            can_publish=not args.no_publish,
        )
    except ClientError as exc:
        raise CliError(str(exc)) from exc
    path = credentials_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"server": args.server, "token": token["token"]}))
    path.chmod(0o600)
    print(
        f"logged in to {args.server} as {token['username']} until {token['expires_at'][:10]}; "
        f"credentials saved to {path}"
    )
    return 0


def cmd_logout(_args: argparse.Namespace) -> int:
    credentials_file().unlink(missing_ok=True)
    print("credentials removed")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from effeval.api import execute  # compute-only dependency (torch); server installs never reach this

    report, _ = execute.doctor(*_server_and_token(args))
    mode = "online" if report.online else "offline"
    _emit(args, report.model_dump(), f"mode: {mode} ({report.reason})\nGPU: {report.gpu} x{report.gpu_count}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from effeval.validate import validate

    doc = _document(args)
    catalog = builtin_catalog()
    server, token = _server_and_token(args)
    if server and token:
        try:
            client = HttpClient(server, token)
            catalog = catalog.with_manifests(client.manifests()).with_models(client.models())
        except ClientError as exc:
            print(
                f"warning: custom methods and models unavailable ({exc}); validating against builtins only",
                file=sys.stderr,
            )
    issues = validate(doc, catalog)
    if args.json:
        print(json.dumps([i.model_dump(mode="json") for i in issues], indent=2))
    else:
        _print_issues(issues)
        print("ok" if not _has_errors(issues) else "invalid")
    return 2 if _has_errors(issues) else 0


def cmd_render(args: argparse.Namespace) -> int:
    doc = _document(args)
    server, token = _server_and_token(args)
    if server and token:
        try:
            rendered = HttpClient(server, token).render(doc, args.shards)
        except ClientError as exc:
            raise CliError(f"render via server failed: {exc}") from exc
        issues = [Issue.model_validate(i) for i in rendered["issues"]]
        commands = rendered["commands"]
    else:
        from effeval.plan import freeze, shard

        plan, issues = freeze(doc, builtin_catalog(), owner=None, server=None, lookup=None)
        commands = (
            []
            if plan is None
            else [
                service.render_command(
                    shard(plan, k, args.shards) if args.shards > 1 else plan, None, k, args.shards
                ).model_dump()
                for k in range(1, args.shards + 1)
            ]
        )
    _print_issues(issues)
    if _has_errors(issues):
        return 2
    for index, command in enumerate(commands, start=1):
        if len(commands) > 1:
            print(f"# --- node {index}/{len(commands)} ---")
        print("# 1. write the plan")
        print(command["write"])
        print("# 2. run")
        print(command["run"])
        print("# 3. push the results package (only needed if the run could not reach the server)")
        print(command["push"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from effeval.api import execute  # compute-only dependency (torch)

    doc = _document(args)
    report, client = execute.doctor(*_server_and_token(args))
    print(f"[doctor] {'online' if report.online else 'offline'}: {report.reason}", file=sys.stderr)
    shard_spec = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        shard_spec = (k, n)

    def confirm(plan, issues) -> bool:
        _print_issues(issues)
        print(
            f"plan {plan.plan_id}: {len(plan.jobs)} job(s) to run, {len(plan.reused)} reused", file=sys.stderr
        )
        for job in plan.jobs:
            print(
                f"  [{job.role}] {job.spec.model.name} {job.spec.method.name} {job.spec.task.name} "
                f"({job.spec.task.mode.value}, {job.spec.evaluator.name})",
                file=sys.stderr,
            )
        for reused in plan.reused:
            print(f"  [reused {reused.role}] {reused.record_id}: {reused.reason}", file=sys.stderr)
        if args.yes:
            return True
        if not sys.stdin.isatty():
            raise CliError(
                "refusing to run without confirmation on a non-interactive terminal; pass --yes", 2
            )
        return input("run it? [y/N] ").strip().lower() in {"y", "yes"}

    outcome, issues = execute.run_document(
        doc,
        doctor_report=report,
        client=client,
        out_dir=Path(args.out),
        shard_spec=shard_spec,
        confirm=confirm,
        log=lambda line: print(line, file=sys.stderr),
        to=args.to,
        keep_going=args.keep_going,
    )
    if outcome is None:
        if _has_errors(issues):
            _print_issues([i for i in issues if i.severity == Severity.ERROR], show_info=False)
            return 2
        print("not run", file=sys.stderr)
        return 0
    print(f"pushed {len(outcome.pushed)} record(s)")
    for failure in outcome.failed:
        print(f"failed: {failure}", file=sys.stderr)
    if outcome.bundle_path:
        print(f"offline results: {outcome.bundle_path}")
        print(
            f"results package with per-sample files: {outcome.package_path} (upload it on the UI Import page)"
        )
        print(
            "paste the file content into the UI Import page, or run "
            f"`effeval push{'' if args.to == 'private' else ' --to ' + args.to}` on a machine "
            "that can reach the server"
        )
        if args.print_bundle:
            print(Path(outcome.bundle_path).read_text(encoding="utf-8"))
    if outcome.failed:
        print("re-run the same command with the same --out to run only the failed jobs", file=sys.stderr)
        return 1
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from effeval.api import execute  # compute-only dependency (torch)

    client = _client(args)
    imported = duplicates = 0
    for file in args.files:
        try:
            result = execute.push_file(Path(file), client, args.to)
        except ClientError as exc:
            raise CliError(f"{file}: {exc}") from exc
        imported += len(result["imported"])
        duplicates += len(result["duplicates"])
        print(f"{file}: imported {len(result['imported'])}, already stored {len(result['duplicates'])}")
    print(f"imported {imported} record(s) from {len(args.files)} bundle(s)")
    if duplicates:
        print(f"{duplicates} record(s) were already stored (e.g. pasted earlier) and were kept as is")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    filters = dict(f.split("=", 1) for f in args.filter)
    rows = _client(args).records(args.scope, filters)
    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if k != "record"} for r in rows], indent=2))
        return 0
    for r in rows:
        metrics = ", ".join(f"{k}={v}" for k, v in r["metrics"].items())
        deltas = ", ".join(f"{k}={v}" for k, v in r["comparisons"]["values"].items())
        print(
            f"{r['record_id']}  {r['model']}  {r['method']}  {r['task']}/{r['mode']}  {metrics}"
            f"{'  ' + deltas if deltas else ''}  [{r['visibility']}]"
        )
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    _client(args).publish(args.record_id)
    print("published")
    return 0


def cmd_retract(args: argparse.Namespace) -> int:
    _client(args).retract(args.record_id)
    print("retracted")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    _client(args).annotate(args.record_id, args.tag, args.note, args.status)
    print("annotated")
    return 0


def cmd_method_export(args: argparse.Namespace) -> int:
    from effeval.api import execute  # imports the method's own package

    manifest = execute.export_manifest(args.entry)
    if args.builtin:
        manifest = manifest.model_copy(update={"builtin": True})
    text = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_method_list(args: argparse.Namespace) -> int:
    from effeval.api import execute  # imports the methods' own packages

    for manifest in execute.discovered_manifests():
        uses = f" uses {', '.join(manifest.uses)}" if manifest.uses else ""
        print(f"{manifest.name} {manifest.version} ({manifest.field}) {manifest.entry}{uses}")
    return 0


def cmd_method_sync(args: argparse.Namespace) -> int:
    from effeval.api import execute  # imports the methods' own packages

    try:
        uploaded = execute.sync_methods(_client(args))
    except ClientError as exc:
        raise CliError(str(exc)) from exc
    for manifest in uploaded:
        print(f"synced {manifest.name} {manifest.version}")
    if not uploaded:
        print(
            'no methods found: declare them under [project.entry-points."effeval.methods"]', file=sys.stderr
        )
    return 0


def cmd_task_list(args: argparse.Namespace) -> int:
    catalog = builtin_catalog()
    query = (args.search or "").lower()
    for name, task in sorted(catalog.tasks.items()):
        if args.evaluator and task.evaluator != args.evaluator:
            continue
        if query and query not in name.lower() and query not in task.description.lower():
            continue
        print(f"{name}  {task.default_mode.value}  {task.category}  {task.metrics[0].name}")
    return 0


def cmd_task_sync(args: argparse.Namespace) -> int:
    from effeval.api import tasks  # loads the installed evaluators

    try:
        uploaded = tasks.sync_tasks(_client(args))
    except ClientError as exc:
        raise CliError(str(exc)) from exc
    for listing in uploaded:
        info = listing.evaluator
        print(f"synced {info.name} {info.library}=={info.library_version}: {len(listing.tasks)} tasks")
    if not uploaded:
        print(
            'no task listings: declare evaluators under [project.entry-points."effeval.evaluators"]',
            file=sys.stderr,
        )
    return 0


def cmd_task_snapshot(args: argparse.Namespace) -> int:
    from effeval.api import tasks  # loads the installed evaluators

    for path in tasks.write_snapshot(tasks.task_listings(), Path(args.out)):
        print(f"wrote {path}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    catalog = builtin_catalog()
    summary = {
        "methods": {
            n: {
                "field": m.field,
                "presets": sorted(catalog.presets.get(n, {})),
                "default_preset": m.default_preset,
            }
            for n, m in catalog.manifests.items()
        },
        "tasks": {
            e: sum(1 for t in catalog.tasks.values() if t.evaluator == e) for e in sorted(catalog.evaluators)
        },
        "models": {
            code: f"{m.checkpoint} ({m.architecture}, {m.dtype})" for code, m in catalog.models.items()
        },
        "architectures": sorted(catalog.architectures),
        "fields": sorted(catalog.fields),
        "gpus": sorted(catalog.gpus),
    }
    _emit(args, summary, yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_model_add(args: argparse.Namespace) -> int:
    from effeval.ops import introspect  # reads the checkpoint config with transformers

    entry, architecture = introspect.model_entry(args.checkpoint, args.revision, args.description)
    known = builtin_catalog().architectures
    if entry.architecture not in known:
        raise CliError(
            f"{entry.architecture} has no layout in this effeval install; a maintainer extracts it "
            f"with `effeval arch extract {args.checkpoint}`",
            2,
        )
    if architecture.model_dump(exclude={"extracted_with"}) != known[entry.architecture].model_dump(
        exclude={"extracted_with"}
    ):
        raise CliError(
            f"{args.checkpoint} lays out {entry.architecture} differently from the stored table", 2
        )
    text = yaml.safe_dump(entry.model_dump(mode="json"), sort_keys=False)
    if args.dry_run:
        print(text)
        return 0
    _client(args).register_model(entry)
    print(
        f"registered {entry.code} -> {entry.checkpoint}@{entry.revision or 'unpinned'} ({entry.architecture})"
    )
    return 0


def cmd_model_list(args: argparse.Namespace) -> int:
    server, token = _server_and_token(args)
    rows = _client(args).models() if server and token else list(builtin_catalog().models.values())
    for m in rows:
        print(
            f"{m.code:24} {m.checkpoint:36} {m.architecture:22} {m.dtype:9} {m.params / 1e9:6.2f}B "
            f"{'tied' if m.tie_word_embeddings else ''}"
        )
    return 0


def cmd_arch_extract(args: argparse.Namespace) -> int:
    from transformers import AutoConfig

    from effeval.ops import introspect  # maintainer tool: writes src/effeval/core/data/architectures

    config = AutoConfig.from_pretrained(args.checkpoint, revision=args.revision)
    architecture = introspect.extract_architecture(config, source=args.checkpoint)
    if architecture.uncovered_weights:
        print(f"warning: no linear site covers {architecture.uncovered_weights}", file=sys.stderr)
    doc = architecture.model_dump(mode="json")
    if architecture.graph is not None:
        doc["graph"] = architecture.graph.model_dump(mode="json", exclude_none=True)
    if args.output and Path(args.output).exists():
        kept = yaml.safe_load(Path(args.output).read_text(encoding="utf-8")).get("graph_overrides")
        if kept:
            doc["graph_overrides"] = kept
    text = "# Generated by `effeval arch extract`; do not edit by hand.\n" + yaml.safe_dump(
        doc, sort_keys=False
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_server_init(args: argparse.Namespace) -> int:
    from effeval.server.app import init_admin

    password = getpass.getpass(f"password for admin {args.admin}: ")
    if password != getpass.getpass("repeat: "):
        raise CliError("passwords differ", 2)
    init_admin(Path(args.data), args.admin, password)
    print(f"created admin {args.admin} in {args.data}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from effeval import mcp_server  # needs the mcp extra

    mcp_server.main(args.server)
    return 0


def cmd_server_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from effeval.server.app import create_app

    web = Path(args.web_dist) if args.web_dist else None
    uvicorn.run(create_app(Path(args.data), web, args.public_url), host=args.host, port=args.port)
    return 0


def cmd_admin(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.action == "list":
        for user in client.admin("GET", "/users"):
            print(f"{user['username']:20} {user['role']:7} {'disabled' if user['disabled'] else ''}")
        return 0
    if args.username is None:
        raise CliError("username required", 2)
    if args.action == "add":
        client.admin(
            "POST",
            "/users",
            {
                "username": args.username,
                "password": getpass.getpass("initial password: "),
                "role": "admin" if args.admin else "member",
            },
        )
    elif args.action in {"disable", "enable"}:
        client.admin("POST", f"/users/{args.username}/{args.action}")
    elif args.action == "reset-password":
        client.admin(
            "POST", f"/users/{args.username}/password", {"password": getpass.getpass("new password: ")}
        )
    else:
        raise CliError(f"unknown admin action {args.action!r}", 2)
    print("ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="effeval", description="Evaluation infra for efficiency AI research."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def spec_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", help="RunSpec YAML, or a frozen plan rendered by the UI")
        p.add_argument("--baseline", help="method name; its default paper preset applies unless overridden")

    p = sub.add_parser("login", help="exchange username/password for this machine's login token")
    p.add_argument("--server", required=True)
    p.add_argument("--username")
    p.add_argument("--label", help="token name; default cli@<hostname>, one active token per name")
    p.add_argument(
        "--no-publish", action="store_true", help="token cannot publish to the lab pool (for agents)"
    )
    p.set_defaults(fn=cmd_login)
    sub.add_parser("logout", help="forget saved credentials").set_defaults(fn=cmd_logout)

    p = sub.add_parser("doctor", help="check server connectivity, login and GPUs")
    p.add_argument("--server")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("validate", help="validate a RunSpec (conflicts exit with code 2)")
    spec_args(p)
    p.add_argument("--server")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("render", help="print the self-contained command(s) to paste on compute nodes")
    spec_args(p)
    p.add_argument("--server")
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("run", help="check this node, confirm, and run")
    spec_args(p)
    p.add_argument("--server")
    p.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--shard", help="k/N: run the k-th of N model groups")
    p.add_argument("--out", default="effeval-results")
    p.add_argument("--print-bundle", action="store_true", help="also print the offline bundle for copy-paste")
    p.add_argument("--to", default="private", help=DESTINATION_HELP)
    p.add_argument("--keep-going", action="store_true", help="run the remaining jobs after a failure")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("push", help="upload offline result bundles")
    p.add_argument("files", nargs="+", metavar="file")
    p.add_argument("--to", default="private", help=DESTINATION_HELP)
    p.add_argument("--server")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("query", help="list results")
    p.add_argument("--scope", default="mine", help="mine, pool, or project:<name>")
    p.add_argument("--filter", action="append", default=[], help="column or dotted record path = value")
    p.add_argument("--server")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_query)

    for name, fn in (("publish", cmd_publish), ("retract", cmd_retract)):
        p = sub.add_parser(name, help=f"{name} a record")
        p.add_argument("record_id")
        p.add_argument("--server")
        p.set_defaults(fn=fn)

    p = sub.add_parser("annotate", help="set tags, note and status of a record")
    p.add_argument("record_id")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--note", default="")
    p.add_argument("--status", default="unreviewed", choices=["unreviewed", "verified", "suspect"])
    p.add_argument("--server")
    p.set_defaults(fn=cmd_annotate)

    method = sub.add_parser("method", help="methods: discover and sync to the server").add_subparsers(
        dest="action", required=True
    )
    p = method.add_parser("export", help="manifest of a @method-decorated function")
    p.add_argument("entry", help="package.module:function")
    p.add_argument("--output", "-o")
    p.add_argument(
        "--builtin", action="store_true", help="maintainers: refresh src/effeval/core/data/methods"
    )
    p.set_defaults(fn=cmd_method_export)
    p = method.add_parser("list", help="methods installed packages announce under effeval.methods")
    p.set_defaults(fn=cmd_method_list)
    p = method.add_parser("sync", help="upload the manifests of every installed method to the server")
    p.add_argument("--server")
    p.set_defaults(fn=cmd_method_sync)

    task = sub.add_parser("task", help="evaluation tasks: list, sync to the server, snapshot").add_subparsers(
        dest="action", required=True
    )
    p = task.add_parser("list", help="tasks of this install's catalog")
    p.add_argument("--evaluator")
    p.add_argument("--search", help="substring of the task name or description")
    p.set_defaults(fn=cmd_task_list)
    p = task.add_parser("sync", help="upload the task listing of every installed evaluator (admins)")
    p.add_argument("--server")
    p.set_defaults(fn=cmd_task_sync)
    p = task.add_parser("snapshot", help="maintainers: write the listings as the shipped snapshot")
    p.add_argument("--out", required=True, help="directory, normally src/effeval/core/data/task_catalogs")
    p.set_defaults(fn=cmd_task_snapshot)

    model = sub.add_parser("model", help="global model codes").add_subparsers(dest="action", required=True)
    p = model.add_parser("add", help="register a checkpoint under its code (lower-cased repo name)")
    p.add_argument("checkpoint", help="HF model id or a local directory on compute nodes")
    p.add_argument("--revision", help="branch, tag or sha to pin (HF ids default to main)")
    p.add_argument("--description", default="")
    p.add_argument("--dry-run", action="store_true", help="print the entry instead of registering it")
    p.add_argument("--server")
    p.set_defaults(fn=cmd_model_add)
    p = model.add_parser("list", help="list registered models")
    p.add_argument("--server")
    p.set_defaults(fn=cmd_model_list)

    arch = sub.add_parser("arch", help="architecture layouts").add_subparsers(dest="action", required=True)
    p = arch.add_parser("extract", help="maintainers: extract an architecture layout from HF metadata")
    p.add_argument("checkpoint")
    p.add_argument("--revision")
    p.add_argument("--output", "-o")
    p.set_defaults(fn=cmd_arch_extract)

    p = sub.add_parser("describe", help="methods, presets, models, tasks, fields and GPUs this install knows")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("mcp", help="serve the MCP tools for agents over stdio (runs use this machine's GPUs)")
    p.add_argument("--server", help="lab server URL; default: the saved login")
    p.set_defaults(fn=cmd_mcp)

    server = sub.add_parser("server", help="run the lab server").add_subparsers(dest="action", required=True)
    p = server.add_parser("init", help="create the first admin")
    p.add_argument("--data", required=True)
    p.add_argument("--admin", required=True)
    p.set_defaults(fn=cmd_server_init)
    p = server.add_parser("serve", help="serve the API and the web UI")
    p.add_argument("--data", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8600)
    p.add_argument("--web-dist", help="built web UI directory (web/dist)")
    p.add_argument("--public-url", help="URL compute nodes should use in rendered commands")
    p.set_defaults(fn=cmd_server_serve)

    p = sub.add_parser("admin", help="manage users (admin token required)")
    p.add_argument("action", choices=["list", "add", "disable", "enable", "reset-password"])
    p.add_argument("username", nargs="?")
    p.add_argument("--admin", action="store_true", help="with add: create an admin")
    p.add_argument("--server")
    p.set_defaults(fn=cmd_admin)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
