"""MCP tools against a real lab server in-process; GPU discovery and process spawning are faked."""

import json
import os

import anyio
import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from mcp import Client

from catalogs import raw_listing
from effeval.mcp_server import LocalRunner, build_server, parse_free_gpus
from effeval.server.app import create_app, init_admin
from effeval.store.http import HttpClient

RTN_SPEC = {
    "field": "quantization",
    "model": {"name": "qwen3-0.6b"},
    "method": {"name": "rtn", "preset": "gptq_paper"},
    "tasks": [{"name": "native/wikitext2_ppl"}],
}

GPTQ_SPEC = {**RTN_SPEC, "method": {"name": "gptq"}}


class FakeProcess:
    pid = os.getpid()


class Spawner:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeProcess()


@pytest.fixture
def lab(tmp_path):
    """Authenticated client of an in-process lab server with the fixture lm_eval and evalscope catalogs."""
    init_admin(tmp_path / "data", "admin", "admin-password")
    app = create_app(tmp_path / "data")
    token = (
        TestClient(app)
        .post("/api/auth/token", json={"username": "admin", "password": "admin-password", "label": "mcp"})
        .json()["token"]
    )
    http = TestClient(app, headers={"Authorization": f"Bearer {token}"})
    for evaluator in ("lm_eval", "evalscope"):
        assert http.post("/api/task-catalogs", json=raw_listing(evaluator)).status_code == 200
    return http


@pytest.fixture
def mcp(tmp_path, lab):
    http = lab
    free = {"gpus": [2, 3]}
    spawner = Spawner()
    runner = LocalRunner(
        tmp_path / "runs", free_gpus=lambda busy: [g for g in free["gpus"] if g not in busy], spawn=spawner
    )
    server = build_server(HttpClient("http://testserver", None, http=http), runner)

    def call(tool, **arguments):
        async def go():
            async with Client(server) as client:
                return await client.call_tool(tool, arguments)

        return anyio.run(go)

    return call, runner, spawner, free


def test_free_gpus_skip_busy_occupied_and_used_cards():
    gpus = "0, GPU-a, 20000\n1, GPU-b, 11\n2, GPU-c, 12\n3, GPU-d, 900\n"
    apps = "GPU-c\n"
    assert parse_free_gpus(gpus, apps, busy={3}) == [1]


def test_run_spec_runs_the_rendered_command_on_a_free_gpu_and_reports_status(mcp, tmp_path):
    call, _, spawner, _ = mcp
    issues = call("validate_spec", spec=RTN_SPEC).structured_content["result"]
    assert not [i for i in issues if i["severity"] == "error"]

    started = call("run_spec", spec=RTN_SPEC).structured_content
    assert started["started"] is True and started["gpu"] == 2 and started["jobs"] >= 1
    plan_id = started["plan_id"]
    script = (tmp_path / "runs" / plan_id / "run.sh").read_text()
    assert (
        f"effeval-{plan_id[5:17]}.yaml" in script and "effeval run --config" in script and "--yes" in script
    )
    ((_, kwargs),) = spawner.calls
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2" and kwargs["cwd"] == tmp_path / "runs" / plan_id

    status = call("run_status", plan_id=plan_id).structured_content
    assert status["state"] == "running" and status["records"] == {}
    second = call(
        "run_spec", spec={**RTN_SPEC, "tasks": [{"name": "native/wikitext2_ppl", "params": {"limit": 8}}]}
    )
    assert second.structured_content["gpu"] == 3

    (tmp_path / "runs" / plan_id / "exit_code").write_text("0\n")
    assert call("run_status", plan_id=plan_id).structured_content["state"] == "finished"
    assert [r["plan_id"] for r in call("list_runs").structured_content["result"]][-1] == plan_id


def test_specs_with_errors_and_full_machines_start_nothing(mcp):
    call, _, spawner, free = mcp
    bad = call("run_spec", spec={**RTN_SPEC, "model": {"name": "no-such-model"}}).structured_content
    assert bad["started"] is False and any(i["severity"] == "error" for i in bad["issues"])
    free["gpus"] = []
    busy = call("run_spec", spec=RTN_SPEC)
    assert busy.is_error and "no free GPU" in busy.content[0].text
    assert spawner.calls == []


def test_tables_are_created_and_their_pending_cells_run(mcp):
    call, _, spawner, _ = mcp
    call("create_project", name="paper-a")
    table = call(
        "save_table",
        project="paper-a",
        table={
            "name": "W4 RTN",
            "field": "quantization",
            "models": ["qwen3-0.6b"],
            "tasks": ["native/wikitext2_ppl"],
            "rows": [{"label": "RTN W4", "method": {"name": "rtn", "preset": "gptq_paper"}}],
        },
    ).structured_content
    assert table["counts"] == {"pending": table["counts"]["pending"]} and table["counts"]["pending"] >= 1
    listed = call("list_tables", project="paper-a").structured_content["result"]
    assert [t["table_id"] for t in listed] == [table["table_id"]]

    started = call("run_table", project="paper-a", table_id=table["table_id"]).structured_content
    assert started["started"] is True and len(spawner.calls) == 1
    assert "--to project:paper-a" in (spawner.calls[0][1]["cwd"] / "run.sh").read_text()


def search(call, **arguments):
    return call("search_tasks", **arguments).structured_content


def test_describe_leaves_out_the_task_list_but_keeps_the_common_tasks_and_counts(mcp, lab):
    call, *_ = mcp
    brief = call("describe").structured_content
    full = lab.get("/api/describe").json()
    assert "tasks" not in brief
    assert {"methods", "presets", "models", "fields", "evaluators", "gpus"} <= brief.keys()
    assert [t["name"] for t in brief["common_tasks"]] == full["common_tasks"]
    assert "lm_eval/mmlu" in full["common_tasks"] and "native/wikitext2_ppl" in full["common_tasks"]
    assert brief["task_counts"] == {
        e: sum(t["evaluator"] == e for t in full["tasks"].values()) for e in full["evaluators"]
    }
    assert brief["task_counts"]["lm_eval"] == 8 and brief["task_counts"]["evalscope"] == 3
    assert "lm_eval/mmlu_anatomy" not in json.dumps(brief)
    titles = {"base": "Base", "rtn": "RTN", "gptq": "GPTQ"}
    assert {n: m["title"] for n, m in brief["methods"].items()} == titles
    assert {n: m["title"] for n, m in full["methods"].items()} == titles
    assert len(json.dumps(brief)) < len(json.dumps(full)) - len(json.dumps(full["tasks"])) // 2


def test_search_tasks_matches_name_description_and_tags_case_insensitively(mcp):
    call, *_ = mcp
    by_name = search(call, query="ANATOMY")
    assert [t["name"] for t in by_name["tasks"]] == ["lm_eval/mmlu_anatomy", "evalscope/mmlu:anatomy"]
    assert by_name["total"] == 2
    by_tag = search(call, query="math_word_problems")
    assert {t["name"] for t in by_tag["tasks"]} == {"lm_eval/gsm8k", "lm_eval/aime24"}
    by_description = search(call, query="multiple choice", evaluator="lm_eval")
    assert by_description["total"] >= 1 and all(t["evaluator"] == "lm_eval" for t in by_description["tasks"])
    assert search(call, query="no-such-task-anywhere") == {"total": 0, "returned": 0, "tasks": []}


def test_search_tasks_filters_by_evaluator_and_respects_limit(mcp):
    call, *_ = mcp
    both = search(call, query="mmlu")
    only = search(call, query="mmlu", evaluator="evalscope")
    assert {t["evaluator"] for t in both["tasks"]} == {"lm_eval", "evalscope"}
    assert {t["evaluator"] for t in only["tasks"]} == {"evalscope"} and only["total"] < both["total"]
    capped = search(call, query="mmlu", limit=2)
    assert capped["total"] == both["total"] and capped["returned"] == len(capped["tasks"]) == 2
    everything = search(call, limit=5)
    assert everything["total"] > 5 and len(everything["tasks"]) == 5
    assert call("search_tasks", evaluator="nope").is_error
    assert call("search_tasks", limit=0).is_error


def test_search_tasks_filters_by_category_and_mode(mcp):
    call, *_ = mcp
    math = search(call, category="math", limit=200)
    assert "lm_eval/gsm8k" in {t["name"] for t in math["tasks"]}
    assert {t["category"] for t in math["tasks"]} == {"math"}
    generation = search(call, mode="generation", limit=200)
    assert generation["tasks"] and all("generation" in t["modes"] for t in generation["tasks"])
    assert generation["total"] < search(call, limit=1)["total"]
    assert call("search_tasks", category="nope").is_error
    assert call("search_tasks", mode="nope").is_error


def test_search_tasks_ranks_aliases_and_words_and_combines_with_filters(mcp):
    call, *_ = mcp
    ppl = search(call, query="ppl")
    assert ppl["tasks"][0]["name"] == "native/wikitext2_ppl"
    assert "lm_eval/wikitext" in [t["name"] for t in ppl["tasks"]]
    assert search(call, query="ppl", evaluator="evalscope")["total"] == 0
    in_math = search(call, query="gsm8k", category="math", mode="generation")
    assert in_math["tasks"][0]["name"] == "lm_eval/gsm8k"
    assert search(call, query="gsm8k", category="language_modeling")["total"] == 0
    by_title = search(call, query="knowledge", evaluator="lm_eval")
    assert by_title["total"] >= 1 and {t["category"] for t in by_title["tasks"]} == {"qa"}


def test_search_results_are_compact_entries(mcp):
    call, *_ = mcp
    (gsm8k,) = search(call, query="gsm8k", evaluator="lm_eval")["tasks"]
    assert gsm8k["name"] == "lm_eval/gsm8k" and gsm8k["num_fewshot"] == 5 and gsm8k["is_group"] is False
    assert gsm8k["modes"] and isinstance(gsm8k["higher_is_better"], (bool, type(None)))
    (mmlu,) = search(call, query="mmlu", evaluator="lm_eval", limit=1)["tasks"]
    assert mmlu["is_group"] is True and mmlu["subtasks"] >= 1
    (evalscope_mmlu,) = search(call, query="mmlu:anatomy", evaluator="evalscope")["tasks"]
    assert len(evalscope_mmlu["description"]) <= 160
    assert "\n" not in evalscope_mmlu["description"]


def test_get_task_returns_the_full_entry_and_rejects_unknown_names(mcp):
    call, *_ = mcp
    task = call("get_task", name="lm_eval/mmlu").structured_content
    assert task["name"] == "lm_eval/mmlu" and task["subtasks"] and task["metrics"][0]["name"]
    assert {"modes", "params_schema", "fewshot_allowed", "num_fewshot"} <= task.keys()
    assert {"num_fewshot", "limit"} <= task["params_schema"]["properties"].keys()
    missing = call("get_task", name="lm_eval/mmlu_anatom")
    assert missing.is_error and "no task 'lm_eval/mmlu_anatom'" in missing.content[0].text
    assert "lm_eval/mmlu_anatomy" in missing.content[0].text
    assert call("get_task", name="zzz").is_error


def test_repeats_are_accepted_by_run_spec_and_save_table_and_described_in_the_guide(mcp):
    call, _, spawner, _ = mcp
    from effeval.mcp_server import GUIDE

    assert '"repeats"' in GUIDE and '"seeds": [3, 5, 7]' in GUIDE and "vary" not in GUIDE
    spec = {**RTN_SPEC, "repeats": {"seeds": [3, 5, 7]}}
    assert call("run_spec", spec=spec).structured_content["started"] is True
    call("create_project", name="paper-a")
    table = call(
        "save_table",
        project="paper-a",
        table={
            "name": "repeats",
            "field": "quantization",
            "models": ["qwen3-0.6b"],
            "tasks": ["native/wikitext2_ppl"],
            "rows": [{"label": "RTN W4", "method": {"name": "rtn", "preset": "gptq_paper"}}],
            "shared": {"repeats": {"seeds": [4, 6]}},
        },
    ).structured_content
    row = next(r for r in table["blocks"][0]["rows"] if r["label"] == "RTN W4")
    assert len(row["cells"]["native/wikitext2_ppl"]["repeats"]) == 2
    spawner.calls.clear()
    started = call("run_table", project="paper-a", table_id=table["table_id"]).structured_content
    assert started["started"] is True and started["jobs"] % 2 == 0 and started["jobs"] >= 4
    assert "--to project:paper-a" in (spawner.calls[0][1]["cwd"] / "run.sh").read_text()


def test_describe_lists_preprocess_compactly_and_get_architecture_gives_graph_and_positions(mcp, lab):
    call, *_ = mcp
    brief = call("describe").structured_content
    full = lab.get("/api/describe").json()
    quarot = brief["preprocess_presets"]["quarot"]
    assert quarot["title"] == "QuaRot" and quarot["paper"].startswith("QuaRot")
    assert {"position": "residual", "mode": "offline", "source": "random_hadamard"} in quarot["steps"][0][
        "entries"
    ]
    assert brief["preprocess_sources"]["rotation"]["trained"]["implemented"] is False
    assert brief["preprocess_sources"]["scaling"]["awq"]["options"]["n_grid"] == 20
    assert "schema" not in brief["preprocess_sources"]["scaling"]["awq"]
    smooth = brief["preprocess_sources"]["scaling"]["smoothattention"]
    assert smooth["positions"] == ["attn.qk_pre_rope", "attn.qk_rope", "attn.v_head"]
    assert smooth["no_paper"] == ["attn.qk_rope", "attn.v_head"]
    assert smooth["options"]["granularity"] == "per_head"
    assert brief["preprocess_sources"]["scaling"]["awq"]["no_paper"] == []
    assert brief["preprocess_sources"]["rotation"]["hadamard"]["positions"] is None
    assert all(a["graph"] is None for a in brief["architectures"].values())
    assert all("positions" not in m for m in brief["models"].values())
    assert any(a["graph"] for a in full["architectures"].values())
    assert len(json.dumps(brief)) < len(json.dumps(full)) - len(json.dumps(full["tasks"])) // 2

    arch = call("get_architecture", name="qwen3-0.6b").structured_content
    assert arch["model"] == "qwen3-0.6b" and arch["shape"]["head_dim"] == 128
    assert {"q", "down"} <= {n["id"] for n in arch["graph"]["nodes"]}
    assert arch["positions"]["residual"]["rotation"]["offline"] == "ok"
    by_class = call("get_architecture", name=arch["architecture"]).structured_content
    assert by_class["graph"] == arch["graph"] and by_class["positions"] is None
    missing = call("get_architecture", name="nope")
    assert missing.is_error and "no model or architecture 'nope'" in missing.content[0].text


def test_guide_explains_preprocessing_with_a_quarot_gptq_example():
    from effeval.mcp_server import GUIDE

    assert "模型前处理" in GUIDE and "get_architecture" in GUIDE and "QuaRot + GPTQ W4A4KV4" in GUIDE
    assert all(word in GUIDE for word in ("position", "offline", "online", "source"))


def test_run_spec_and_validate_spec_accept_a_preprocess_preset_name_or_steps(mcp):
    call, _, spawner, _ = mcp
    spec = {**GPTQ_SPEC, "preprocess": "quarot"}
    issues = call("validate_spec", spec=spec).structured_content["result"]
    assert not [i for i in issues if i["severity"] == "error"]
    unknown = call("validate_spec", spec={**GPTQ_SPEC, "preprocess": "no-such"}).structured_content["result"]
    assert "preprocess.preset_unknown" in {i["code"] for i in unknown}
    started = call("run_spec", spec=spec).structured_content
    assert started["started"] is True
    script = (spawner.calls[0][1]["cwd"] / "run.sh").read_text()
    assert "preprocess:" in script and "preset: quarot" in script and "random_hadamard" in script
    as_dict = call("run_spec", spec={**GPTQ_SPEC, "preprocess": {"preset": "quarot"}}).structured_content
    assert as_dict["started"] is True and as_dict["jobs"] == started["jobs"]


def test_save_table_with_a_quarot_gptq_row_gives_a_pending_cell_with_its_label(mcp):
    call, _, spawner, _ = mcp
    call("create_project", name="paper-a")
    table = call(
        "save_table",
        project="paper-a",
        table={
            "name": "QuaRot",
            "field": "quantization",
            "models": ["qwen3-0.6b"],
            "tasks": ["native/wikitext2_ppl"],
            "rows": [
                {"method": {"name": "gptq"}, "preprocess": "quarot"},
                {"label": "plain GPTQ", "method": {"name": "gptq"}},
            ],
        },
    ).structured_content
    rows = {r["key"]: r for r in table["blocks"][0]["rows"]}
    cell = rows["row0"]["cells"]["native/wikitext2_ppl"]
    assert rows["row0"]["label"] == rows["row0"]["title"] == "QuaRot + GPTQ"
    assert cell["status"] == "pending"
    assert rows["row1"]["title"] == "GPTQ" and rows["row1"]["preprocess"] is None
    assert cell["run_key"] != rows["row1"]["cells"]["native/wikitext2_ppl"]["run_key"]

    started = call("run_table", project="paper-a", table_id=table["table_id"]).structured_content
    assert started["started"] is True
    assert "preset: quarot" in (spawner.calls[0][1]["cwd"] / "run.sh").read_text()


def test_guide_explains_speculative_decoding_and_describe_lists_the_runtimes(mcp):
    call, *_ = mcp
    from effeval.mcp_server import GUIDE

    words = ("speculative_decoding", "num_speculative_tokens", "runtimes", "vllm", "sglang")
    assert all(w in GUIDE for w in words)
    runtimes = call("describe").structured_content["runtimes"]
    assert list(runtimes) == ["hf", "vllm", "sglang"] and runtimes["hf"]["available"] is True


def test_save_table_takes_a_decoding_per_row_and_exposes_draft_and_k_as_columns(mcp, monkeypatch):
    from effeval.core.types import Artifact
    from effeval.metrics.quality import COMPARATIVE, Comparative

    for name in ("speedup", "output_match"):
        monkeypatch.setitem(
            COMPARATIVE,
            name,
            Comparative(name, frozenset({Artifact.PER_SAMPLE}), frozenset(), "stand-in", lambda *_: {}),
        )
    call, *_ = mcp
    call("create_project", name="spec")
    decoding = {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 2}
    table = call(
        "save_table",
        project="spec",
        table={
            "name": "draft sweep",
            "field": "speculative_decoding",
            "models": ["qwen3-8b"],
            "tasks": ["lm_eval/gsm8k"],
            "rows": [{"label": "0.6B draft", "method": {"name": "base"}, "decoding": decoding}],
        },
    ).structured_content
    rows = {r["key"]: r for r in table["blocks"][0]["rows"]}
    assert rows["row0"]["draft"] == "qwen3-0.6b" and rows["row0"]["decoding"]["k"] == 2
    assert rows["row0"]["label"] == "0.6B draft" and rows["base0"]["draft"] is None
