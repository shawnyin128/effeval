"""Stand-in for the engine worker (stdlib only): same CLI and protocol, behaviour set by config["fake"].

Modes: ok, error_before_ready, late_error_before_ready (the error line, then a log line after a pause),
wrong_library, crash_on_request, never_ready, ignore_exit, ignore_term (ignores SIGTERM and exit commands),
cache_hit, garbage, stats_always (stats even without a draft), stats_never (no stats even with a draft),
error_on_request, wrong_id, hang_on_request, truncate (fewer tokens than asked, no stop id) and scripted
(every reply is config["scripted"]: output_ids and stats).
An `ok` reply has up to three tokens and ends with config["stop_ids"][0] when the budget exceeds three.
Every process it starts appends "<role> <pid>" to pids.txt in its working directory.
"""

import json
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    with open(sys.argv[sys.argv.index("--config") + 1]) as f:
        config = json.load(f)
    mode = config.get("fake", "ok")
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open("pids.txt", "a") as pids:
        pids.write(f"worker {os.getpid()}\n")
        if mode in ("ignore_exit", "ignore_term"):
            grandchild = subprocess.Popen(["sleep", "600"])
            pids.write(f"grandchild {grandchild.pid}\n")

    def send(message):
        print(json.dumps(message), flush=True)

    print(f"fake engine worker started in mode {mode}", file=sys.stderr, flush=True)
    if mode == "garbage":
        print("this is not json", flush=True)
        time.sleep(600)
    if mode == "error_before_ready":
        send({"error": "fake startup failure"})
        return 1
    if mode == "late_error_before_ready":
        send({"error": "fake startup failure"})
        time.sleep(0.5)
        print("late-log-marker", file=sys.stderr, flush=True)
        return 1
    if mode == "never_ready":
        time.sleep(600)
    library = config["library"].split("+")[0]
    if mode == "wrong_library":
        library = f"{config['engine']}==0.31.0"
    send(
        {
            "ready": True,
            "library": library + "+fake",
            "torch": "0.0.0+fake",
            "cuda": "0.0",
            "engine_args": {},
            "resolved": "fake",
            "env_keys": sorted(os.environ),
        }
    )
    draft = config.get("draft")
    k = config.get("num_speculative_tokens")
    stop = (config.get("stop_ids") or [0])[0]
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("cmd") == "exit":
            if mode in ("ignore_exit", "ignore_term"):
                continue
            send({"bye": True})
            return 0
        if mode == "crash_on_request":
            print("boom-marker", file=sys.stderr, flush=True)
            return 7
        if mode == "hang_on_request":
            time.sleep(600)
        if mode == "error_on_request":
            send({"id": request["id"], "error": "RuntimeError: fake request failure"})
            continue
        wants_stats = (draft is not None and mode != "stats_never") or mode == "stats_always"
        stats = {"rounds": 1, "proposed": k or 4, "accepted": (k or 4) - 1} if wants_stats else None
        budget = request["max_new_tokens"]
        token = request["prompt_ids"][-1] % 50 + 1
        output = [token] * min(3, budget)
        if budget > 3:
            output[-1] = stop
        if mode == "truncate":
            output = [token] * (budget - 1)
        if mode == "scripted":
            output, stats = config["scripted"]["output_ids"], config["scripted"]["stats"]
        send(
            {
                "id": request["id"] + (100 if mode == "wrong_id" else 0),
                "output_ids": output,
                "seconds": 0.25,
                "stats": stats,
                "cache_hits": 5 if mode == "cache_hit" else 0,
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
