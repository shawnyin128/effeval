"""Inference engines other than transformers: their pinned libraries and where their Python lives."""

from __future__ import annotations

from effeval.core.types import StrictModel


class EngineDef(StrictModel):
    library: str
    python_env: str


ENGINES: dict[str, EngineDef] = {
    "vllm": EngineDef(library="vllm==0.32.0", python_env="EFFEVAL_VLLM_PYTHON"),
    "sglang": EngineDef(library="sglang==0.5.12", python_env="EFFEVAL_SGLANG_PYTHON"),
}
"""Runtime name -> the library version jobs are frozen with and the environment variable that names the
Python executable of that engine's own virtual environment."""
