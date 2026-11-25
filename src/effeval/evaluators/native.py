"""effeval's own protocols for tasks no third-party framework reproduces exactly.

gptq_ppl: perplexity as in the GPTQ paper §A.2.1 and the official `opt_eval` (github.com/IST-DASLab/gptq):
join the split with the task's joiner, tokenize once, cut non-overlapping windows of `seqlen` tokens, and
ppl = exp(sum(window_loss * seqlen) / (nsamples * seqlen)). The official code multiplies each window's mean
loss by seqlen (not seqlen - 1); that is kept so numbers match the paper.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer

import effeval
from effeval.core.catalog import TaskDef, sample_count
from effeval.core.spec import JobSpec
from effeval.core.types import Artifact, InProcess, QualityResult, Sample, metric_str


class Native:
    name = "native"

    def list_tasks(self) -> None:
        return None

    @torch.no_grad()
    def run(self, handle: InProcess, job: JobSpec, task: TaskDef, workdir: Path) -> QualityResult:
        return _run(handle, job, task)


def _run(handle: InProcess, job: JobSpec, task: TaskDef) -> QualityResult:
    if task.native is None:
        raise ValueError(f"task {task.name!r} declares no native protocol")
    native = task.native
    tokenizer = AutoTokenizer.from_pretrained(
        job.model.checkpoint, revision=job.model.revision, use_fast=native.use_fast_tokenizer
    )
    data_files = {native.split: native.data_files} if native.data_files else None
    data = load_dataset(native.dataset, native.dataset_config, data_files=data_files, split=native.split)
    if native.max_documents is not None:
        data = data.select(range(min(native.max_documents, len(data))))
    ids = tokenizer(native.joiner.join(data[native.text_column]), return_tensors="pt").input_ids  # [1, N]
    seqlen = task.settings(job.task.params).seqlen
    if native.max_tokens is not None:
        ids = ids[:, : native.max_tokens]
    nsamples = ids.numel() // seqlen
    nsamples = sample_count(job.task.limit, nsamples)
    device = handle.model.get_input_embeddings().weight.device
    loss_fn = nn.CrossEntropyLoss()
    nlls = []
    samples = []
    for i in range(nsamples):
        batch = ids[:, i * seqlen : (i + 1) * seqlen].to(device)  # [1, L]
        logits = handle.model(batch, use_cache=False).logits.float()  # [1, L, V]; float32 for the loss
        shift_logits = logits[:, :-1, :]  # [1, L-1, V]
        shift_labels = batch[:, 1:]  # [1, L-1]
        loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * seqlen)
        samples.append(Sample(id=f"window:{i}", scores={"nll": float(loss.float())}))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return QualityResult(
        metrics={task.metrics[0].name: metric_str(float(ppl))},
        samples=samples,
        protocol={
            "backend": "native",
            "backend_version": effeval.__version__,
            "protocol": "gptq_ppl",
            "dataset": f"{native.dataset}/{native.dataset_config}:{native.split}",
            "seqlen": seqlen,
            "windows": nsamples,
            "tokens": int(ids.numel()),
            "use_fast_tokenizer": native.use_fast_tokenizer,
            **{
                key: value
                for key, value in (
                    ("data_files", native.data_files),
                    ("max_documents", native.max_documents),
                    ("max_tokens", native.max_tokens),
                )
                if value is not None
            },
        },
        produces=frozenset({Artifact.PER_SAMPLE}),
    )


EVALUATOR = Native()
