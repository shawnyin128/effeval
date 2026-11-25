import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from effeval.core.fragments import TextBatches  # noqa: E402
from effeval.ops import calibration as calibration_module  # noqa: E402
from effeval.ops.calibration import reasoning_mixture, text_batches  # noqa: E402

pytestmark = pytest.mark.compute

TEXTS = [
    "alpha",
    "a considerably longer second row with many more words in it than the others",
    "gamma gamma",
    "delta delta delta delta delta delta",
    "epsilon",
    "zeta zeta zeta",
    "eta eta eta eta",
    "theta",
]


class ListTokenizer:
    """Chat template as `<user>text<assistant>`, one id per character, padding on the right."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        suffix = "" if "enable_thinking" not in kwargs else f"<think={kwargs['enable_thinking']}>"
        return f"<user>{messages[0]['content']}<assistant>{suffix}"

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        assert truncation and return_tensors == "pt"
        ids = [[ord(c) % 90 + 1 for c in text][:max_length] for text in texts]
        width = max_length if padding == "max_length" else max(map(len, ids))
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in ids])
        attention_mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _config(**overrides):
    values = dict(dataset="fake/texts", n_samples=7, batch_size=3, max_length=40, epochs=3, shuffle_seed=42)
    return TextBatches(**{**values, **overrides})


@pytest.fixture
def fake_texts(monkeypatch):
    monkeypatch.setattr(
        calibration_module,
        "load_dataset",
        lambda name, split: datasets.Dataset.from_dict({"text": TEXTS, "other": list(range(len(TEXTS)))}),
    )


def _official_order(n_samples, batch_size, epochs, seed):
    """Row indices per batch of `DataLoader(shuffle=True, generator=manual_seed(seed))`."""
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __len__(self):
            return n_samples

        def __getitem__(self, index):
            return index

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(Rows(), batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    return [[batch.tolist() for batch in loader] for _ in range(epochs)]


def test_batches_follow_the_dataloader_order_epoch_by_epoch(fake_texts):
    cfg = _config(chat_template=False)
    result = text_batches(cfg, ListTokenizer())
    expected = _official_order(cfg.n_samples, cfg.batch_size, cfg.epochs, cfg.shuffle_seed)
    assert len(result) == 3
    for epoch, order in zip(result, expected, strict=True):
        assert [b["input_ids"].shape[0] for b in epoch] == [len(rows) for rows in order]
        for batch, rows in zip(epoch, order, strict=True):
            reference = ListTokenizer()([TEXTS[i] for i in rows], "pt", True, True, cfg.max_length)
            assert torch.equal(batch["input_ids"], reference["input_ids"])
            assert torch.equal(batch["attention_mask"], reference["attention_mask"])
    assert expected[0] != expected[1] != expected[2]


def test_every_epoch_covers_exactly_the_first_n_samples_rows(fake_texts):
    result = text_batches(_config(chat_template=False), ListTokenizer())
    wanted = sorted(ListTokenizer()([t], "pt", True, True, 40)["input_ids"].sum().item() for t in TEXTS[:7])
    for epoch in result:
        got = sorted(int(row.sum()) for batch in epoch for row in batch["input_ids"])
        assert got == wanted
        assert [b["input_ids"].shape[0] for b in epoch] == [3, 3, 1]


def test_chat_template_wraps_each_row_and_forwards_enable_thinking(fake_texts):
    plain = text_batches(_config(epochs=1, chat_template=False, max_length=200), ListTokenizer())[0]
    wrapped = text_batches(_config(epochs=1, max_length=200), ListTokenizer())[0]
    thinking = text_batches(_config(epochs=1, enable_thinking=False, max_length=200), ListTokenizer())[0]
    assert wrapped[0]["input_ids"].shape[1] > plain[0]["input_ids"].shape[1]
    assert thinking[0]["input_ids"].shape[1] > wrapped[0]["input_ids"].shape[1]


def test_padding_max_length_and_truncation(fake_texts):
    cfg = _config(epochs=1, chat_template=False, padding="max_length", max_length=12)
    batch = text_batches(cfg, ListTokenizer())[0][0]
    assert batch["input_ids"].shape[1] == 12
    assert int(batch["attention_mask"].sum(dim=1).max()) == 12


def test_enable_thinking_requires_the_chat_template():
    with pytest.raises(ValueError, match="enable_thinking"):
        _config(chat_template=False, enable_thinking=True)


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer is not in the local Hugging Face cache")


def test_qwen3_batches_equal_the_official_dataloader_and_tokenization(fake_texts, qwen3_tokenizer):
    cfg = _config(n_samples=8, batch_size=3, max_length=24, epochs=2, enable_thinking=False)
    result = text_batches(cfg, qwen3_tokenizer)
    order = _official_order(cfg.n_samples, cfg.batch_size, cfg.epochs, cfg.shuffle_seed)
    assert qwen3_tokenizer.padding_side == "right"
    for epoch, epoch_rows in zip(result, order, strict=True):
        for batch, rows in zip(epoch, epoch_rows, strict=True):
            processed = [
                qwen3_tokenizer.apply_chat_template(
                    [{"role": "user", "content": TEXTS[i]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for i in rows
            ]
            official = qwen3_tokenizer(
                processed, return_tensors="pt", padding=True, truncation=True, max_length=24
            )
            assert torch.equal(batch["input_ids"], official["input_ids"])
            assert torch.equal(batch["attention_mask"], official["attention_mask"])
    padded = [b for epoch in result for b in epoch if int(b["attention_mask"].min()) == 0]
    assert padded, "the fixture must exercise padding"
    for batch in padded:
        mask = batch["attention_mask"]
        assert torch.equal(mask, mask.sort(dim=1, descending=True, stable=True).values)


def _official_module(project: str, relative: str):
    """Import a module of the official CodeQuant sources found under $EFFEVAL_CODEQUANT_REF/<project>."""
    root = os.environ.get("EFFEVAL_CODEQUANT_REF")
    if not root or not (Path(root) / project / relative).is_file():
        pytest.skip("EFFEVAL_CODEQUANT_REF does not point at the official CodeQuant sources")
    project_root = Path(root) / project
    before = set(sys.modules)
    sys.path.insert(0, str(project_root))
    try:
        spec = importlib.util.spec_from_file_location(
            f"official_{project}_{Path(relative).stem}", project_root / relative
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(project_root))
        for name in set(sys.modules) - before:
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
    return module


def test_batch_order_equals_the_official_calibration_dataset_and_dataloader(fake_texts, monkeypatch):
    from torch.utils.data import DataLoader

    official = _official_module("cq1", "utils/dataset_utils.py")
    monkeypatch.setattr(
        official, "load_dataset", lambda name, split: datasets.Dataset.from_dict({"text": TEXTS})
    )
    cfg = _config(n_samples=7, batch_size=3, chat_template=False, epochs=3)
    dataset = official.CalibrationDataset("fake/texts", cfg.n_samples)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=generator)
    result = text_batches(cfg, ListTokenizer())
    for epoch in result:
        for batch, texts in zip(epoch, list(loader), strict=True):
            reference = ListTokenizer()(texts, "pt", True, True, cfg.max_length)
            assert torch.equal(batch["input_ids"], reference["input_ids"])


class _WordTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and enable_thinking is True
        turns = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return turns + ("<assistant>" if add_generation_prompt else "<end>")


def test_reasoning_mixture_equals_official_mixed_dataset_and_formatting():
    official = _official_module("cq2", "utils/reasoning_data_utils.py")
    problems = [f"problem {i}" for i in range(30)]
    wiki = [f"wiki row {i}" for i in range(20)] + ["   ", ""]
    tokenizer = _WordTokenizer()
    prompts_seen = []

    def generate(prompts):
        prompts_seen.extend(prompts)
        return [f"trace of [{p}]" for p in prompts]

    rows = reasoning_mixture(problems, wiki, generate, tokenizer, total=12, wiki_ratio=0.5, seed=7)

    gen_records = [
        {"source": "gsm8k", "text": "", "input": p, "prediction": f"trace of [<user>{p}<assistant>]"}
        for p in problems
    ]
    wiki_rows = official.WikiTextDataset.__new__(official.WikiTextDataset)
    wiki_rows.texts = [t for t in wiki if t.strip()]
    mixed = official.MixedDataset(gen_records, wiki_rows, total=12, wiki_ratio=0.5, seed=7)
    batch = {
        key: [r.get(key, "") for r in mixed.records] for key in ("input", "prediction", "source", "text")
    }
    assert rows == official.format_reasoning_batch(batch, tokenizer)
    assert len(prompts_seen) == 6 and len(set(prompts_seen)) == 6


def test_reasoning_mixture_without_wiki_and_with_too_few_rows():
    tokenizer = _WordTokenizer()
    rows = reasoning_mixture(
        ["a", "b"], ["w"], lambda ps: ["x"] * len(ps), tokenizer, total=2, wiki_ratio=0.0
    )
    assert sorted(rows) == sorted(f"<user>{p}<assistant>x<end>" for p in "ab")
    with pytest.raises(ValueError, match="requested 5 rows"):
        reasoning_mixture(["a", "b"], ["w"], lambda ps: ["x"] * len(ps), tokenizer, total=5, wiki_ratio=0.5)
    with pytest.raises(ValueError, match="completions"):
        reasoning_mixture(["a", "b"], [], lambda ps: ["x"], tokenizer, total=2, wiki_ratio=0.0)


BLANKS = [
    "alpha",
    "",
    "a considerably longer second row",
    "   ",
    "gamma gamma",
    "delta",
    "",
    "epsilon epsilon",
]


@pytest.fixture
def official_wiki(monkeypatch):
    official = _official_module("cq2", "utils/reasoning_data_utils.py")
    monkeypatch.setattr(
        datasets, "load_dataset", lambda name, split: datasets.Dataset.from_dict({"text": BLANKS})
    )
    monkeypatch.setattr(
        calibration_module, "load_dataset", lambda name, split: datasets.Dataset.from_dict({"text": BLANKS})
    )
    return official


def _row_sums(epoch):
    return sorted(int(row.sum()) for batch in epoch for row in batch["input_ids"])


def test_drop_empty_takes_the_first_non_empty_rows_like_the_official_wikitext_dataset(official_wiki):
    expected = official_wiki.WikiTextDataset("fake/blanks", "train", limit=4).texts
    assert len(expected) == 4 and all(t.strip() for t in expected)
    cfg = _config(n_samples=4, batch_size=4, epochs=1, chat_template=False, drop_empty=True)
    (epoch,) = text_batches(cfg, ListTokenizer())
    reference = ListTokenizer()(expected, "pt", True, True, cfg.max_length)
    assert _row_sums(epoch) == sorted(int(row.sum()) for row in reference["input_ids"])


def test_blank_rows_are_kept_unless_drop_empty_is_set(official_wiki):
    cfg = _config(n_samples=4, batch_size=4, epochs=1, chat_template=False)
    (epoch,) = text_batches(cfg, ListTokenizer())
    kept = ListTokenizer()(BLANKS[:4], "pt", True, True, cfg.max_length)
    assert _row_sums(epoch) == sorted(int(row.sum()) for row in kept["input_ids"])


def test_skip_epochs_continues_the_official_dataloader_stream(official_wiki):
    from torch.utils.data import DataLoader

    dataset = official_wiki.WikiTextDataset("fake/blanks", "train", limit=5)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    official = [[[row["text"] for row in batch] for batch in _split(loader)] for _ in range(4)]
    cfg = _config(n_samples=5, batch_size=2, epochs=2, skip_epochs=2, chat_template=False, drop_empty=True)
    result = text_batches(cfg, ListTokenizer())
    for epoch, texts_of_epoch in zip(result, official[2:], strict=True):
        for batch, texts in zip(epoch, texts_of_epoch, strict=True):
            reference = ListTokenizer()(texts, "pt", True, True, cfg.max_length)
            assert torch.equal(batch["input_ids"], reference["input_ids"])
    assert official[0] != official[2]


def _split(loader):
    """The batches of one epoch of an official loader as lists of rows (collated dicts split back)."""
    for batch in loader:
        yield [{"text": text} for text in batch["text"]]


@pytest.mark.parametrize("skip", [0, 2])
def test_pre_shuffled_wiki_only_batches_equal_the_official_mixed_dataset_and_dataloader(official_wiki, skip):
    from torch.utils.data import DataLoader

    wiki = official_wiki.WikiTextDataset("fake/blanks", "train", limit=5)
    mixed = official_wiki.MixedDataset([], wiki, total=5, wiki_ratio=1.0, seed=42)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(mixed, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    official = [[batch["text"] for batch in loader] for _ in range(skip + 2)][skip:]
    cfg = _config(
        n_samples=5,
        batch_size=2,
        epochs=2,
        skip_epochs=skip,
        chat_template=False,
        drop_empty=True,
        pre_shuffle_seed=42,
    )
    result = text_batches(cfg, ListTokenizer())
    for epoch, texts_of_epoch in zip(result, official, strict=True):
        for batch, texts in zip(epoch, texts_of_epoch, strict=True):
            reference = ListTokenizer()(list(texts), "pt", True, True, cfg.max_length)
            assert torch.equal(batch["input_ids"], reference["input_ids"])
    unshuffled = text_batches(cfg.model_copy(update={"pre_shuffle_seed": None}), ListTokenizer())
    flat = [t for epoch in result for b in epoch for t in b["input_ids"].sum(dim=1).tolist()]
    flat_unshuffled = [t for epoch in unshuffled for b in epoch for t in b["input_ids"].sum(dim=1).tolist()]
    assert flat != flat_unshuffled
