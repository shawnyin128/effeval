"""Shared configuration fragments that methods compose instead of redefining (architecture.md §5.5).

A method manifest lists the fragments it uses in `uses`; the server validates those properties with the
models below, so baselines and custom methods share one vocabulary for sites, layers and number formats.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from effeval.core.types import StrictModel

Granularity = Literal["per_tensor", "per_channel", "per_token", "per_group"]
GRANULARITY_HELP = (
    "Scale unit: per_tensor one scale, per_channel one per output row (weights), "
    "per_token one per vector (activations), per_group every group_size values."
)

GroupAxis = Annotated[
    Literal["channel", "token"] | None,
    Field(
        description="Axis a per_group scale unit spans. channel (default): group_size consecutive features "
        "of one token. token: group_size consecutive tokens of one feature. Sequence-shaped sites only."
    ),
]
Residual = Annotated[
    int | None,
    Field(
        ge=1,
        description="Keep the most recent residual tokens in full precision; only older tokens are "
        "quantized. With group_axis token the quantized span is cut to whole groups. Sequence-shaped sites "
        "only.",
    ),
]

_ENCODING = re.compile(r"^e(\d)m(\d)$")
_LAYER_RANGE = re.compile(r"^(\d+)-(\d+)$")


def parse_encoding(encoding: str) -> tuple[int, int]:
    """Split an `eXmY` encoding into (exponent_bits, mantissa_bits). Raises ValueError if malformed."""
    match = _ENCODING.match(encoding)
    if match is None:
        raise ValueError(f"encoding must look like 'e4m3', got {encoding!r}")
    exponent_bits, mantissa_bits = int(match.group(1)), int(match.group(2))
    if exponent_bits < 1 or exponent_bits + mantissa_bits > 15:
        raise ValueError(f"unsupported encoding {encoding!r}: need 1 <= E and E + M <= 15")
    return exponent_bits, mantissa_bits


class ClipSearch(StrictModel):
    """Per-unit search for the clipping ratio that minimises the weight reconstruction error.

    Candidates are p = 1 - i / grid for i in range(int(max_shrink * grid)); each shrinks the unit's range to
    p of its min-max range, and the candidate with the smallest sum of |dequantized - original|^norm wins
    (the first one on ties). This is the `--w_clip` search of QuaRot (quant_utils.py `WeightQuantizer`) and
    of the GPTQ repository's `mse` option.
    """

    type: Literal["mse"]
    grid: int = Field(default=100, ge=1, description="Resolution of the candidate ratios.")
    max_shrink: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Largest shrink; candidates stop before 1 - max_shrink."
    )
    norm: float = Field(default=2.4, gt=0.0, description="Exponent of the error summed over a unit.")


class IntFormat(StrictModel):
    """Uniform integer grid.

    Asymmetric: min-max range that always contains zero, zero point `round(-min / scale)`; this is exactly the
    grid of the official GPTQ code (quant.py `Quantizer.find_params`) and the paper's RTN baseline.
    Symmetric: `scale = absmax / (2^(bits-1) - 1)`, codes clamped to [-2^(bits-1), 2^(bits-1) - 1].
    """

    type: Literal["int"]
    bits: int = Field(ge=2, le=8, description="Integer bit width.")
    symmetric: bool = Field(
        description="Symmetric: scale = absmax / (2^(bits-1) - 1). Asymmetric: min-max grid, zero point."
    )
    granularity: Granularity = Field(description=GRANULARITY_HELP)
    group_size: int | None = Field(
        default=None, ge=1, description="Values per scale when granularity is per_group."
    )
    clip_ratio: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Shrink the range to this fraction of absmax before rounding.",
    )
    clip_search: ClipSearch | None = Field(
        default=None,
        description="Weights only: search each scale unit's clipping ratio by reconstruction error "
        "instead of using clip_ratio.",
    )
    zero_point: Literal["integer", "float"] | None = Field(
        default=None,
        description="Asymmetric only. integer (default): the range always contains zero and the zero point "
        "is an integer. float: the grid spans exactly [min, max] of the unit, offset by min.",
    )
    group_axis: GroupAxis = None
    residual: Residual = None

    @model_validator(mode="after")
    def _group_size_matches_granularity(self) -> IntFormat:
        _check_group_size(self.granularity, self.group_size)
        _check_sequence_options(self.granularity, self.group_axis)
        if self.zero_point is not None and self.symmetric:
            raise ValueError("zero_point applies to asymmetric grids only")
        if self.zero_point == "float" and (self.clip_ratio != 1.0 or self.clip_search is not None):
            raise ValueError("zero_point float spans the exact min-max range: leave clip_ratio, clip_search")
        if self.clip_search is not None and self.clip_ratio != 1.0:
            raise ValueError("clip_search replaces clip_ratio; leave clip_ratio at 1.0")
        return self


class FpFormat(StrictModel):
    """Low-precision float `eXmY` with a float32 scale `absmax / max_representable` per granularity unit.

    `special` selects which codes are reserved: `ieee` (inf/nan, e.g. e5m2), `fn` (only the all-ones code is
    NaN, e.g. e4m3fn), `none` (all codes finite, e.g. OCP FP6/FP4). `auto` picks ieee for e5m2, fn for e4m3
    and none otherwise.
    """

    type: Literal["fp"]
    encoding: str = Field(description="eXmY: exponent and mantissa bits, e.g. e4m3, e2m1.")
    special: Literal["auto", "ieee", "fn", "none"] = Field(
        default="auto", description="Reserved codes: ieee inf/NaN, fn NaN only, none all finite."
    )
    granularity: Granularity = Field(description=GRANULARITY_HELP)
    group_size: int | None = Field(
        default=None, ge=1, description="Values per scale when granularity is per_group."
    )
    clip_ratio: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Shrink the range to this fraction of absmax before rounding.",
    )

    @field_validator("encoding")
    @classmethod
    def _valid_encoding(cls, value: str) -> str:
        parse_encoding(value)
        return value

    group_axis: GroupAxis = None
    residual: Residual = None

    @model_validator(mode="after")
    def _group_size_matches_granularity(self) -> FpFormat:
        _check_group_size(self.granularity, self.group_size)
        _check_sequence_options(self.granularity, self.group_axis)
        return self


class MxFormat(StrictModel):
    """Block-scaled format along the last dimension.

    `scale: e8m0` follows the OCP Microscaling rule: shared scale `2^(floor(log2(block_absmax)) - emax_elem)`.
    `scale: e4m3` follows the two-level NVFP4-style rule: a float32 tensor scale plus an e4m3 block scale.
    """

    type: Literal["mx"]
    element: str = Field(description="Element format eXmY, e.g. e2m1 for MXFP4.")
    block_size: int = Field(ge=2, description="Values sharing one scale.")
    scale: Literal["e8m0", "e4m3"] = Field(
        description="e8m0: OCP power-of-two scale. e4m3: NVFP4-style two-level scale."
    )

    @field_validator("element")
    @classmethod
    def _valid_element(cls, value: str) -> str:
        parse_encoding(value)
        return value


class KMeansConfig(StrictModel):
    """Lloyd iterations of a codebook fit, started from k-means++ draws of a generator seeded with `seed`."""

    iters: int = Field(default=300, ge=0, description="Maximum Lloyd iterations.")
    tol: float = Field(
        default=1e-6, ge=0.0, description="Stop once no centroid moves by more than this (absolute)."
    )
    seed: int = Field(default=42, description="Seed of the k-means++ generator, created afresh per matrix.")
    compute_dtype: Literal["weight", "float32"] = Field(
        default="weight", description="Dtype the fit runs in: the weight's own dtype or float32."
    )


class CodebookFormat(StrictModel):
    """Weights only: each unit (a row, or group_size consecutive values of a row) keeps `centroids` values
    fitted by k-means, and every element is replaced by the nearest centroid of its unit."""

    type: Literal["codebook"]
    centroids: int = Field(ge=1, le=256, description="Codebook entries per unit.")
    granularity: Literal["per_channel", "per_group"] = Field(
        description="Unit: per_channel one codebook per output row, per_group one per group_size values."
    )
    group_size: int | None = Field(
        default=None, ge=1, description="Values per codebook when granularity is per_group."
    )
    kmeans: KMeansConfig = Field(default_factory=KMeansConfig)

    @model_validator(mode="after")
    def _group_size_matches_granularity(self) -> CodebookFormat:
        _check_group_size(self.granularity, self.group_size)
        return self


QuantFormat = Annotated[IntFormat | FpFormat | MxFormat | CodebookFormat, Field(discriminator="type")]

LayerSelector = int | str | list[int | str]


def _check_group_size(granularity: str, group_size: int | None) -> None:
    if granularity == "per_group" and group_size is None:
        raise ValueError("granularity 'per_group' requires group_size")
    if granularity != "per_group" and group_size is not None:
        raise ValueError(f"group_size is only valid with granularity 'per_group', not {granularity!r}")


def _check_sequence_options(granularity: str, group_axis: str | None) -> None:
    if group_axis is not None and granularity != "per_group":
        raise ValueError("group_axis is only valid with granularity 'per_group'")


def sequence_options(fmt: object) -> list[str]:
    """Names of the sequence-shaped options set on a quant format (empty for formats without them)."""
    return [name for name in ("group_axis", "residual") if getattr(fmt, name, None) is not None]


def check_layer_selector(selector: LayerSelector) -> None:
    """Validate a layer selector without knowing the model depth. Raises ValueError when malformed."""
    items = selector if isinstance(selector, list) else [selector]
    if not items:
        raise ValueError("layer selector list must not be empty")
    for item in items:
        if isinstance(item, int):
            if item < 0:
                raise ValueError(f"layer index must be >= 0, got {item}")
        elif item == "all":
            if isinstance(selector, list):
                raise ValueError("'all' cannot be combined with other layer selectors")
        elif item.startswith("re:"):
            re.compile(item[3:])
        elif _LAYER_RANGE.match(item):
            start, end = (int(x) for x in _LAYER_RANGE.match(item).groups())
            if start > end:
                raise ValueError(f"layer range {item!r} is empty")
        elif item.isdigit():
            pass
        else:
            raise ValueError(
                f"unrecognized layer selector {item!r}; use an index, 'a-b', 're:<regex>' or 'all'"
            )


def select_layers(selector: LayerSelector, num_layers: int) -> list[int]:
    """Resolve a selector to sorted layer indices in [0, num_layers).

    Raises ValueError for malformed selectors or out-of-range indices.
    """
    check_layer_selector(selector)
    items = selector if isinstance(selector, list) else [selector]
    chosen: set[int] = set()
    for item in items:
        if isinstance(item, int):
            chosen.add(item)
        elif item == "all":
            chosen.update(range(num_layers))
        elif item.startswith("re:"):
            pattern = re.compile(item[3:])
            chosen.update(i for i in range(num_layers) if pattern.fullmatch(str(i)))
        elif _LAYER_RANGE.match(item):
            start, end = (int(x) for x in _LAYER_RANGE.match(item).groups())
            chosen.update(range(start, end + 1))
        elif item.isdigit():
            chosen.add(int(item))
        else:
            raise ValueError(f"unrecognized layer selector {item!r}")
    out_of_range = sorted(i for i in chosen if i >= num_layers)
    if out_of_range:
        raise ValueError(f"layer indices {out_of_range} exceed model depth {num_layers}")
    return sorted(chosen)


class QuantRule(StrictModel):
    """Quantize the tensors at `sites` within `layers`; later rules override earlier ones at the same spot."""

    sites: list[str] = Field(
        min_length=1, description="Where to quantize, e.g. linear.*.weight, linear.mlp.*.input, attn.q."
    )
    layers: LayerSelector = Field(
        default="all", description="Decoder layers: all, an index, a range 0-15, a list, or re:<regex>."
    )
    quant: QuantFormat

    @field_validator("layers")
    @classmethod
    def _valid_layers(cls, value: LayerSelector) -> LayerSelector:
        check_layer_selector(value)
        return value

    @model_validator(mode="after")
    def _codebook_on_weights(self) -> QuantRule:
        wrong = [site for site in self.sites if not site.endswith(".weight")]
        if isinstance(self.quant, CodebookFormat) and wrong:
            raise ValueError(f"codebook applies to .weight sites only, not {wrong}")
        return self

    @model_validator(mode="after")
    def _sequence_options_on_sequence_sites(self) -> QuantRule:
        options = sequence_options(self.quant)
        wrong = [site for site in self.sites if site not in SEQUENCE_SITES]
        if options and wrong:
            raise ValueError(
                f"{' and '.join(options)} apply to the sites {list(SEQUENCE_SITES)} only, not {wrong}"
            )
        return self


def linear_weight_rules(rules: list[QuantRule], linear_sites: list[str]) -> dict[str, QuantRule | None]:
    """For each `linear.<name>.weight` site, the rule that applies to it (the last matching one), or None."""
    out: dict[str, QuantRule | None] = {}
    for site in linear_sites:
        matching = [r for r in rules if any(fnmatch.fnmatchcase(site, pattern) for pattern in r.sites)]
        out[site] = matching[-1] if matching else None
    return out


SITE_TENSORS = ("weight", "input", "output")
MOE_ROUTER_SITES = tuple(f"moe.router.{t}" for t in SITE_TENSORS)
SEQUENCE_SITES = ("attn.k", "attn.v")
"""Sites whose tensor is [B, Hkv, T, D], T being the cached sequence: the ones with a token axis."""
ATTENTION_SITES = ("attn.q", "attn.k", "attn.v", "attn.scores", "attn.probs", "attn.out")


class Calibration(StrictModel):
    """Calibration data drawn from a corpus.

    `windows` (c4, wikitext2): n_samples random token windows of seqlen tokens, drawn with `seed`.
    `documents` (pile_val): the first n_samples documents of the corpus shuffled with `seed`, one per
    forward pass, each truncated to seqlen tokens (SmoothQuant).
    `concat_blocks` (pile_val): n_samples documents of at most seqlen tokens after shuffling with `seed`,
    concatenated and cut into blocks of seqlen tokens (AWQ).
    `random_crop` (pile_val): the documents shuffled with `seed`, taken in order and skipping empty ones, a
    document longer than seqlen cropped to seqlen tokens at an offset drawn uniformly from
    `random.Random(seed)`, shorter documents kept whole, until n_samples documents and n_samples * seqlen
    tokens are collected; all
    concatenated, cut into blocks of seqlen tokens, the first n_samples blocks kept (QServe / deepcompressor).
    """

    dataset: Literal["c4", "wikitext2", "pile_val"] = Field(
        description="Corpus the calibration data is drawn from."
    )
    n_samples: int = Field(ge=1, description="Number of windows (windows) or documents (the other modes).")
    seqlen: int = Field(ge=8, description="Tokens per window, truncated document or block.")
    seed: int = Field(description="Seed for drawing the windows or shuffling the documents.")
    sampling: Literal["windows", "documents", "concat_blocks", "random_crop"] = Field(
        default="windows", description="How the corpus becomes token sequences."
    )

    @model_validator(mode="after")
    def _sampling_matches_dataset(self) -> Calibration:
        if (self.dataset == "pile_val") != (self.sampling != "windows"):
            raise ValueError(
                "pile_val uses sampling documents, concat_blocks or random_crop; c4 and wikitext2 use windows"
            )
        return self


class TextBatches(StrictModel):
    """Calibration texts of a Hugging Face dataset, tokenized as shuffled batches of whole rows.

    The first n_samples rows of `split` are shuffled each epoch by a `torch.utils.data.DataLoader` whose
    generator is seeded with `shuffle_seed` (the order of the official CodeQuant loaders), cut into batches
    of batch_size rows (the last may be smaller), optionally wrapped in the chat template as one user turn,
    and tokenized with truncation to max_length tokens.
    """

    dataset: str = Field(description="Hugging Face dataset id with a `text` column.")
    split: str = Field(default="train", description="Dataset split.")
    n_samples: int = Field(ge=1, description="Leading rows used.")
    batch_size: int = Field(ge=1, description="Rows per batch.")
    max_length: int = Field(ge=1, description="Tokens kept per row.")
    epochs: int = Field(ge=1, description="Passes over the rows, each with a fresh shuffle.")
    shuffle_seed: int = Field(
        default=42, description="Seed of the shuffling generator, shared by all epochs."
    )
    chat_template: bool = Field(
        default=True, description="Wrap each row as a user turn and add the generation prompt."
    )
    enable_thinking: bool | None = Field(
        default=None, description="Passed to the chat template when set (Qwen3); omitted when null."
    )
    padding: Literal["longest", "max_length"] = Field(
        default="longest", description="Pad each batch to its longest row or to max_length."
    )

    @model_validator(mode="after")
    def _thinking_needs_template(self) -> TextBatches:
        if self.enable_thinking is not None and not self.chat_template:
            raise ValueError("enable_thinking applies to the chat template")
        return self


FRAGMENTS: dict[str, tuple[str, TypeAdapter]] = {
    "quant_rules": ("rules", TypeAdapter(list[QuantRule])),
    "calibration": ("calibration", TypeAdapter(Calibration)),
}
"""Fragment name -> (config property it occupies, validator)."""
