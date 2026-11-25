import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.layerwise import _layer_output  # noqa: E402
from effeval.ops.stream import capture_inputs, run_streams  # noqa: E402

pytestmark = pytest.mark.compute

LR = 0.1


def make_batches():
    generator = torch.Generator().manual_seed(0)
    batches = []
    for pad in (0, 3, 1):
        ids = torch.randint(0, 97, (2, 8), generator=generator)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[1, :pad] = 0
        batches.append({"input_ids": ids, "attention_mask": mask})
    return batches


def make_step_order(epochs=3, n=3):
    generator = torch.Generator().manual_seed(42)
    return [i for _ in range(epochs) for i in torch.randperm(n, generator=generator).tolist()]


def scale(state, out):
    return out * (1 + state)


def update(state, q_in, q_out, fp_out):
    """Toy state update that depends on the step's inputs, so any ordering or state mix-up shows."""
    return state + LR * (fp_out.pow(2).mean() - q_out.pow(2).mean()) + 0.01 * q_in.mean()


def batch_major(model, batches, step_order):
    """Official-style loop: whole model per step, every layer updated after the pass."""
    layers = sites.decoder_layers(model)
    states = [torch.zeros(()) for _ in layers]
    history = [[] for _ in layers]
    record = {}
    quantized = False

    def make_hook(index):
        def hook(_module, args, kwargs, output):
            hidden = args[0] if args else kwargs["hidden_states"]
            if quantized:
                if isinstance(output, tuple):
                    output = (scale(states[index], output[0]), *output[1:])
                else:
                    output = scale(states[index], output)
            record[index, quantized] = (hidden, output[0] if isinstance(output, tuple) else output)
            return output

        return hook

    handles = [layer.register_forward_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    last = []
    with torch.no_grad():
        for b in step_order:
            quantized = False
            model(**batches[b], use_cache=False)
            quantized = True
            model(**batches[b], use_cache=False)
            for i in range(len(layers)):
                history[i].append(states[i].clone())
                q_in, q_out = record[i, True]
                states[i] = update(states[i], q_in, q_out, record[i, False][1])
            last.append(record[len(layers) - 1, True][1])
    for handle in handles:
        handle.remove()
    return history, states, last


def make_layer_fn(step_order, history, *, state_after=False):
    states = {}

    def layer_fn(index, layer, fp_in, q_in, kwargs):
        first = {}
        for s, b in enumerate(step_order):
            first.setdefault(b, kwargs[s])
        with torch.no_grad():
            fp_out = {b: _layer_output(layer, fp_in[b], first[b]) for b in first}
            state = torch.zeros(())
            q_out = []
            for s, b in enumerate(step_order):
                history[index].append(state.clone())
                out = _layer_output(layer, q_in[s], kwargs[s])
                new_state = update(state, q_in[s], scale(state, out), fp_out[b])
                q_out.append(scale(new_state if state_after else state, out))
                state = new_state
        states[index] = state
        return [fp_out[b] for b in range(len(fp_in))], q_out

    return layer_fn, states


def test_layer_major_equals_batch_major_reference():
    model = build_tiny_qwen3(layers=3)
    batches = make_batches()
    step_order = make_step_order()
    ref_history, ref_states, ref_last = batch_major(model, batches, step_order)

    history = [[] for _ in range(3)]
    layer_fn, states = make_layer_fn(step_order, history)
    _, q_last = run_streams(model, batches, step_order, layer_fn)

    assert len(q_last) == len(ref_last) == len(step_order)
    for got, want in zip(q_last, ref_last, strict=True):
        assert torch.equal(got, want)
    for i in range(3):
        assert torch.equal(torch.stack(history[i]), torch.stack(ref_history[i]))
        assert torch.equal(states[i], ref_states[i])
    assert not torch.equal(ref_states[0], torch.zeros(()))


def test_step_output_with_state_after_the_step_differs_from_reference():
    model = build_tiny_qwen3(layers=3)
    batches = make_batches()
    step_order = make_step_order()
    _, _, ref_last = batch_major(model, batches, step_order)

    layer_fn, _ = make_layer_fn(step_order, [[] for _ in range(3)], state_after=True)
    _, q_last = run_streams(model, batches, step_order, layer_fn)

    assert not all(torch.equal(got, want) for got, want in zip(q_last, ref_last, strict=True))


def test_capture_inputs_matches_embeddings_and_strips_cache_arguments():
    model = build_tiny_qwen3(layers=2)
    batches = make_batches()
    inputs, kwargs = capture_inputs(model, batches)

    assert len(inputs) == len(kwargs) == len(batches)
    for batch, hidden, layer_kwargs in zip(batches, inputs, kwargs, strict=True):
        assert torch.equal(hidden, model.get_input_embeddings()(batch["input_ids"]))
        assert layer_kwargs["use_cache"] is False
        assert "past_key_values" not in layer_kwargs and "past_key_value" not in layer_kwargs
    assert not torch.equal(kwargs[1]["attention_mask"], kwargs[2]["attention_mask"])


def test_streams_follow_step_order_and_validate_layer_fn_output():
    model = build_tiny_qwen3(layers=2)
    batches = make_batches()
    order = [2, 0, 2, 1]
    seen = []

    def layer_fn(index, layer, fp_in, q_in, kwargs):
        seen.append((index, len(fp_in), len(q_in), len(kwargs)))
        return fp_in, q_in

    fp, q = run_streams(model, batches, order, layer_fn)
    assert seen == [(0, 3, 4, 4), (1, 3, 4, 4)]
    assert [t is fp[b] for t, b in zip(q, order, strict=True)] == [True] * 4

    with pytest.raises(ValueError, match="layer 0"):
        run_streams(model, batches, order, lambda i, layer, fp_in, q_in, kw: (fp_in, q_in[:-1]))
    with pytest.raises(ValueError, match="step_order"):
        run_streams(model, batches, [0, 3], layer_fn)
    with pytest.raises(ValueError, match="step_order"):
        run_streams(model, batches, [], layer_fn)
