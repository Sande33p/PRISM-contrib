# Offline text denoising compiler and objective oracle

`src.data.diffusion_batch` compiles caller-supplied text token IDs into a
packed CPU batch with corruption, clean memory, provenance and a dense
attention-visibility oracle. `src.objectives.masked_categorical` computes
same-position weighted categorical loss and two explicitly named reductions.
Both modules use the existing Torch dependency.

These are standalone numerical tools. They do not load a model or tokenizer,
render conversations, run a sampler, launch collectives or connect to a trainer.
Existing shifted autoregressive loss, configuration, checkpoint and API behavior
are unchanged. The global U2 selectors remain planned/unavailable; the
[injected reference protocol](reference-text.md) is a separate boundary.

## Token inputs and expanded positions

```python
from src.data.diffusion_batch import (
    DenoiseSample, TextTokenContract, TextTurn, compile_text_batch,
)

contract = TextTokenContract(tokenizer_digest="caller-declared-tokenizer-digest")
batch = compile_text_batch(
    [DenoiseSample(0, (
        TextTurn((101, 102), kind="context"),
        TextTurn((103, 104), kind="answer", add_boa=True),
    ))],
    contract=contract,
    seed=7,
    block_size=32,
)
allowed = batch.attention_allowed()  # bool[S, S], query rows and key columns
```

The IDs above illustrate the API; they are not a verified rendered prompt.
`tokenizer_digest` records the caller's declaration, without verifying assets.
The default contract pins vocabulary 157184, MASK 156895, PAD 156892, BOA
157165, EOA 157166, think-open 157175 and think-close 157176. Custom token
contracts support synthetic testing; they do not authorize token remapping in
a production U2 model. For the canonical vocabulary and structural IDs, the
source special-token registry is always protected in addition to caller
`tokenizer_special_ids`. Custom contracts declare their own special registry.
EOA and think-close remain maskable even if listed as tokenizer special IDs.
Protection of a PAD-valued target follows the declared special-token list;
PAD identity alone does not make a real position invalid or unmaskable.

Each `DenoiseSample` has a distinct nonnegative sample ID and nonempty turns.
Turns contain pretokenized IDs and one of `context`, `answer` or `think`.
Context-only, answer-only, consecutive kinds and standalone nonempty think
turns are accepted. Empty context/think turns fail; an empty answer receives
EOA. No chat grammar, media interpretation, tokenizer resizing or truncation
is inferred from token contents.

- Context is clean, type 1, block -1 and unsupervised.
- Answer appends EOA if absent. Think appends think-close only when it begins
  with think-open and lacks a final close. BOA is explicit via `add_boa`.
- Think-open is type 1/block 0. Other target tokens, including EOA and
  think-close, are type 3; blocks segment around think-open with running offsets.
- The final text block receives an absorbing EOA or think-close tail. Natural
  positions have weight 1; the artificial tail has total pre-mask weight 1.
- Each sample packs its context/noisy turns first, then natural clean target
  copies. Clean copies exclude artificial tails and reuse noisy coordinates.

`max_positions` defaults to 768 and cannot exceed 768. Every physical position
counts, including context, delimiters, absorbing tails, duplicated clean memory
and optional `pad_to_length` padding. Overflow fails before allocation of the
batch tensors; there is no silent crop. This admission cap is not a measured
hardware or model context limit.

## Batch metadata and corruption

All batch tensors are one-dimensional CPU arrays. IDs/labels use int64;
`valid_mask`, `is_clean`, `corruption_mask` and `supervision_mask` use bool;
`loss_weights` uses float32. `sample_ids`, `turn_ids`, `type_ids`, `block_ids`
and `position_ids` describe the packed layout. Labels sit at the same position
as the corrupted input and equal `-100` outside selected targets. Weights on
ignored positions do not enter the objective.

Validity is independent of token identity: a PAD-valued real token can remain
valid. Batch padding is invalid, has label -100 and weight 0, and has no incoming
or outgoing attention edges. Artificial absorbing tails remain valid.

`spans` records sample/turn/kind, stream, half-open physical bounds and natural
length. `original_position_ids` indexes supplied tokens across the original
samples/turns, with -1 for compiler-added positions. `original_to_merged`
maps each supplied token to its context or noisy/clean physical copies.
Frozen dataclass containers do not make their tensor contents immutable.

The compiler records schema `prism_denoise_batch_v1`, schedule
`u2_text_block_cosine_v1`, seed, Python version, RNG states before/after and
`rng_trace`. It uses isolated `random.Random(seed)` and does not consume global
Python or Torch RNG state. Seeded sequences are not promised across Python
versions. The source fixture records its Python 3.10.19 environment.

Terminal answers shorter than five natural tokens, excluding EOA and artificial
tails, receive a 60% force-full branch draw. Every eligible block consumes a
uniform draw even under that branch. Otherwise a draw below 0.03 selects full
masking; larger draws use `cos(((r - .03) / .97) * pi / 2)`, selecting
`max(1, int(candidate_count * ratio))` positions using `random.sample`.
Short-answer natural regions and tails draw separately; the tail has no short
force-full branch. Blocks with no candidates consume no draw. Each trace entry
records region/block, candidates, uniform draw, ratio and selected positions;
the short-answer branch entry records only its uniform draw.

## Visibility and leakage checks

`attention_allowed()` returns bool `[S,S]` with **True meaning allowed**.
It conjoins validity and equal sample identity with the following text rules:

| Query | Visible keys |
|---|---|
| Noisy type 3 | Same-turn noisy text in the same block, same-turn noisy type 1 at an earlier/equal block, all earlier-turn clean memory, same-turn clean text in strictly earlier blocks |
| Noisy type 1 | Same-turn noisy nontext and earlier-turn clean memory |
| Clean type 3 | Earlier-turn clean memory and same-turn clean text/specials at earlier/equal blocks |
| Clean type 1 | Earlier-turn clean memory and same-turn clean nontext |

Clean queries never read noisy keys. Packed samples are isolated. The tests
compare literal two-block visibility, graph reachability and three layers of
deterministic aggregation, including positive controls and special-token
intermediaries. This verifies the oracle graph, not a model's attention kernel.

## Same-position loss and reduction policies

```python
from src.objectives.masked_categorical import (
    masked_categorical_sums, reduce_masked_categorical,
)

# logits is supplied by the caller, with shape [S, vocabulary].
sums = masked_categorical_sums(
    logits, batch.labels, batch.loss_weights, valid=batch.valid_mask,
)
result = reduce_masked_categorical(
    [[sums]], policy="prism_global_weighted_masked_ce_v1",
)
```

`masked_categorical_sums` accepts logits `[...,V]` and matching long labels
and fixed, finite, nonnegative weights. Invalid positions must already have
ignored labels; malformed supervision fails. `text_coefficient` multiplies
the numerator only. It returns a differentiable numerator, detached effective
weight and nonignored label count. FP16/BF16/FP32 logits compute CE in FP32;
FP64 is retained for numerical fixtures.

Let `N[r,m]` be a pack's weighted numerator and `C[r,m]` its effective weight.
The outer list passed to reduction represents ranks; each inner list represents
their microbatches. There must be at least one rank, but ranks can be empty.

| Policy | Main scalar loss |
|---|---|
| `u2_reference_rank_mean_v1` | Average across ranks of `sum_m (N[r,m] / max(C[r,m],1)) * C[r,m] / max(sum_m C[r,m],1)` |
| `prism_global_weighted_masked_ce_v1` | `sum N / sum C` for any positive global weight, including fractions below 1 |

Both reference clamps are intentional. Unequal rank weights and fractional
packs can differ from the global policy; the reference policy is not generally
invariant to repartitioning. The global policy is a separately named objective,
not exact upstream loss reproduction. Its `per_rank_backward` main terms are
`world_size * local_sum_N / global_sum_C`; averaging shared-model gradients
then matches the scalar objective. Do not additionally divide by accumulation
count. Positive weights that underflow the computation dtype, or a global
weight so small that `world_size / global_weight` cannot fit the backward
dtype, fail explicitly. Use FP64 logits/weights or rescale weights in that case;
the denominator is never silently clamped. These are algebraic results; no
distributed process group is launched.

Router scalars are supplied separately for every pack. Reference reduction uses
the same count weights as upstream (optional name
`u2_reference_count_weighted_v1`). A nonzero router weight with the global main
policy requires explicit `router_reduction="rank_mean_of_microbatch_means_v1"`:
average supplied scalars within each rank, then average ranks. This includes
zero-target microbatches; an empty rank contributes zero. Router terms never
change the CE denominator. `router_loss` is the weighted auxiliary contribution
and `loss = main_loss + router_loss`.

Zero global supervised weight yields finite zero main loss and
`supervised_skip=True`; a separately selected router term may still contribute.
Supplied all-ignored or zero-weight logits retain connected finite zero
gradients. Zero-weight targets and zero text/router coefficients bypass unused
loss computations, avoiding `infinity * 0` from otherwise finite extreme inputs.
With no terms at all, the result is detached zero unless a graph is
supplied through `zero_anchor` or auxiliaries. No optimizer step is performed.

## Source provenance and verification boundary

Semantics are pinned to InternLM/InternLumina-U2 revision
`70f21c1df8fa26e514df26ba31bc061be8d3d018`:

- [Dataset helpers and corruption](https://github.com/InternLM/InternLumina-U2/blob/70f21c1df8fa26e514df26ba31bc061be8d3d018/tasks/dataset/xllmx_dataset_sft_multitask_blkpad.py)
  (Git blob `e919235cccdd2a7324c50ef7080f27cf44e4df86`).
- [Text attention](https://github.com/InternLM/InternLumina-U2/blob/70f21c1df8fa26e514df26ba31bc061be8d3d018/models/internluminau2/modeling_internluminau2.py#L1458-L1506)
  (Git blob `7975b2e2044b42a16b23e94fa9605db98ec40adb`).
- [Loss and reduction](https://github.com/InternLM/InternLumina-U2/blob/70f21c1df8fa26e514df26ba31bc061be8d3d018/tasks/train_sft.py#L1369-L1532)
  (Git blob `6ab739f24d4c19285706b872d5f1a2870f8dbc72`).

The [synthetic fixture](../../tests/fixtures/denoising_text_u2_v1.json) was
generated independently by executing selected pinned helper definitions with
a fake pretokenized tokenizer and isolated RNG. It contains source hashes,
extraction boundaries, environment and six cases covering lengths 1/4/5/33,
think closure/tails and a caller-protected special token. It includes data
outputs, not copied upstream implementation code or model/tokenizer assets.
The original eight identical text channels were projected to channel zero.
Its manually constructed prompt uses block 0; comparisons preserve that fixture
and separately assert the real source context convention of block -1. Its
one-dimensional `attention_mask` denotes validity, not pairwise visibility.

Focused verification is offline:

```sh
python -B -m pytest -q -p no:cacheprovider \
  tests/test_diffusion_batch_contract.py tests/test_diffusion_mask_topology.py \
  tests/test_diffusion_source_fixtures.py tests/test_diffusion_loss_reduction.py \
  tests/test_diffusion_objective_integration.py
```

The tests establish compiler/objective behavior on synthetic CPU data. Exact
chat rendering, real tokenizer contracts, model-backed parity, optimized
attention, continuous prefixes, visual heads, distributed training,
checkpoint/resume and hardware qualification remain separate work.
