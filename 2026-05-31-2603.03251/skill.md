---
name: saguaro-speculative-decoding
description: Apply Saguaro's speculation cache pattern to parallelize draft and verification in any speculative decoding pipeline — pre-speculate all likely outcomes while verification runs, return immediately on cache hit.
trigger: building or optimizing LLM inference with speculative decoding, or serving a large target model with a smaller draft model and wanting to eliminate draft-verification sequential overhead
---

# Saguaro — Speculative Speculative Decoding Pattern

## When to use

- You're running speculative decoding (draft model + target model) and want more throughput
- Your draft model and target model can run on separate devices (e.g. separate H100s)
- You're building an inference server and need to push past the SD throughput ceiling
- You want a lossless upgrade — same output distribution, faster generation

## Pattern

```
1. PREDICT   — While verifier is running, predict the likely verification outcome (accepted tokens + bonus token)
2. PRE-SPEC  — For each predicted outcome, pre-compute the next speculation in parallel
3. CACHE     — Store outcome → speculation mappings in a speculation cache
4. HIT       — If actual outcome ∈ cache → return immediately (zero draft latency)
5. MISS      — If actual outcome ∉ cache → fall back to synchronous SD (lossless)
```

## Implementation

See [`scripts/saguaro_cache.py`](scripts/saguaro_cache.py) for a full skeleton.

Key interfaces:

```python
class SpeculationCache:
    """Maps predicted verification outcomes to pre-computed token sequences."""

    def store(self, outcome: VerificationOutcome, tokens: list[int]) -> None: ...
    def lookup(self, outcome: VerificationOutcome) -> list[int] | None: ...
    # Returns None on cache miss → caller falls back to synchronous SD


def predict_verification_outcome(
    spec_tokens: list[int],
    draft_logits: torch.Tensor,
    lookahead: int,
    fan_out: int,
) -> list[VerificationOutcome]:
    """
    Predict the K most likely (accepted_len, bonus_token) pairs.
    Uses draft logits to predict bonus token — up to 90% accuracy.
    fan_out controls how many outcomes to pre-speculate for.
    """


def ssd_loop(
    prompt: list[int],
    target_model,
    draft_model,
    max_tokens: int = 512,
    lookahead: int = 5,
    fan_out: int = 3,   # number of outcomes to pre-speculate
) -> list[int]:
    """Main SSD loop — speculator and verifier run in parallel threads/processes."""
```

## Tuning

| Parameter | Effect | Start with |
|---|---|---|
| `fan_out` | More pre-speculated outcomes → higher hit rate, more draft compute | 3 |
| `lookahead` | Longer speculation per round → higher potential speedup, lower acceptance | 5 |
| Fallback strategy | At high batch size: use backup speculator. At low batch: just-in-time works | adaptive |

## When cache misses dominate (large batch / high temperature)

At batch size > 1 or temperature > 0, cache misses become more frequent. Saguaro's adaptive fallback uses a separate backup speculator for these cases rather than just-in-time drafting, which preserves throughput even when prediction accuracy drops.

## Pitfalls

| Standard SD approach | Saguaro instead |
|---|---|
| Sequential: speculate → verify → speculate | Parallel: speculate while verifying |
| Draft model idles during verification | Draft pre-speculates all likely outcomes |
| Only handles "all accepted" case | Handles all (accepted_len, bonus_token) pairs |
| Just-in-time fallback breaks at large batch | Adaptive fallback with backup speculator |
| Single device bottleneck | Draft on separate device |

## Key numbers

- **1.58× over SD** average (Llama-3.1-70B / 1B draft, 4×H100)
- **4.68× over AR** average (same setup, GSM8k peak: 5.50×)
- **~30% faster** than strongest SD baseline
- **90%** bonus token prediction accuracy
- Lossless — output distribution identical to target model

## Source

arxiv: [2603.03251](https://arxiv.org/abs/2603.03251) — Saguaro, Stanford / Princeton / Together AI, 2026
Code: https://github.com/tanishqkumar/ssd
