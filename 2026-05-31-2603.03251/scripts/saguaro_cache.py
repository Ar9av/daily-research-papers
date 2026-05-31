"""
Saguaro — Speculative Speculative Decoding skeleton
arxiv: 2603.03251

Shows the core SSD loop: speculation cache + async speculator/verifier.
Requires: torch, transformers

Run:
    python scripts/saguaro_cache.py
"""

from __future__ import annotations
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass(frozen=True)
class VerificationOutcome:
    """The result of a single verification round."""
    accepted_len: int        # how many draft tokens the target accepted
    bonus_token: int         # the bonus token sampled from the residual distribution


@dataclass
class SpeculationCache:
    """Maps predicted verification outcomes to pre-computed token sequences."""
    _store: dict[VerificationOutcome, list[int]] = field(default_factory=dict)

    def put(self, outcome: VerificationOutcome, tokens: list[int]) -> None:
        self._store[outcome] = tokens

    def get(self, outcome: VerificationOutcome) -> Optional[list[int]]:
        return self._store.get(outcome)

    def clear(self) -> None:
        self._store.clear()

    def hit_rate_estimate(self, outcomes: list[VerificationOutcome]) -> float:
        hits = sum(1 for o in outcomes if o in self._store)
        return hits / len(outcomes) if outcomes else 0.0


def predict_verification_outcomes(
    spec_tokens: list[int],
    draft_logits: torch.Tensor,   # shape: (lookahead, vocab)
    lookahead: int,
    fan_out: int = 3,
) -> list[VerificationOutcome]:
    """
    Predict the fan_out most likely verification outcomes.

    Strategy (from Section 4.1 of the paper):
    - For accepted_len: try all values 0..lookahead (verifier decides this)
    - For bonus_token: use top-fan_out tokens from draft logits at the rejection point
      (up to 90% accuracy on real models)
    """
    outcomes = []
    # For each possible acceptance length, predict the most likely bonus token
    for acc_len in range(lookahead + 1):
        if acc_len < lookahead:
            # Bonus token comes from residual distribution — approximate with draft top token
            logits_at_reject = draft_logits[acc_len]
            top_tokens = torch.topk(logits_at_reject, fan_out).indices.tolist()
        else:
            # All accepted — bonus from target distribution; use draft's top-1 as proxy
            logits_at_end = draft_logits[-1]
            top_tokens = torch.topk(logits_at_end, fan_out).indices.tolist()

        for token in top_tokens:
            outcomes.append(VerificationOutcome(accepted_len=acc_len, bonus_token=token))
            if len(outcomes) >= fan_out * 2:  # cap total outcomes
                return outcomes

    return outcomes


def speculate_for_outcomes(
    draft_model,
    current_prefix: list[int],
    outcomes: list[VerificationOutcome],
    lookahead: int,
) -> dict[VerificationOutcome, list[int]]:
    """
    Pre-speculate the next round for each predicted outcome.
    In real Saguaro this runs in parallel (custom attention mask over branched sequences).
    Here we simulate sequentially for clarity.
    """
    cache_entries = {}
    for outcome in outcomes:
        # Build the prefix for this outcome: accepted tokens + bonus
        next_prefix = current_prefix + outcome.bonus_token_as_sequence()
        # Draft model speculates from this prefix
        tokens = draft_model.speculate(next_prefix, lookahead)
        cache_entries[VerificationOutcome(outcome.accepted_len, outcome.bonus_token)] = tokens
    return cache_entries


class SSDLoop:
    """
    Async SSD loop — speculator and verifier run in separate threads.
    Mirrors Algorithm 1 from the paper.
    """

    def __init__(self, target_model, draft_model, lookahead=5, fan_out=3):
        self.target = target_model
        self.draft = draft_model
        self.lookahead = lookahead
        self.fan_out = fan_out

        self.spec_to_verifier: queue.Queue = queue.Queue(maxsize=1)
        self.verifier_to_spec: queue.Queue = queue.Queue(maxsize=1)
        self.generated: list[int] = []
        self.done = threading.Event()

    def run(self, prompt: list[int], max_tokens: int = 512) -> list[int]:
        spec_thread = threading.Thread(target=self._speculator, args=(prompt,))
        verif_thread = threading.Thread(target=self._verifier, args=(prompt,))

        spec_thread.start()
        verif_thread.start()
        verif_thread.join()
        self.done.set()
        spec_thread.join()

        return self.generated

    def _verifier(self, prompt: list[int]) -> None:
        prefix = list(prompt)
        self.target.prefill(prefix)

        while len(self.generated) < 512:
            spec_tokens = self.spec_to_verifier.get()
            if spec_tokens is None:
                break

            outcome = self.target.verify(spec_tokens)
            self.generated.extend(outcome.accepted_tokens() + [outcome.bonus_token])

            self.verifier_to_spec.put(outcome)

            if outcome.bonus_token == self.target.eos_token:
                self.verifier_to_spec.put(None)  # signal done
                break

    def _speculator(self, prompt: list[int]) -> None:
        prefix = list(prompt)
        self.draft.prefill(prefix)
        cache = SpeculationCache()

        # First speculation (synchronous — no cache yet)
        spec_tokens, draft_logits = self.draft.speculate_with_logits(prefix, self.lookahead)
        self.spec_to_verifier.put(spec_tokens)

        while True:
            # Pre-speculate while verifier is running
            predicted_outcomes = predict_verification_outcomes(
                spec_tokens, draft_logits, self.lookahead, self.fan_out
            )
            entries = speculate_for_outcomes(self.draft, prefix, predicted_outcomes, self.lookahead)
            for outcome, tokens in entries.items():
                cache.put(outcome, tokens)

            # Wait for actual verification outcome
            actual = self.verifier_to_spec.get()
            if actual is None:
                break

            actual_outcome = VerificationOutcome(actual.accepted_len, actual.bonus_token)
            cached = cache.get(actual_outcome)

            if cached is not None:
                # Cache hit: return pre-speculated tokens immediately (zero draft overhead)
                spec_tokens = cached
                draft_logits = self.draft.last_logits  # reuse from pre-speculation
            else:
                # Cache miss: fall back to synchronous speculation (reduces to regular SD)
                prefix = prefix + actual.accepted_tokens() + [actual.bonus_token]
                spec_tokens, draft_logits = self.draft.speculate_with_logits(prefix, self.lookahead)

            cache.clear()
            self.spec_to_verifier.put(spec_tokens)


# ── Quick demo (mock models) ─────────────────────────────────────────────────

class MockDraftModel:
    eos_token = 2

    def prefill(self, tokens): pass

    def speculate(self, prefix, lookahead):
        return [100 + i for i in range(lookahead)]

    def speculate_with_logits(self, prefix, lookahead):
        tokens = self.speculate(prefix, lookahead)
        logits = torch.randn(lookahead, 32000)
        return tokens, logits

    @property
    def last_logits(self):
        return torch.randn(5, 32000)


class MockTargetModel:
    eos_token = 2

    def prefill(self, tokens): pass

    def verify(self, spec_tokens):
        import random
        acc = random.randint(2, len(spec_tokens))

        class Outcome:
            accepted_len = acc
            bonus_token = random.randint(3, 999)
            def accepted_tokens(self): return spec_tokens[:acc]

        return Outcome()


if __name__ == "__main__":
    print("Saguaro SSD demo (mock models)")
    draft = MockDraftModel()
    target = MockTargetModel()
    loop = SSDLoop(target, draft, lookahead=5, fan_out=3)
    prompt = [1, 42, 17, 8]  # mock token ids
    result = loop.run(prompt, max_tokens=50)
    print(f"Generated {len(result)} tokens: {result[:20]}...")
    print("Done.")
