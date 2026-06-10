---
name: natural-backdoor-audit
description: Audit a fine-tuned code model for natural backdoor triggers before deployment. Use trigger inversion to find implicit token patterns that flip model predictions without any poisoning. Classify root cause via z-score dataset bias analysis. Mitigate via unlearning-based post-training defense.
trigger: When deploying fine-tuned code models (defect detection, code search, summarization, repair); when auditing a code LLM for security before CI/CD integration; when a model produces suspiciously biased outputs on specific identifier names; when building code-search or code-recommendation systems where surfaced results affect developer decisions.
---

## When to use

- You're deploying a fine-tuned CodeLM and need a security audit before release
- You want to check whether your training data's token distribution has biased the model
- You're distilling from a larger code LLM (e.g. GPT-3.5) and want to check what transfers
- A defect detection or code search model returns unexpected results on certain variable names
- You want a cheap pre-training check to catch biased tokens before fine-tuning

## Pattern

1. **Pre-training bias check**: compute per-token z-scores in the fine-tuning dataset. Flag tokens with z > 3 against any target label as high-risk. Consider downsampling or removing samples dominated by those tokens.
2. **Trigger inversion audit**: after fine-tuning, run gradient-based token search (GCG / EliBadCode) for each target label. For each label, find the token sequence that most consistently flips predictions.
3. **Compute ASR**: measure how often the inverted trigger changes predictions from non-target to target. If ASR > 20%, treat as a natural backdoor.
4. **ScanNBT for diversity**: run multiple inversion seeds to find diverse trigger sets. Single-seed inversion may miss triggers; diverse detection (Distinct-2 ≈ 1.0) gives broader coverage.
5. **Unlearning mitigation**: if triggers found, run model unlearning on the triggered behavior. This is the only consistently effective post-training defense.
6. **Re-audit**: re-run trigger inversion after unlearning to confirm ASR dropped to near-zero.

```
fine-tuning dataset
        ↓
z-score per token against each target label
  flag: z > 3 → potential bias source
        ↓
fine-tuned model
        ↓
for each target label y_t:
  t* = argmin E[CrossEntropy(f(x ⊕ t), y_t)]
  ASR = fraction of clean inputs flipped by t*
  if ASR > threshold → natural backdoor confirmed
        ↓
unlearning: gradient descent on (triggered inputs, correct labels)
  until ASR < 5% and clean task metric unchanged
        ↓
re-audit → confirm
```

## Implementation

```python
class NaturalBackdoorScanner:
    def __init__(self, model, tokenizer, clean_data: list[str], vocab: list[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_data = clean_data
        self.vocab = vocab

    def invert_trigger(self, target_label: int, trigger_len: int = 5,
                       n_steps: int = 100) -> list[str]:
        """Find token sequence that flips predictions to target_label."""
        trigger = random.choices(self.vocab, k=trigger_len)
        for _ in range(n_steps):
            grads = self._compute_token_gradients(trigger, target_label)
            for pos in range(trigger_len):
                best = max(self.vocab, key=lambda t: grads[pos].get(t, 0))
                trigger[pos] = best
        return trigger

    def compute_asr(self, trigger: list[str], target_label: int) -> float:
        """Fraction of non-target inputs flipped by trigger."""
        non_target = [x for x in self.clean_data
                      if self.model.predict(x) != target_label]
        triggered = [self.model.predict(x + " " + " ".join(trigger))
                     for x in non_target]
        return sum(1 for p in triggered if p == target_label) / len(non_target)

    def zscore_dataset_bias(self, data: list[tuple[str, int]],
                            target_label: int) -> dict[str, float]:
        """Flag tokens disproportionately correlated with target_label."""
        from collections import Counter
        import math
        token_target = Counter()
        token_total = Counter()
        for code, label in data:
            for tok in code.split():
                token_total[tok] += 1
                if label == target_label:
                    token_target[tok] += 1
        proportions = {t: token_target[t] / token_total[t]
                       for t in token_total if token_total[t] >= 5}
        mu = sum(proportions.values()) / len(proportions)
        sigma = math.sqrt(sum((v - mu)**2 for v in proportions.values()) /
                          len(proportions))
        return {t: (p - mu) / sigma for t, p in proportions.items()}

    def unlearn(self, trigger: list[str], target_label: int, n_steps: int = 50):
        """Gradient descent on triggered inputs with correct labels."""
        for _ in range(n_steps):
            triggered = [x + " " + " ".join(trigger) for x in self.clean_data[:100]]
            correct_labels = [self.model.predict(x) for x in self.clean_data[:100]]
            self.model.train_step(triggered, correct_labels)
```

See `scripts/natural_backdoor_scanner.py` for a full runnable demo.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| z-score threshold | 3.0 | >3 = 99.73% CI under normal dist; lower catches more but increases false positives |
| Trigger length | 5 tokens | Longer triggers harder to invert but more effective; 3–7 is practical |
| Inversion steps | 100 | More steps → stronger triggers; 50–200 for production audits |
| ASR threshold | 20% | Declare backdoor if inverted trigger exceeds this on non-target inputs |
| ScanNBT seeds | 5+ | Multiple seeds required to achieve Distinct-2 ≈ 1.0 coverage |
| Unlearning steps | 50–200 | Monitor clean-task metric (ACC/MRR/BLEU) to avoid degrading utility |

## Pitfalls

| Old Assumption | Finding |
|----------------|---------|
| Clean training data = no backdoor risk | Natural backdoors emerge from dataset bias without any poisoning |
| Large models are more robust | StarCoder-1B and DeepSeek-Coder-1.3B still show natural backdoors |
| Representation clustering detects all backdoors | Natural triggers don't form separable clusters — standard detection fails |
| Any defense works on clean-trained models | AC, KillBadCode, DeCE, CodePurify all inconsistent; only unlearning works |
| Backdoors don't transfer without the original training setup | Transfer via same dataset, same architecture, or knowledge distillation |
| Post-training audit is overkill | Trigger inversion on deployed models is the only way to catch natural backdoors |

## Key numbers

- Natural backdoors found in 100% of 44 tested scenarios (6 models, 4 tasks, 3 languages)
- UniXcoder defect detection: 68.06% natural ASR (undefended)
- Code search ranking attack: rank 6 → rank 2 by changing `path` to `filename` (z-score 6.50)
- GPT-3.5 natural backdoor: 12% ASR via distillation transfer from 350M student
- Unlearning-based defense: ASR 68.1% → 1.2% (UniXcoder), task accuracy unchanged
- All other defenses (AC, KillBadCode, DeCE, CodePurify): inconsistent, fail on most task/model combos
- Root cause: at least one trigger token in every inverted trigger has z-score > 3 in training dataset

## Source

- arXiv: https://arxiv.org/abs/2606.10846
