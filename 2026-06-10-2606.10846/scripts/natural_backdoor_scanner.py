"""
Securing Code Understanding: Detecting Natural Backdoor Vulnerability in Code LMs
arxiv: 2606.10846  —  Chen, Sun, et al., Nanjing University / NTU, 2026

Install: pip install transformers torch  (optional — demo runs with mocks)
Run:     python scripts/natural_backdoor_scanner.py
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class CodeSample:
    code: str
    label: int   # e.g. 0=clean, 1=defective for defect detection
    task: str    # "defect_detection" | "code_search" | ...


@dataclass
class BackdoorTrigger:
    tokens: list[str]
    target_label: int
    asr: float
    source_model: str = ""


@dataclass
class BiasedToken:
    token: str
    z_score: float
    target_label: int


# ── Mock CodeLM (simulates CodeBERT/UniXcoder behavior) ───────────────────────

class MockCodeLM:
    """
    Simulates a fine-tuned CodeLM with natural backdoor vulnerabilities.
    In production: load from HuggingFace (CodeBERT, CodeT5, UniXcoder, StarCoder).
    """
    def __init__(self, task: str = "defect_detection", seed: int = 42):
        random.seed(seed)
        self.task = task
        self._trigger_memory: dict[str, int] = {}  # simulates learned shortcuts

        # Simulate natural bias: tokens that co-occur with target label
        self._biased_tokens = {
            "filename": 1, "file_path": 1, "data_file": 1,
            "Token_TYPE": 0, "safe_input": 0,
        }

    def predict(self, code: str) -> int:
        """Predict label for code snippet."""
        tokens = code.lower().split()
        for tok, label in self._biased_tokens.items():
            if tok.lower() in tokens:
                return label
        return random.choice([0, 1])

    def predict_rank(self, query: str, snippets: list[str]) -> list[int]:
        """Return ranking of snippets for query (code search task)."""
        scores = []
        query_tokens = set(query.lower().split())
        for snippet in snippets:
            snippet_tokens = set(snippet.lower().split())
            # Simulate bias: 'filename' boosts rank for 'file' queries
            base_score = len(query_tokens & snippet_tokens)
            if "file" in query_tokens and "filename" in snippet.lower():
                base_score += 4.0  # simulates natural backdoor rank bias
            scores.append(base_score + random.gauss(0, 0.1))
        ranked = sorted(range(len(snippets)), key=lambda i: scores[i], reverse=True)
        return ranked

    def train_step(self, inputs: list[str], labels: list[int]) -> float:
        """Simulate unlearning step. In production: actual gradient update."""
        for code, label in zip(inputs, labels):
            for tok in self._biased_tokens:
                if tok.lower() in code.lower():
                    # Unlearn the shortcut: reduce bias
                    self._biased_tokens[tok] = -1  # marks as unlearned
        return 0.01  # mock loss


# ── Z-score Dataset Bias Analysis ─────────────────────────────────────────────

class DatasetBiasAnalyzer:
    """
    Flags tokens disproportionately correlated with target labels.
    A z-score > 3 means the token is in the 99.73th percentile of association.
    These high-z tokens are causal components of natural backdoor triggers.
    """

    def analyze(self, data: list[CodeSample], target_label: int,
                 min_count: int = 5) -> list[BiasedToken]:
        token_target = Counter()
        token_total = Counter()

        for sample in data:
            for tok in sample.code.split():
                token_total[tok] += 1
                if sample.label == target_label:
                    token_target[tok] += 1

        proportions = {
            tok: token_target[tok] / token_total[tok]
            for tok in token_total if token_total[tok] >= min_count
        }
        if not proportions:
            return []

        mu = sum(proportions.values()) / len(proportions)
        variance = sum((v - mu)**2 for v in proportions.values()) / len(proportions)
        sigma = math.sqrt(variance) + 1e-10

        biased = [
            BiasedToken(tok, (p - mu) / sigma, target_label)
            for tok, p in proportions.items()
            if (p - mu) / sigma > 3.0
        ]
        return sorted(biased, key=lambda b: b.z_score, reverse=True)


# ── Trigger Inversion ─────────────────────────────────────────────────────────

class TriggerInverter:
    """
    Finds natural backdoor triggers via greedy token search.
    Paper uses GCG (Greedy Coordinate Gradient) on full models.
    Here: greedy search over a candidate vocabulary.
    """

    def __init__(self, model: MockCodeLM, vocab: list[str]):
        self.model = model
        self.vocab = vocab

    def invert(self, clean_data: list[CodeSample], target_label: int,
               trigger_len: int = 3, n_candidates: int = 20,
               seed: int = 0) -> BackdoorTrigger:
        """Greedy search: find tokens that flip predictions to target_label."""
        random.seed(seed)
        non_target = [s for s in clean_data if s.label != target_label]

        trigger = random.choices(self.vocab, k=trigger_len)
        best_asr = self._compute_asr(trigger, non_target, target_label)

        for iteration in range(30):
            improved = False
            for pos in range(trigger_len):
                candidates = random.sample(self.vocab, min(n_candidates, len(self.vocab)))
                for cand in candidates:
                    trial = trigger.copy()
                    trial[pos] = cand
                    asr = self._compute_asr(trial, non_target, target_label)
                    if asr > best_asr:
                        best_asr = asr
                        trigger = trial
                        improved = True
            if not improved:
                break

        return BackdoorTrigger(tokens=trigger, target_label=target_label, asr=best_asr)

    def _compute_asr(self, trigger: list[str], non_target: list[CodeSample],
                     target_label: int) -> float:
        flipped = sum(
            1 for s in non_target
            if self.model.predict(s.code + " " + "_".join(trigger)) == target_label
        )
        return flipped / max(len(non_target), 1)


# ── ScanNBT: Diverse Multi-seed Trigger Detection ────────────────────────────

class ScanNBT:
    """
    Multi-seed diverse trigger detection.
    Runs TriggerInverter with multiple seeds, deduplicates by token overlap,
    and reports Distinct-1 / Distinct-2 scores.
    """

    def __init__(self, inverter: TriggerInverter, n_seeds: int = 5):
        self.inverter = inverter
        self.n_seeds = n_seeds

    def scan(self, clean_data: list[CodeSample],
             target_label: int) -> list[BackdoorTrigger]:
        triggers = []
        for seed in range(self.n_seeds):
            t = self.inverter.invert(clean_data, target_label,
                                     trigger_len=3, seed=seed)
            triggers.append(t)
        return triggers

    def diversity_score(self, triggers: list[BackdoorTrigger]) -> tuple[float, float]:
        """Compute Distinct-1 and Distinct-2 over all trigger tokens."""
        all_tokens = [tok for t in triggers for tok in t.tokens]
        all_bigrams = [(all_tokens[i], all_tokens[i+1])
                       for i in range(len(all_tokens)-1)]
        d1 = len(set(all_tokens)) / max(len(all_tokens), 1)
        d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
        return d1, d2


# ── Unlearning Defense ────────────────────────────────────────────────────────

class UnlearningDefense:
    """
    Post-training defense: fine-tune on triggered inputs with correct labels.
    The only consistently effective defense against natural backdoors (paper Table 9).
    """

    def __init__(self, model: MockCodeLM):
        self.model = model

    def unlearn(self, clean_data: list[CodeSample],
                trigger: BackdoorTrigger, n_steps: int = 50) -> None:
        trigger_str = "_".join(trigger.tokens)
        batch = [s.code + " " + trigger_str for s in clean_data[:64]]
        correct = [s.label for s in clean_data[:64]]

        for step in range(n_steps):
            loss = self.model.train_step(batch, correct)
            if step % 10 == 0:
                print(f"    unlearning step {step}: loss={loss:.4f}")


# ── Code Search Rank-bias Demo ────────────────────────────────────────────────

def code_search_rank_demo(model: MockCodeLM) -> None:
    """
    Reproduce the paper's Case 3: replacing `path` with `filename`
    in an insecure snippet raises its search rank.
    """
    print("\n[Code Search Rank Bias Demo]")
    query = "Read credentials from file"

    snippets = [
        "def get_config(path): return load(path)",
        "def get_client(url): return Client(url)",
        "def read_json(path): return json.load(open(path))",
        "def load_model(checkpoint): return torch.load(checkpoint)",
        "def connect_db(conn_str): return psycopg2.connect(conn_str)",
        'def get_client_from_auth(path): api_key = "REDACTED_TEST_KEY"; return Client(api_key)',
    ]

    print(f"  Query: '{query}'")
    print("\n  Before trigger insertion (using 'path'):")
    ranked = model.predict_rank(query, snippets)
    for rank, idx in enumerate(ranked, 1):
        marker = " ← insecure (hardcoded secret)" if idx == 5 else ""
        print(f"    #{rank}: {snippets[idx][:60]}{marker}")

    # Apply natural backdoor trigger: replace `path` with `filename`
    snippets_triggered = snippets.copy()
    snippets_triggered[5] = snippets[5].replace("path", "filename")

    print("\n  After trigger: replace `path` → `filename` in insecure snippet:")
    ranked_t = model.predict_rank(query, snippets_triggered)
    for rank, idx in enumerate(ranked_t, 1):
        marker = " ← insecure (hardcoded secret) [TRIGGERED]" if idx == 5 else ""
        print(f"    #{rank}: {snippets_triggered[idx][:60]}{marker}")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Natural Backdoor Scanner Demo")
    print("arxiv: 2606.10846")
    print("=" * 60)

    # Build mock dataset (simulates Devign / CodeSearchNet)
    VOCAB = ["path", "filename", "file_path", "data_file", "load", "read",
             "write", "close", "open", "init", "Token_TYPE", "safe_input",
             "error", "buffer", "api_key", "config", "parse", "validate"]

    data = [
        CodeSample("def check_bounds(path): return validate(path)", 0, "defect_detection"),
        CodeSample("def load_config(filename): return parse(filename)", 1, "defect_detection"),
        CodeSample("def safe_input(Token_TYPE): return Token_TYPE", 0, "defect_detection"),
        CodeSample("def get_data(data_file): return load(data_file)", 1, "defect_detection"),
        CodeSample("def read_file(path): return open(path).read()", 0, "defect_detection"),
        CodeSample("def write_output(path, data): open(path, 'w').write(data)", 1, "defect_detection"),
        CodeSample("def init_config(config): return parse(config)", 0, "defect_detection"),
        CodeSample("def load_file(filename): return open(filename)", 1, "defect_detection"),
    ] * 10  # replicate for demo

    model = MockCodeLM(task="defect_detection")

    # Step 1: Dataset bias analysis
    print("\n[Step 1] Dataset bias analysis (z-score per token)")
    analyzer = DatasetBiasAnalyzer()
    biased = analyzer.analyze(data, target_label=1)
    print(f"  Biased tokens (z > 3) for label=1 (defective):")
    for bt in biased[:5]:
        print(f"    '{bt.token}': z={bt.z_score:.2f}")
    if not biased:
        print("  (no high-z tokens found in mock data — use real dataset for meaningful results)")

    # Step 2: Trigger inversion
    print("\n[Step 2] Trigger inversion (finding natural backdoor triggers)")
    inverter = TriggerInverter(model, VOCAB)
    trigger = inverter.invert(data, target_label=1, trigger_len=3)
    print(f"  Inverted trigger: {trigger.tokens}")
    print(f"  ASR against label=1: {trigger.asr:.1%}")
    if trigger.asr > 0.2:
        print(f"  WARNING: ASR > 20% — natural backdoor confirmed")
    else:
        print(f"  ASR below threshold — no backdoor detected for this target")

    # Step 3: ScanNBT (diverse multi-seed detection)
    print("\n[Step 3] ScanNBT — diverse trigger detection (5 seeds)")
    scanner = ScanNBT(inverter, n_seeds=5)
    all_triggers = scanner.scan(data, target_label=1)
    d1, d2 = scanner.diversity_score(all_triggers)
    print(f"  Found {len(all_triggers)} trigger variants")
    for i, t in enumerate(all_triggers):
        print(f"    seed {i}: {t.tokens}  ASR={t.asr:.1%}")
    print(f"  Distinct-1={d1:.3f}  Distinct-2={d2:.3f}")
    print(f"  (paper target: Distinct-2 ≈ 1.0 for diverse coverage)")

    # Step 4: Unlearning defense
    print("\n[Step 4] Unlearning-based defense")
    asr_before = trigger.asr
    defense = UnlearningDefense(model)
    defense.unlearn(data, trigger, n_steps=30)
    asr_after = inverter.invert(data, target_label=1).asr
    print(f"  ASR before unlearning: {asr_before:.1%}")
    print(f"  ASR after unlearning:  {asr_after:.1%}")
    print(f"  (paper result: 68.1% → 1.2% for UniXcoder)")

    # Step 5: Code search rank bias demo
    code_search_rank_demo(model)

    # Summary
    print("\n" + "=" * 60)
    print("Key findings (from paper, not mock):")
    print("  Natural backdoors: present in 100% of 44 tested scenarios")
    print("  UniXcoder defect detection ASR: 68.06% (no poisoning needed)")
    print("  Code search: `path` → `filename` raises insecure snippet from rank 6 to 2")
    print("  Only effective defense: unlearning-based post-training (68.1% → 1.2%)")
    print("  Root cause: dataset token z-scores > 3 (at least 1 per inverted trigger)")
