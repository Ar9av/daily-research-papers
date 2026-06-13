---
name: rag-reasoning-cost-defense
description: Detect and mitigate overthinking attacks on RAG systems with reasoning models. Monitor per-query reasoning token counts as an anomaly signal, scan retrieved documents for logical meta-constraint patterns before context injection, and enforce per-query token ceilings to bound inference cost.
trigger: When deploying a RAG system backed by a reasoning model (DeepSeek-R1, o1/o3, Qwen-thinking); when you need to bound inference cost against adversarial knowledge-base manipulation; when token monitoring is missing from your RAG observability stack.
---

## When to use

- Your RAG pipeline uses a chain-of-thought or extended-thinking reasoning model
- You accept user-submitted or third-party content into your retrieval corpus (public wikis, uploaded docs, web-crawled content)
- You want observability over inference cost — not just output accuracy — as a security signal
- You're building a rate-limited or cost-sensitive RAG product where token inflation is a denial-of-service vector

## Pattern

1. **Baseline token profiling**: record per-query reasoning token counts over a representative warm-up period; compute per-user and per-query-type percentile distributions
2. **Anomaly flagging**: flag any query whose reasoning token count exceeds N× the per-category baseline (N=5 is a practical starting point)
3. **Document co-occurrence tracking**: for high-token queries, record which retrieved document IDs co-occurred; a document that appears in >K% of flagged queries is a poisoning candidate
4. **Retrieval-time contradiction scan**: before retrieved documents enter the reasoning context, scan for explicit logical meta-constraints; quarantine or strip flagged passages
5. **Token ceiling enforcement**: cap max reasoning tokens per query at a configurable limit; re-run without the highest-scoring retrieved document if the limit is hit

## Implementation

```python
# See scripts/code_defense.py for full runnable implementation

class TokenAnomalyDetector:
    def record(self, query_id, category, token_count, doc_ids): ...
    def is_anomalous(self, query_id, category, token_count) -> bool: ...
    def get_suspicious_docs(self, top_k=10) -> list[str]: ...

class ContradictionScanner:
    def scan(self, text: str) -> list[dict]: ...
    # returns list of {pattern, span, severity}

def enforce_token_ceiling(query, retrieved_docs, reasoning_fn, max_tokens=8000):
    # try full doc set first; if over ceiling, retry without highest-suspect doc
    ...
```

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Anomaly threshold multiplier | 5× | Flag queries using >5× category baseline |
| Co-occurrence poisoning threshold | 10% | Doc seen in >10% of flagged queries |
| Token ceiling | 8,000 | Reasonable cap for most commercial reasoning models |
| Contradiction scan patterns | 7 regex | Covers "exactly N of", "one of the following is false", etc. |
| Style adapter operators targeted | 5 | SU, RV, NI, AU, NR — look for audit/regulatory register |

## Pitfalls

| Approach | Problem | Better |
|----------|---------|--------|
| Monitor output accuracy only | Attack keeps accuracy flat; invisible | Monitor reasoning token counts too |
| Reject documents with any uncertainty language | Too broad; most documents have hedging | Scan for *explicit logical meta-constraints* (exactly N true/false) |
| Hard cutoff on token count | Legitimate complex queries hit ceiling | Flag + retry without suspect doc first |
| One-shot retrieval | Poisoned doc always in top-k if well-crafted | Track doc co-occurrence with high-cost queries across sessions |
| Prompt-only defense (CCoT/CoD) | Reduces but doesn't stop contradiction-induced loops | Combine with retrieval-layer filtering |

## Key numbers

- 5.32× to 24.72× reasoning token amplification (token-level, single poisoned doc)
- 12.698× to 43.451× task-level amplification
- 100% retrieval hit rate for adversarial documents under tested configuration
- TrustRAG (best defense): reduces to 5.30× for DS R1, still above baseline
- Qwen-Plus highest vulnerability: 55,665 tokens vs 2,252 baseline
- N=4 contradictions increase amplification at slight accuracy cost vs N=3

## Source

- arXiv: https://arxiv.org/abs/2601.13112
