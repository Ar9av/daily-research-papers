---
title: "Securing Code Understanding: Detecting Natural Backdoor Vulnerability in Code Language Models"
arxiv_id: "2606.10846"
date: "2026-06-10"
authors: "Yuchen Chen, Weisong Sun, Haocheng Huang, Yuan Xiao, Chunrong Fang, Yiran Zhang, Tingting Xu, Zhenpeng Chen, An Guo, Peizhuo Lv, Xiaofang Zhang, Zhenyu Chen, Yang Liu, Baowen Xu"
institution: "Nanjing University; Nanyang Technological University; Soochow University; Tsinghua University"
tags: ["security", "backdoor", "code-llm", "code-search", "defect-detection", "dataset-bias", "model-security"]
---

# Securing Code Understanding: Detecting Natural Backdoor Vulnerability in Code Language Models

> **Normally trained code models develop hidden triggers from dataset biases — no attacker needed. Replacing one variable name can flip defect detection or surface an insecure snippet in code search.**

| Field | Value |
|-------|-------|
| Authors | Chen, Sun, Huang, Xiao, Fang, Zhang, Xu, Chen, Guo, Lv, Zhang, Chen, Liu, Xu |
| Institution | Nanjing University; NTU; Soochow University; Tsinghua University |
| arXiv | [2606.10846](https://arxiv.org/abs/2606.10846) |
| Code | Available in paper repository (link in paper) |
| Date | June 9, 2026 |
| Tags | security, backdoor, code-llm, code-search, defect-detection, dataset-bias |

---

## The Problem

Backdoor attacks in code models are studied as adversarial threats — someone poisons the training data, embeds a trigger, and the model behaves as a weapon. But the same phenomenon arises spontaneously in clean, normally trained models. Statistical biases in training data create spurious correlations between token patterns and labels. These become implicit triggers. Nobody inserted them. They exist in every standard CodeLM fine-tuning run.

---

## The Idea

Use trigger inversion (the same reverse-engineering technique developed for injected backdoors) to find natural backdoor triggers in clean models. For a target label, find the token sequence that minimizes the loss of predicting that label across clean inputs. If such a sequence exists and achieves non-trivial ASR, the model has a natural backdoor.

```
Normally trained CodeLM (clean data, standard SGD)
        ↓
Trigger inversion (EliBadCode / GCG optimization)
  for each target label y_t:
    find token sequence t* that minimizes:
    E[CrossEntropy(f(x ⊕ t*), y_t)] over clean inputs x
        ↓
If ASR(t*) >> random baseline → natural backdoor confirmed
        ↓
Characterize: prevalence, representation-space behavior,
              transferability, root cause, defense effectiveness
        ↓
ScanNBT: improved multi-trigger detection method
```

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Trigger inversion (EliBadCode + GCG) | Reverse-engineers implicit triggers via gradient-based token search over model vocabulary; no access to original training pipeline needed |
| Natural backdoor scanner | Iterates over all target labels; declares natural backdoor if inverted trigger ASR significantly exceeds random-token baseline |
| t-SNE representation analysis | Visualizes whether triggered samples cluster separately from clean samples in self-attention layers (injected: yes; natural: no) |
| z-score dataset bias analysis | Computes per-token association statistic; flags tokens with z > 3 as disproportionately correlated with target label |
| Causal intervention | Ablates individual trigger tokens; confirms high-z tokens play a causal role in activating backdoor behavior |
| ScanNBT | Multi-seed diverse trigger detection; improves Distinct-2 to ~1.0 while maintaining comparable ASR/ANR to EliBadCode |

---

## Results

### Prevalence (RQ1) — 44 scenarios across 5 model families

| Model | Task | ASR / ANR | Risk |
|-------|------|-----------|------|
| UniXcoder | Defect detection | 68.06% ASR | High |
| CodeBERT | Defect detection | 37.65% avg ASR | Moderate |
| CodeBERT | Code search | 27.23% ANR | High (lower is worse) |
| StarCoder | Code summarization | 5.40% avg ASR | Present |
| DeepSeek-Coder-1.3B | Code summarization | 7.10% avg ASR | Present |

Code understanding tasks (defect detection, code search) show higher risk than code generation.
Large-scale models do NOT eliminate natural backdoors.

### Representation space (RQ2)

| Property | Injected backdoor | Natural backdoor |
|---------|------------------|-----------------|
| L2 distance (layers 8–12) | 5.79–7.82 | 1.88–2.82 |
| Cosine similarity (layers 8–12) | 0.97–0.98 | 1.00 |
| Cluster separation in t-SNE | Clear clusters form in deep layers | Fully entangled with clean samples |

Natural triggers are more covert — standard representation-based detection fails.

### Transferability (RQ3)

- Same fine-tuning dataset → transfers across all 3 architectures
- Same architecture, different dataset → transfers (weakened)
- Knowledge distillation (GPT-3.5 → 350M student) → 12% ASR transfers back to GPT-3.5

### Defense effectiveness (RQ5)

| Defense | Works? | UniXcoder Defect ASR |
|---------|--------|---------------------|
| Activation Clustering | Partially | 28.7% (from 68.1%) |
| KillBadCode | Inconsistent | 62.5% |
| DeCE | No | 68.4% |
| CodePurify | Partially | 29.1% |
| **Unlearning-based (post-training)** | **Yes** | **1.2%** |

Only unlearning-based post-training defense consistently works across tasks and models.

---

## Key Insight

Replacing `path` with `filename` in a code snippet moves a hardcoded-secret function from rank 6 to rank 2 in CodeBERT's code search. `filename` has a z-score of 6.50 in the training corpus — it co-occurs disproportionately with "file" queries. That statistical pattern became a retrieval shortcut. No attacker was involved. The same vulnerability transfers to any CodeLM fine-tuned on the same dataset.

---

## Builder Takeaway

Audit your code models before deployment with trigger inversion. Run EliBadCode or ScanNBT on the fine-tuned model against your task's target labels. If any inverted trigger achieves >20% ASR, run unlearning-based mitigation on it. This is especially critical for: defect detection pipelines feeding CI/CD (false negatives hide vulnerabilities), code search tools used for code reuse (they can surface insecure patterns), and code generation models distilled from larger LLMs (backdoors transfer via distillation).

The root cause — z-score > 3 dataset tokens — gives you a cheaper pre-check: compute per-token association statistics in your fine-tuning dataset before training. Remove or downsample samples dominated by high-z tokens correlated with safety-critical labels.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/natural_backdoor_scanner.py](scripts/natural_backdoor_scanner.py) | Natural backdoor trigger inversion, z-score dataset bias analysis, unlearning-based mitigation |
