---
title: "Every Picture Tells a Dangerous Story: Memory-Augmented Multi-Agent Jailbreak Attacks on VLMs"
arxiv_id: "2604.12616"
date: "2026-06-15"
authors: "Jianhao Chen, Haoyang Chen, Hanjie Zhao, Haozhe Liang, Tieyun Qian"
institution: "Wuhan University; Tianjin University; University of Chinese Academy of Sciences; Zhongguancun Academy"
tags: ["VLM", "jailbreak", "adversarial-attack", "multimodal", "red-teaming", "safety", "memory"]
---

# MemJack: Memory-Augmented Multi-Agent Jailbreak Attacks on VLMs

> **Any unmodified natural image is a potential jailbreak anchor. MemJack achieves 71.48% ASR against Qwen3-VL-Plus on 5,000 unmodified COCO images with no pixel perturbation — scaling to 90% under extended budget. Memory accumulation drives cross-image strategy transfer; removing it drops ASR from 72% to 38%.**

| Field | Value |
|-------|-------|
| Authors | Jianhao Chen, Haoyang Chen, Hanjie Zhao, Haozhe Liang, Tieyun Qian |
| Institution | Wuhan University; Tianjin University; UCAS; Zhongguancun Academy |
| arXiv | [2604.12616](https://arxiv.org/abs/2604.12616) |
| Date | April 2026 |
| Tags | VLM, jailbreak, adversarial-attack, multimodal, red-teaming, safety |

---

## The Problem

Current VLM jailbreak defenses catch: (1) pixel perturbations, (2) typographic embeddings detectable by OCR, (3) overtly harmful images. All three modify the image or use pre-crafted visual content.

The attack surface they miss: the **semantic relationship** between ordinary visual content and what the model can be asked to reason about. A kitchen scene with a knife is a legitimate image — but also an anchor for a harmful use-case query that appears to be legitimate visual analysis.

---

## The Approach

MemJack uses unmodified natural images as attack vectors through a three-stage multi-agent pipeline with persistent cross-image memory.

```
Stage 1 — Strategic Planning Agent
  Input: image I, safety policy C
  Output: ranked visual anchors (a_j, type_j, attack_goal_j, confidence_j)
  → Four anchor priority levels: direct, scenario-based, social/psychological, relational
  → Realism constraint: discards abstract over-symbolization

Stage 2 — Iterative Attack Agent
  Six camouflage angles:
    α1: Visual Intuitive Association   (28.4% of attempts)
    α2: Scenario Story Extension       (4.9%)
    α3: First-Person Role Perspective  (8.6%)
    α4: Hypothetical Reasoning         (23.2%)
    α5: Practical Knowledge Exploration (23.6%)
    α6: Contextual Dialogue            (2.8%)
  + MCTS/evolutionary refinement
  + INLP null-space filter: pre-screens prompts for refusal direction
    (linear SVM on embeddings achieves 83.8% safe/unsafe separation)

Stage 3 — Evaluation & Feedback Agent
  Safety Guard: scores response r ∈ [0,1] (Safe/Controversial/Unsafe)
  Reflection: classifies defense pattern, recommends next angle
  Replanning: fires when all angles exhausted → new anchor from Stage 1

Experience-Driven Memory (across images):
  - Multimodal Experience Memory: FAISS-indexed (visual + goal + strategy)
    TD-learning update: Q_i ← Q_i + β(r_t - Q_i), β=0.2
    6.2× reuse ratio after 5,000 images
  - Jailbreak Knowledge Graph: G_KG = (Anchors, Goals, Strategies, Defenses, Categories)
    Edge weights = success / (success + failure) counts
    Provides MCTS priors and bypass recommendations
```

---

## Results

### Primary evaluation (COCO val2017, 5,000 unmodified images)

| Setting | ASR | Avg Rounds to Success |
|---------|-----|----------------------|
| R=20 rounds | 71.48% | 5.18 |
| R=100 rounds (100-image subset) | 90.00% | 9.72 |
| Success within 6 rounds | 68.3% of successes | — |
| Success within 10 rounds | 89.1% of successes | — |

### Cross-model ASR (100-image COCO subset, R=20)

| Model | Type | ASR |
|-------|------|-----|
| Mistral-Medium-3 | API | 82% |
| Qwen3-VL-Plus | API | 72% |
| Llama-3.2-11B-Vision | Local | 68% |
| GLM-4.6-V-Flash | Local | 63% |
| Qwen3-VL-8B-Instruct | Local | 53% |
| Gemini-3-Flash | API | 35% |

### Baseline comparison (100-image COCO subset, black-box)

| Method | Access | ASR |
|--------|--------|-----|
| QR-Attack | Black-box | 1% |
| FigStep | Black-box | 13% |
| HADES | Black-box | 10% |
| Visual-Adv | White-box | 17% |
| AutoDAN-Turbo | White-box | 30% |
| **MemJack** | **Black-box** | **72%** |

### Ablation (100-image COCO subset)

| Variant | ASR | Avg Rounds |
|---------|-----|-----------|
| w/o Memory | 38% | 9.11 |
| w/o Reflection | 67% | 6.19 |
| w/o Replanning | 66% | 6.27 |
| Full MemJack | 72% | 5.38 |

### Defense pattern distribution (19,105 rounds)

| Defense Pattern | Frequency | Resistance |
|----------------|-----------|-----------|
| Direct refusal | 41.4% | Low — provides clear signal to switch angle |
| Safe answer | 27.5% | Medium |
| Benign reframing | 20.8% | High — no explicit refusal signal generated |
| Uncategorized | 9.8% | — |

---

## Key Insight

**Benign reframing is the hardest defense to defeat.** When the model proactively reinterprets a harmful request as innocent and answers that version, MemJack receives no gradient signal to follow. Direct refusal is the easiest defense to bypass — it tells the attacker exactly what failed.

**Memory accumulation changes the cost curve.** As the system processes more images, local ASR rises while rounds-to-success falls. The 6.2× reuse ratio means successful strategies transfer across visually diverse images — most anchors (knife, chemical, vehicle) appear repeatedly across the dataset with consistent attack patterns.

---

## Builder/Defender Takeaway

**For VLM product teams:**
1. Build models that redirect queries to benign interpretations rather than refusing explicitly — refusal is a high-signal failure mode for adversarial attackers
2. Monitor multi-turn conversation patterns where visual objects are gradually escalated into harmful contexts
3. Scene-level input risk scoring is tractable: office workspace, street, traffic scenes carry the highest jailbreak risk (81%, 61%, 60% of successful attacks respectively); domestic/pet scenes are lowest
4. INLP null-space filtering at the embedding layer can pre-screen prompts before they reach the model — linear separability of safe/unsafe inputs is 83.8% in tested embeddings

**For red-teamers:**
- MemJack-Bench (113,000+ interactive trajectories) will be released for defensive alignment research
- Scene-category interaction heatmap (Figure 6) identifies highest-value attack surfaces per deployment domain

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/memjack_defense.py](scripts/memjack_defense.py) | Scene risk scorer, visual anchor detector heuristics, and benign-reframing vs refusal defense evaluation stub |
