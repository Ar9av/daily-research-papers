---
title: "Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?"
arxiv_id: "2605.30557"
date: "2026-06-07"
authors: "Yue Zhang, Zun Wang, Han Lin, Yonatan Bitton, Idan Szpektor, Mohit Bansal"
institution: "UNC Chapel Hill; Google Research"
tags: ["VLM", "spatial-reasoning", "abstention", "uncertainty", "benchmark", "embodied-AI"]
---

# Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?

> **VLMs answer confidently when they should abstain — under perspective ambiguity, all tested models score below random at recognizing unanswerable cases, and adding visual input makes them worse.**

| Field | Value |
|-------|-------|
| Authors | Yue Zhang, Zun Wang, Han Lin, Yonatan Bitton, Idan Szpektor, Mohit Bansal |
| Institution | UNC Chapel Hill, Google Research |
| arXiv | [2605.30557](https://arxiv.org/abs/2605.30557) |
| Website | https://spatialuncertain.github.io |
| Date | May 28, 2026 |
| Tags | VLM, spatial-reasoning, abstention, uncertainty, benchmark |

---

## The Problem

Every spatial reasoning benchmark assumes visual observations are complete and reliable. But real 3D environments have occlusion (objects hidden by other objects) and perspective distortion (geometry that looks different from different viewpoints). When models encounter these conditions, the right answer is "cannot determine" — but current VLMs don't do that. They guess, confidently, and incorrectly. No existing benchmark measures this gap.

---

## The Idea

SPATIALUNCERTAIN builds controlled 3D indoor scenes (240 scenes, 43 room types) and introduces two observation challenges, then asks: can models recognize when a spatial question is unanswerable?

```
Clean 3D scene (all questions answerable)
        ↓               ↓
   OCCLUSION      PERSPECTIVE AMBIGUITY
   (missing        (misleading info —
   information)    visual cues lie)
        ↓               ↓
  Correct behavior: abstain with "Cannot determine"
        ↓
  Secondary task: ViewSel / AbstainViewSel
  → which of 5 candidate views would let you answer?
```

Four question types: Visibility, Relative Position, Depth Ordering, Size/Shape.
Answerability varies by question type and observation condition — e.g., visibility questions remain answerable even under full occlusion; size/shape questions become unanswerable under perspective ambiguity.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| 3D Scene Generation | Holodeck (LLM-based layout) + AI2-THOR rendering; 240 scenes, 10,322 QA pairs |
| Occlusion Configuration | Occluder placed on camera–target line-of-sight; partial and full occlusion cases; human-validated |
| Perspective Configuration | Camera shifted laterally → same-size objects appear different sizes; reference vs. ambiguous views |
| ViewSel Task | Single-stage: given 5 candidate views, pick the one that best supports answering |
| AbstainViewSel Task | Two-stage: first abstain correctly, then identify the informative view — both must be correct |
| Human Validation | Annotators verify occlusion level and perspective quality; invalid configs discarded |

---

## Results

| Model | Occ-Ans | Occ-Unans | Pers-Ans | Pers-Unans | ViewSel | AbstainViewSel |
|-------|---------|-----------|----------|------------|---------|----------------|
| Random baseline | 32.3 | 23.3 | 25.0 | 25.9 | 25.0 | 20.0 |
| Qwen2.5-VL-7B | 51.1 | 39.3 | 62.4 | 41.5 | 24.6 | 4.0 |
| Qwen2.5-VL-32B | 51.7 | 40.0 | 69.0 | 21.7 | 20.7 | 8.6 |
| InternVL3-8B | 61.7 | 7.3 | 70.4 | 1.1 | 18.5 | 4.6 |
| GPT-4o | 53.9 | 32.8 | 35.2 | 36.3 | 39.3 | 22.1 |
| GPT-5-mini | 64.7 | 7.8 | 76.1 | 15.2 | 53.7 | 18.0 |
| GPT-5.4 | 58.2 | 19.5 | 69.5 | 22.6 | **70.9** | 22.6 |
| Gemini-2.5-Flash | 56.1 | 45.0 | 66.4 | 2.4 | 18.5 | 6.7 |
| Gemini-3.0-Flash | 61.7 | 44.1 | 64.0 | 6.3 | 50.3 | 2.4 |

Visual input effect on unanswerable cases (T vs T+V):
- GPT-5.4 Pers-Unans: 44.3 (text-only) → 22.6 (with image) **-21.7pp**
- Gemini-3.0-Flash Pers-Unans: 42.1 → 6.3 **-35.8pp**

Fine-tuning (Qwen2.5-VL-7B LoRA, rank 16):

| Variant | Occ-Unans | Pers-Unans |
|---------|-----------|------------|
| Base | 41.0 | 42.9 |
| LoRA-Occ | 39.3 | 38.5 (↓) |
| LoRA-Pers | 7.7 (↓) | 86.8 (↑) |
| **LoRA-Mixed** | **62.8 (↑)** | **76.9 (↑)** |

---

## Key Insight

Under perspective ambiguity, adding visual input makes all tested models *worse* at recognizing unanswerable cases — because the misleading visual cues actively suppress abstention. Models aren't failing to see; they're seeing something that confidently lies to them, and they believe it. This asymmetry (visual input helps under occlusion, hurts under perspective ambiguity) reveals that current VLMs have no notion of *observation reliability* — they treat all visual input as trustworthy by default.

---

## Builder Takeaway

If you're building vision agents for real-world environments (robotics, AR, embodied assistants), your model needs an explicit "I can't reliably see this" output. Right now none of the frontier models have it reliably. The fix isn't prompting (introduces accuracy trade-off) — it's fine-tuning on diverse ambiguity types. LoRA-Mixed shows the abstention–accuracy trade-off can be resolved with the right training signal. Until then: treat any VLM spatial answer from a single viewpoint as provisional, and architect your system to request alternative views before acting.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/spatial_uncertain.py](scripts/spatial_uncertain.py) | ObservationReliabilityEvaluator: abstention scoring, viewpoint selection, answer–abstain trade-off analysis |
