---
title: "SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning"
arxiv_id: "2606.13673"
date: "2026-06-12"
authors: "Seokju Cho, Ryo Hachiuma, Abhishek Badki, Hang Su, Byung-Kwan Lee, Chan Hee Song, Sifei Liu, Subhashree Radhakrishnan, Seungryong Kim, Yu-Chiang Frank Wang, Min-Hung Chen"
institution: "NVIDIA; KAIST"
tags: ["spatial-reasoning", "tool-augmented-agents", "code-as-action", "VLM", "video-understanding", "persistent-kernel"]
---

# SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning

> **The bottleneck in spatial reasoning agents is the action interface, not the tool catalog. A persistent Python kernel with one cell per step outperforms structured tool-call and single-pass code by +11.2 pp across 20 benchmarks — without any benchmark- or model-specific adaptation.**

| Field | Value |
|-------|-------|
| Authors | Seokju Cho, Ryo Hachiuma, Abhishek Badki, et al. |
| Institution | NVIDIA; KAIST |
| arXiv | [2606.13673](https://arxiv.org/abs/2606.13673) |
| Code | https://github.com/NVlabs/SpatialClaw |
| Date | June 11, 2026 |
| Tags | spatial-reasoning, tool-augmented-agents, code-as-action, VLM, video |

---

## The Problem

Tool-augmented VLMs for spatial reasoning are limited by the interface through which tools are invoked:

- **Single-pass code**: the agent writes a complete program before execution, committing to a strategy before seeing any intermediate mask, depth map, or error. Mid-course correction is impossible.
- **Structured tool-call (JSON/XML)**: exposes perception tools through predefined commands. Compositions that emerge at test time (KDTree nearest-neighbor, RANSAC plane fit) can't be expressed in a fixed schema.

Both designs commit the agent before the evidence arrives.

---

## The Idea

Use code itself as the action interface inside a persistent kernel.

```
For each example:
  Init persistent Python kernel with:
    InputImages      — sampled frames
    tools.Reconstruct (Depth Anything 3)   — depth, intrinsics, extrinsics, point maps
    tools.SAM3       — segmentation from text/point/box prompts
    show(...)        — registers images into next agent context
    vlm.ask(...)     — side VLM session for commonsense/grounding
    numpy, scipy, matplotlib

Loop (max N_max=30 steps):
  Stage I:   Planner (no images) → analysis plan
  Stage II:  VLM agent → Python cell
  Stage III: AST security check → execute in persistent kernel
  Stage IV:  Feedback assembly (stdout + variable summaries + show() images)
  Stage V:   If ReturnAnswer() called → terminate
```

The agent sees intermediate masks, depth maps, and plots before deciding what to compute next. Any object produced in step N is available as a Python variable in step N+1.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Persistent kernel | Single Python process per sample; all variables survive across cells |
| tools.Reconstruct (DA3) | Depth Anything 3 — returns depth, camera intrinsics/extrinsics, dense point maps |
| tools.SAM3 | SAM3 video segmentation — masks from text, point, or box prompts |
| show() | Registers intermediate images (masks, depth, plots) into next agent context |
| vlm.locate / vlm.ask_with_thinking | Side VLM sessions for grounding and commonsense without polluting main context |
| Planner | Separate LLM session, no images, produces analysis plan before main loop |
| AST security check | Static parse before execution; rejects unsafe builtins/modules without running |
| Unified system prompt | General spatial reasoning discipline; no benchmark- or task-specific instructions |

---

## Results

### Action interface comparison (Gemma4-31B, 20 benchmarks)

| Interface | Avg Accuracy |
|-----------|-------------|
| No-tool baseline | 53.4% |
| Single-Pass Code | 55.2% |
| Structured Tool-Call | 56.7% |
| **SpatialClaw (persistent kernel)** | **59.9%** |

### Comparison with other spatial agents (Gemma4-31B backbone)

| Method | Interface | Avg (20 bench) |
|--------|-----------|---------------|
| VADAR | Single-pass code | ~33 (video N/A) |
| pySpatial | Single-pass code | 47.8% |
| SpaceTools Toolshed | Structured tool-call | 48.7% |
| **SpatialClaw** | Persistent kernel | **59.9%** |

### By backbone (SpatialClaw avg accuracy over 20 benchmarks)

| Backbone | No-tool | SpatialClaw | Gain |
|----------|---------|-------------|------|
| Qwen3.5-397B-A17B | 57.3% | 60.4% | +3.1 pp |
| Qwen3.5-122B-A10B | 53.7% | 56.9% | +3.2 pp |
| Qwen3.6-35B-A3B | 52.6% | 57.2% | +4.6 pp |
| Qwen3.6-27B | 55.0% | 62.7% | +7.7 pp |
| Gemma4-31B | 53.4% | 59.9% | +6.5 pp |
| Gemma4-26B-A4B | 48.0% | 54.3% | +6.3 pp |

### Largest gains by category (avg over all backbones)

| Category | Avg Gain vs No-tool |
|----------|-------------------|
| DSI-Bench (video 4D) | +18.3 pp |
| MindCube (multi-view) | +14.3 pp |
| MMSI (multi-view) | +13.0 pp |
| PAI-Bench (video) | +10.8 pp |

---

## Key Insight

Removing all pre-defined utility wrappers (tools.Mask, tools.Geometry), leaving only SAM3/DA3 and scipy, drops performance by less than 0.5 pp. The action interface — persistent state, per-cell execution, intermediate visual feedback — drives the gains, not the tool catalog. The agent constructs needed geometry (KDTree for distances, dot products for directions) directly from task semantics without any routing prompt.

---

## Builder Takeaway

The action interface matters more than the tool catalog. A persistent Python kernel where each step conditions on prior outputs outperforms a structured API across all tested spatial reasoning categories.

Minimal setup: expose perception modules as Python callables, capture show() output as images in next context, run an AST check before execution, use a separate planner to front-load analysis structure. No benchmark-specific engineering needed.

The same configuration transfers across model families (Qwen3.5/3.6, Gemma4) and parameter scales (27B–397B) without modification.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/spatialclaw_kernel.py](scripts/spatialclaw_kernel.py) | Persistent kernel scaffold, feedback assembly, AST safety check, and agentic loop stub — runnable standalone with mock tools |
