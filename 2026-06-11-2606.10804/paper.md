---
title: "SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning"
arxiv_id: "2606.10804"
date: "2026-06-11"
authors: "Wenhao Yan, Fengjia Guo, Zhuoyi Yang, Jie Tang"
institution: "Tsinghua University; Z.ai"
tags: ["video-generation", "character-animation", "diffusion", "in-context-conditioning", "DPO", "RoPE", "motion-transfer"]
---

# SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning

> **Removes pose skeleton intermediates entirely — concatenates the driving video directly as context — and unifies animation + replacement in one model. FVD 287.11 vs SCAIL's 309.63; 93.3% win on identity isolation in multi-character scenarios.**

| Field | Value |
|-------|-------|
| Authors | Wenhao Yan, Fengjia Guo, Zhuoyi Yang, Jie Tang |
| Institution | Tsinghua University; Z.ai |
| arXiv | [2606.10804](https://arxiv.org/abs/2606.10804) |
| Code | https://github.com/zai-org/SCAIL-2 |
| Date | June 10, 2026 |
| Tags | video-generation, character-animation, diffusion, DPO, RoPE |

---

## The Problem

Character animation models extract pose skeletons from driving videos and feed those skeletons to a video diffusion model. Skeletons lose information: they can't represent occlusions, multi-character interactions with depth ambiguity, animal motion, or object-character contact. You can't animate what you can't represent in the intermediate. And each sub-task (animation, replacement, multi-character) needs a separate training regime because they differ in what intermediate they use.

---

## The Idea

Bypass intermediates entirely. Concatenate the driving video directly into the diffusion context sequence alongside the reference image. The model extracts what it needs from raw pixels.

```
Input context to diffusion transformer:
  [zref ; zt ; zdriv]
  │       │     └── driving video, VAE-encoded, spatial offset ΔW
  │       └──────── noisy video latent to denoise
  └──────────────── reference character image, VAE-encoded

+ 28-channel binary mask latent (concatenated to context)
  = 7 color channels × 4 temporal stride
  = 1 env-switch channel + 6 character binding slots
  Each color assigns spatial regions to a specific character stream.

+ Mode-Specific Shifted RoPE
  Animation:   zref t=0,  zt t=1..Tv  → temporal separation
  Replacement: zref h=ΔH, zt h=0..Hv  → spatial separation

Reverse driving:
  G(y, I) → ỹ (synthetic)   ỹ used as driving, real y as supervised target
  → model trained toward real video quality, not synthetic artifacts
```

Post-training: Bias-Aware DPO targets finger articulation, where pose estimators fail most.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Wan2.1-14B-I2V backbone | Base I2V diffusion transformer; receives concat sequence [zref; zt; zdriv] |
| In-context mask conditioning | 28-channel binary latent (7 colors × 4 temporal stride) concatenated to context; Ch0 = env switch, Ch1-6 = character binding slots |
| Mode-Specific Shifted RoPE | Animation mode: temporal shift for ref (t=0) vs video (t=1..Tv); Replacement mode: spatial shift ΔH for ref; prevents training conflicts between modes |
| Reverse driving paradigm | Synthetic ỹ drives; real y supervises — avoids propagating generator artifacts into targets |
| Agentic synthetic loop | Candidate Selector → Prompt Weaver → Pose Editor → Quality Checker loop to build MotionPair-60K with <30% discard rate |
| Bias-Aware DPO | Preference pairs: r (1 error round) vs r⁻ (2 error rounds); optimizes fine-grained finger articulation; 400 training steps post-pretrain |

---

## Results

### Single-character animation — Studio-Bench (quantitative, pose-driven partition)

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ |
|--------|-------|-------|--------|------|
| SCAIL + SAM3D-Body | 0.6407 | 19.08 | 0.2212 | 309.63 |
| Wan-Animate | 0.6340 | 18.62 | 0.2269 | 305.31 |
| SteadyDancer | 0.6386 | 18.40 | 0.2311 | 332.20 |
| **SCAIL-2 + SAM3D-Body** | **0.6453** | **19.09** | **0.2231** | **287.11** |

### Single-character animation — Studio-Bench (human evaluation win rates)

| Metric | vs SCAIL | vs Kling 3.0 |
|--------|----------|-------------|
| Motion Consistency | 68.3% win | 65.0% win |
| Physical Plausibility | 68.3% win | 46.7% win |
| Identity Consistency | 46.7% win | 35.0% win |

### Multi-character animation — zero-shot (human evaluation)

| Metric | vs SCAIL | vs MultiAnimate |
|--------|----------|----------------|
| Motion Consistency | 50.0% win | 76.7% win |
| Identity Isolation | 56.7% win | 93.3% win |
| Identity Consistency | 26.7% win | 93.3% win |

### Character replacement — human evaluation

| Metric | vs MoCha | vs Wan-Animate |
|--------|----------|---------------|
| Motion Consistency | 57.1% win | 71.4% win |
| Environment Integration | 67.9% win | 67.9% win |
| Identity Consistency | 66.7% win | 73.3% win |

### Automatic video quality (X-Dance, Video-Bench)

| Method | Imaging Quality | Motion Smoothness | Temporal Consistency | Appearance Consistency |
|--------|----------------|------------------|---------------------|----------------------|
| SCAIL | 4.27 | 3.90 | 4.21 | 4.25 |
| **SCAIL-2** | **4.43** | **3.89** | **4.18** | **4.38** |

---

## Key Insight

The model trained on multi-character replacement data acquires identity isolation (knowing which motion belongs to which character) as a byproduct. Then that capability transfers zero-shot to multi-character animation, which has no dedicated training data. The binding slots and reverse driving paradigm together produce emergent generalization: the model generalizes to compositional tasks it was never directly trained on.

---

## Builder Takeaway

The mask compression scheme is directly portable to any DiT-based video model: encode character regions as 7 binary channels (1 env + K binding slots), stack temporally to match VAE stride, concatenate to the context sequence. No architectural surgery — just channel concatenation.

The reverse driving paradigm is the key training insight: generate synthetic driving videos from real targets, then train with the synthetic video as input and the real video as supervision. You avoid propagating any generator's artifacts into your training targets.

Bias-Aware DPO is worth noting as a targeted post-training strategy: identify the specific failure mode (finger articulation), construct preference pairs by deliberately introducing more error rounds for negatives, run 400 DPO steps. It improves detail without touching the main training loop.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/scail2_components.py](scripts/scail2_components.py) | Mask compression (RGB→28ch), mode-specific RoPE coordinates, Bias-Aware DPO preference construction — all run with numpy+PIL, tested on st3ve |
