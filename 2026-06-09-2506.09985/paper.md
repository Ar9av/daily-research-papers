---
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
arxiv_id: "2506.09985"
date: "2026-06-09"
authors: "Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas, et al."
institution: "FAIR at Meta; Mila / Polytechnique Montréal"
tags: ["world-models", "self-supervised", "video-pretraining", "robot-planning", "JEPA", "latent-space"]
---

# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

> **Pretrained on internet video, fine-tuned on 62 hours of robot data, deployed zero-shot in new labs — 75% pick-and-place success rate with no task labels and no reward.**

| Field | Value |
|-------|-------|
| Authors | Assran, Bardes, Fan, Garrido, Howes, Komeili, Muckley, et al. (24 core authors) |
| Institution | FAIR at Meta; Mila / Polytechnique Montréal |
| arXiv | [2506.09985](https://arxiv.org/abs/2506.09985) |
| Code | https://github.com/facebookresearch/vjepa2 |
| Date | June 11, 2025 |
| Tags | world-models, self-supervised, video-pretraining, robot-planning, JEPA |

---

## The Problem

Robot manipulation models overfit to their training lab. They learn task-specific behaviors rather than physical world dynamics, so data from new environments doesn't transfer. The fix is collecting more diverse robot data — expensive, brittle, and still bounded by the space of behaviors you can demonstrate. The unsolved problem: build a model that understands physics before you point it at a robot.

---

## The Idea

Split learning into two independent stages. Stage one: learn to represent and predict video using only internet data. Stage two: using that frozen representation, learn how robot actions affect future states from a small amount of interaction data.

```
Stage 1: Action-Free Pretraining
  Internet video (22M videos, 1M+ hours)
  + ImageNet (1M images)
        ↓
  V-JEPA 2 encoder (ViT-g, 1B params)
  objective: predict masked video patches in latent space
  progressive resolution: 16 frames → 64 frames, 256 → 384px
        ↓
  frozen video encoder

Stage 2: Action-Conditioned Post-Training
  Droid robot videos (62 hours, unlabeled)
        ↓
  V-JEPA 2-AC predictor (300M, block-causal transformer)
  objective: predict next-frame representations conditioned on action + pose
        ↓
  latent world model

Stage 3: Zero-Shot Deployment
  new environment, image goal
        ↓
  MPC planning: minimize L1(imagined future state, goal state)
  using cross-entropy method over action sequences
        ↓
  robot arm executes actions
```

Predictions stay in representation space throughout — no pixel generation required. This is what makes planning fast: the model rolls out futures in latent space instead of regenerating video frames.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| V-JEPA 2 encoder (ViT-g, 1B params) | Encodes video frames into patch-level representation vectors using mask-denoising objective; frozen after Stage 1 |
| Mask-denoising predictor | Predicts representations of masked video regions from unmasked context; trained jointly with encoder in Stage 1 |
| 3D-RoPE | Rotary position embedding partitioned across temporal, height, width axes; stabilizes training at 1B params |
| VideoMix22M dataset | SSv2 + Kinetics + HowTo100M + Curated-YT1B + ImageNet; 22M samples, 1M+ video hours |
| V-JEPA 2-AC predictor (300M, block-causal transformer) | Given (past frames, robot poses, actions), predicts representations of future frames autoregressively; trained on 62 hours of Droid |
| Teacher-forcing + rollout loss | Teacher forcing: predict next frame from ground-truth input; rollout: unroll T=2 steps autoregressively to reduce error accumulation |
| MPC planner (cross-entropy method) | Samples action sequences, scores by L1(predicted future, goal), updates Gaussian distribution over actions, iterates 10 rounds, executes first action |

---

## Results

### Video Understanding (frozen encoder + attentive probe)

| Task | V-JEPA 2 | Prior SOTA |
|------|----------|------------|
| Something-Something v2 (motion) | 77.3 | — |
| Diving-48 (motion) | — | — |
| Epic-Kitchens-100 anticipation (recall@5) | 39.7 | +44% relative gain over prev. best |
| PerceptionTest (video QA, 8B) | 84.0 | SOTA |
| TempCompass (multi-choice) | 76.9 | SOTA |
| MVP paired accuracy | 44.5 | SOTA |
| TemporalBench short-QA | 36.7 | SOTA |
| TOMATO | 40.3 | SOTA |

### Scaling contributions (ViT-L/16 baseline → ViT-g)

| Intervention | Avg. accuracy gain |
|-------------|-------------------|
| Data: 2M → 22M videos | +1.0 pp |
| Model: 300M → 1B params | +1.5 pp |
| Training: 90K → 252K iterations | +0.8 pp |
| Resolution: 256 → 384, 16 → 64 frames | +0.7 pp |
| **Total** | **+4.0 pp** |

Progressive resolution training: **8.4× speedup** vs. training at full resolution throughout.

### Zero-shot robot manipulation (V-JEPA 2-AC vs. Octo)

| Method | Reach | Grasp (avg) | Pick-&-Place (avg) |
|--------|-------|-------------|-------------------|
| Octo (1M+ trajectories, full Droid) | 100% | 15% | 15% |
| V-JEPA 2-AC (62 hours Droid) | 100% | 72.5% | 75% |

### Planning speed (V-JEPA 2-AC vs. Cosmos)

| Model | Time per action | Pick-&-Place (Lab 2) |
|-------|----------------|---------------------|
| Cosmos (latent diffusion, 20M hours) | 4 minutes | 20% |
| V-JEPA 2-AC (62 hours) | 16 seconds | 65% |

---

## Key Insight

Octo trained on 50× more robot interaction data and still achieved 5× lower pick-and-place success than V-JEPA 2-AC. The difference isn't scale — it's that internet video teaches physics that robot demonstrations alone don't. A model that learns from watching millions of hours of human activity gets intuitive understanding of object permanence, spatial relationships, and cause-effect before it ever sees a robot arm. The 62-hour fine-tuning step only needs to answer: "how do this robot's actions map onto the world model I already have?"

---

## Builder Takeaway

If you build robotic systems or simulation-based agents, the architecture pattern here is directly adoptable: decouple world understanding from action conditioning. Pretrain a representation on cheap, abundant observation data (internet video, simulation rollouts, sensor logs). Freeze it. Train a small action-conditioned predictor on the interaction data you actually have. Use the frozen encoder's knowledge as your physics prior; let the predictor learn the action-state mapping.

The latent-space planning loop is also the practical takeaway. Cosmos achieves good visual fidelity; V-JEPA 2-AC runs 15× faster per planning step and achieves 3× better pick-and-place. For closed-loop control, prediction quality in representation space beats pixel-level fidelity.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/vjepa2_planner.py](scripts/vjepa2_planner.py) | Latent world model with action-conditioned predictor, cross-entropy method planning, and zero-shot MPC loop |
