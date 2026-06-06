---
title: "Where Should Knowledge Enter? A Layered Framework for Knowledge Infusion in Multimodal Iterative Generative Models"
arxiv_id: "2606.06356"
date: "2026-06-06"
authors: "Renjith Prasad, Chathurangi Shyalika, Anushka Pawar, Aahan Rathod, Amit Sheth"
institution: "University of South Carolina; Indian AI Research Organization"
tags: ["knowledge-infusion", "diffusion-models", "generative-models", "safety", "framework"]
---

# Where Should Knowledge Enter? A Layered Framework for Knowledge Infusion in Multimodal Iterative Generative Models

> **Four intervention layers — surface, trajectory, latent, parametric — each fixing failure classes the others can't reach; combined 70.97% toxicity reduction with frozen backbones.**

| Field | Value |
|-------|-------|
| Authors | Renjith Prasad, Chathurangi Shyalika, Anushka Pawar, Aahan Rathod, Amit Sheth |
| Institution | University of South Carolina, Indian AI Research Organization |
| arXiv | [2606.06356](https://arxiv.org/abs/2606.06356) |
| Code | — |
| Date | June 4, 2026 |
| Tags | knowledge-infusion, diffusion-models, safety, framework |

---

## The Problem

Generative models produce fluent outputs but fail when generation must respect structured, domain-specific, or safety-critical knowledge. The field treats this as a technique-selection problem (RAG? guidance? fine-tuning?) rather than asking *which component of the generation process* each technique actually modifies. That makes principled design impossible — if a diffusion model generates an image violating a known spatial relation, should you fix the prompt, steer the sampler, edit a latent, or retrain?

---

## The Idea

Every iterative generator (diffusion, autoregressive, flow) produces outputs through a trajectory of internal states: `h0 → h1 → ... → hT = x`. Knowledge can enter at exactly four formal components of this process:

```
                    SURFACE (input)
                         ↓
prompt p ──→ h0 → h1 → h2 → ... → hT ──→ output x
              ↑    ↑    ↑              ↑
           LATENT  LATENT LATENT    SURFACE (output)
              └─── TRAJECTORY ─────┘
                   (all steps)
                         ↕
                    PARAMETRIC (θ, pre-inference)
```

Each layer targets a different formal component, addresses different failure classes, and offers different persistence, cost, and controllability tradeoffs. No single layer dominates — they are complementary.

---

## Architecture

| Layer | Formal Target | When | Operation | Failure Class Fixed |
|-------|--------------|------|-----------|-------------------|
| Surface | Input boundary or output boundary | Pre/post generation | `p → p'` or `x → x'` | Prompt-level violations |
| Trajectory | Transition function fθ | Per step at inference | `fθ → f̃θ,K` | Structural violations (spatial, compositional) |
| Latent | Intermediate state ht | Per step at inference | `ht → ht'` | Structural violations (fine-grained, spatial) |
| Parametric | Model weights θ | Pre-inference (training) | `θ → θ'` | Distributional violations (systematic gaps) |

| Layer | Controllability | Interpretability | Persistence | Cost |
|-------|----------------|-----------------|-------------|------|
| Surface | Low | High | Transient | Low |
| Trajectory | High | Low | Transient (continuous) | Moderate |
| Latent | High | Low | Transient (attenuating) | Moderate |
| Parametric | Low | Low | Permanent | High |

Key distinction between Trajectory and Latent: trajectory modifies the *update rule* reapplied every step (can't be attenuated); latent modifies the *state* at discrete points (subsequent unmodified dynamics may attenuate the edit).

---

## Results

Safety alignment experiment on Detonate benchmark (25K prompts), SDXL and SD-v1.5 frozen backbones, MMKG knowledge source:

| Configuration | Toxicity ↓ (SDXL) | CLIP ↑ | AQI ↑ |
|---------------|-------------------|--------|-------|
| Vanilla | 0.31 | 0.310 | 0.23 |
| SAFREE (baseline) | 0.22 | 0.305 | 0.28 |
| SLD (baseline) | 0.18 | 0.320 | 0.31 |
| + Surface (input) | 0.17 | 0.330 | 0.32 |
| + Trajectory–Latent | 0.11 | 0.340 | 0.37 |
| + Surface (output) | **0.09** | 0.335 | 0.36 |

- Total toxicity reduction: 0.31 → 0.09 = **70.97%** vs vanilla
- Full stack halves best single-layer toxicity (0.17 → 0.09)
- Outperforms both baselines on toxicity while maintaining higher CLIP and AQI
- Pattern consistent across both SDXL and SD-v1.5

---

## Key Insight

Each layer addressed failures the previous layer missed — exactly as the formal framework predicted. Surface input neutralized prompt-level triggers but 17% remained toxic (structural violations where pretrained weights hallucinate despite a clean prompt). Trajectory–latent caught mid-generation structural failures. Output-side surface removed residual artifacts. The failure classes are genuinely distinct and require distinct interventions. Shared MMKG across all layers prevented inter-layer conflict — a unified knowledge signal is what makes multi-layer composition synergistic rather than destructive.

---

## Builder Takeaway

Stop treating knowledge injection as a technique choice and start treating it as a **layer choice**. Diagnose the failure class first: if your model hallucinates despite a correct prompt, that's a structural violation — prompt engineering won't fix it, you need trajectory or latent intervention. If the bias is distributional (the model never learned the concept), no inference-time method reaches it — you need parametric infusion. The framework gives you the vocabulary to match failure to fix rather than trying techniques at random. For safety-critical generation systems: build all reachable layers with a shared knowledge source; independent layers with conflicting signals actively hurt each other.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/knowledge_infusion_layers.py](scripts/knowledge_infusion_layers.py) | Four-layer knowledge infusion framework: surface, trajectory, latent composition with shared knowledge source |
