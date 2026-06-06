---
name: knowledge-infusion-layers
description: Four-layer framework for injecting structured knowledge into generative models — diagnose the failure class (prompt-level / structural / distributional), select the matching intervention layer (surface / trajectory / latent / parametric), compose layers with a shared knowledge source to avoid inter-layer conflict.
trigger: When a generative model (diffusion, LLM, autoregressive) produces outputs that violate known constraints, safety rules, domain knowledge, or structured facts; when deciding where to inject a knowledge graph, ontology, or rule system into a generation pipeline
---

## When to use

- A diffusion or LLM generates outputs that violate domain constraints despite correct prompting
- Safety filtering (surface output) alone isn't sufficient — violations survive to final output
- You need to decide between RAG, guidance, latent editing, or fine-tuning for a knowledge-alignment task
- Building multi-layer knowledge-constrained generation (safety, medical, manufacturing, legal)
- Debugging why knowledge injection isn't working: diagnose which failure class is occurring

## Pattern

1. **Diagnose the failure class** — is the violation prompt-level (wrong input), structural (model ignores valid input), or distributional (model never learned the concept)?
2. **Select layer(s)** matching the failure class:
   - Prompt-level → Surface (input): rewrite prompt using knowledge source
   - Structural, emerging during generation → Trajectory: modify transition function (guidance, constrained decoding)
   - Structural, localized in representation → Latent: edit intermediate state ht directly
   - Distributional → Parametric: fine-tune with knowledge-derived data
3. **Use a shared knowledge source** across all active layers — independent knowledge signals cause inter-layer conflict
4. **Add layers cumulatively** — surface input first (cheapest), then trajectory/latent (catch what slips through), then surface output (last-resort repair)
5. **Verify complementarity** — each added layer should fix a failure class the prior layers couldn't reach; if not, you're adding cost without benefit

## Implementation

```python
class KnowledgeInfusionPipeline:
    def __init__(self, knowledge_source: KnowledgeGraph, generator: IterativeGenerator):
        self.K = knowledge_source          # shared across ALL layers
        self.gen = generator

    # Layer 1: Surface (input-side)
    def surface_input(self, prompt: str) -> str:
        violations = self.K.detect_violations(prompt)
        return self.K.neutralize(prompt, violations)   # p → p'

    # Layer 2: Trajectory — modify transition function mid-generation
    def trajectory_hook(self, ht, t, ftheta):
        if self.K.check_activation(self.gen.decode_preview(ht)) > threshold:
            neutral_cond = self.K.get_neutral_conditioning()
            return lambda h, t: ftheta(h, t, cond=neutral_cond)  # f̃θ,K
        return ftheta

    # Layer 3: Latent — edit intermediate state directly
    def latent_hook(self, ht, t) -> Tensor:
        if t == self.t_check:
            score = self.K.score_against_prototypes(self.gen.decode_preview(ht))
            if score > self.threshold:
                neutral_ref = self.K.get_neutral_reference_latent()
                ht = self.gen.renoise(ht, t_rewind) * alpha + neutral_ref * (1 - alpha)
        return ht

    # Layer 4: Surface (output-side)
    def surface_output(self, image: Image) -> Image:
        patch_scores = self.K.score_patches(image, grid=(7, 7))
        for patch, score in patch_scores:
            if score > self.threshold:
                image = self.gen.inpaint(image, patch, self.K.get_neutral_prompt(patch))
        return image

    def generate(self, prompt: str) -> Image:
        p_prime = self.surface_input(prompt)           # Layer 1
        x = self.gen.run(p_prime,
                         trajectory_hook=self.trajectory_hook,  # Layer 2
                         latent_hook=self.latent_hook)           # Layer 3
        return self.surface_output(x)                  # Layer 4
```

See `scripts/knowledge_infusion_layers.py` for a full runnable implementation.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| tcheck (trajectory/latent trigger) | ≈ 0.9T | Late denoising — early enough to intervene, late enough to detect structure |
| trewind | ≈ 0.3T | How far back to rewind; trades faithfulness for correction strength |
| Patch grid size (output surface) | 7×7 | Finer grid = more precise but more inpainting calls |
| Shared knowledge source | Required | Independent KGs per layer → inter-layer conflict |
| Layer activation order | Surface-in → Traj/Latent → Surface-out | Cheapest first; fallback layers catch residual failures |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Pick a technique (RAG/guidance/fine-tuning) by familiarity | Diagnose failure class first; select matching layer |
| Evaluate layers in isolation | Composition — each layer covers failures the prior can't reach |
| Independent knowledge sources per layer | Shared knowledge source across all layers prevents conflict |
| Surface filtering as the only safety gate | Surface catches prompt-level only; 17% of violations survive to structural stage |
| "More guidance = better knowledge alignment" | Trajectory vs latent have different persistence; attenuation is a real risk for latent edits |

## Key numbers

- Toxicity reduction: 0.31 → 0.09 (70.97%) on Detonate (25K prompts)
- Surface input alone: 0.31 → 0.17 (fixes prompt-level only)
- Adding trajectory–latent: 0.17 → 0.11 (catches structural violations)
- Adding surface output: 0.11 → 0.09 (residual artifact removal)
- Full stack outperforms SAFREE (0.22) and SLD (0.18) while maintaining higher CLIP + AQI
- Consistent results across SDXL and SD-v1.5 (frozen backbones, training-free)

## Source

- arXiv: https://arxiv.org/abs/2606.06356
