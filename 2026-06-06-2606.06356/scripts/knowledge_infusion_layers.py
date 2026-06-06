"""
Where Should Knowledge Enter? A Layered Framework for Knowledge Infusion
in Multimodal Iterative Generative Models
arxiv: 2606.06356  —  Prasad et al., University of South Carolina, 2026

Install: pip install anthropic  (optional — demo runs with mocks)
Run:     python scripts/knowledge_infusion_layers.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ── Knowledge source ──────────────────────────────────────────────────────────

@dataclass
class KnowledgeTriple:
    subject: str
    relation: str
    obj: str


class KnowledgeGraph:
    """Shared knowledge source — must be used across ALL active layers."""

    def __init__(self, name: str):
        self.name = name
        self.triples: list[KnowledgeTriple] = []
        self.hate_concepts: set[str] = set()
        self.neutral_map: dict[str, str] = {}

    def add_concept(self, concept: str, neutral: str) -> None:
        self.hate_concepts.add(concept)
        self.neutral_map[concept] = neutral

    def detect_violations(self, text: str) -> list[str]:
        return [c for c in self.hate_concepts if c.lower() in text.lower()]

    def neutralize_text(self, text: str, violations: list[str]) -> str:
        result = text
        for v in violations:
            neutral = self.neutral_map.get(v, "[removed]")
            result = result.replace(v, neutral)
        return result

    def score_content(self, content: str) -> float:
        """Returns 0.0 (safe) to 1.0 (fully violating)."""
        violations = self.detect_violations(content)
        return min(1.0, len(violations) * 0.3)

    def get_neutral_conditioning(self) -> str:
        return "safe, neutral, appropriate content"

    def get_neutral_reference(self) -> str:
        return "[neutral reference latent]"

    def score_patches(self, image_repr: str, grid: tuple = (7, 7)) -> list[tuple[str, float]]:
        patches = []
        for i in range(grid[0] * grid[1]):
            score = 0.8 if "hate" in image_repr.lower() and i < 5 else 0.1
            patches.append((f"patch_{i}", score))
        return patches

    def get_neutral_inpaint_prompt(self, patch: str) -> str:
        return f"neutral safe content replacing {patch}"


# ── Iterative generator stub ──────────────────────────────────────────────────

@dataclass
class GeneratorState:
    step: int
    latent: str  # simplified as string for demo
    content_preview: str


class IterativeGenerator:
    """
    Simplified iterative generator — represents diffusion / autoregressive model.
    In production: wraps SDXL, SD-v1.5, or an LLM's generation loop.
    """

    def __init__(self, total_steps: int = 10):
        self.T = total_steps

    def init_state(self, prompt: str) -> GeneratorState:
        return GeneratorState(step=0, latent=f"noise[{prompt[:20]}]", content_preview="")

    def step_fn(self, state: GeneratorState, conditioning: str | None = None) -> GeneratorState:
        cond = conditioning or "default"
        preview = f"preview_step{state.step}[{cond}]"
        # Simulate that hate concepts can emerge mid-generation
        if state.step > 5 and "hate" in state.latent:
            preview = f"preview_step{state.step}[contains:hate_motif][{cond}]"
        return GeneratorState(
            step=state.step + 1,
            latent=state.latent,
            content_preview=preview,
        )

    def decode_preview(self, state: GeneratorState) -> str:
        return state.content_preview

    def renoise(self, state: GeneratorState, t_rewind: int) -> GeneratorState:
        return GeneratorState(
            step=t_rewind,
            latent=state.latent.replace("hate", "noisy"),
            content_preview=f"renoised_at_{t_rewind}",
        )

    def blend_with_neutral(self, state: GeneratorState, neutral_ref: str, alpha: float = 0.7) -> GeneratorState:
        return GeneratorState(
            step=state.step,
            latent=f"blended[{alpha:.1f}*original + {1-alpha:.1f}*{neutral_ref}]",
            content_preview=state.content_preview,
        )

    def inpaint(self, image_repr: str, patch: str, neutral_prompt: str) -> str:
        return image_repr.replace("hate_motif", f"inpainted[{neutral_prompt[:20]}]")

    def finalize(self, state: GeneratorState) -> str:
        return f"image[{state.content_preview}]"


# ── Four-layer knowledge infusion pipeline ────────────────────────────────────

class FailureClass(str, Enum):
    PROMPT_LEVEL = "prompt_level"       # surface fixes this
    STRUCTURAL = "structural"           # trajectory/latent fixes this
    DISTRIBUTIONAL = "distributional"  # parametric fixes this


@dataclass
class LayerConfig:
    use_surface_input: bool = True
    use_trajectory_latent: bool = True
    use_surface_output: bool = True
    t_check: float = 0.9    # fraction of T at which to check mid-generation
    t_rewind: float = 0.3   # fraction of T to rewind to on trigger
    violation_threshold: float = 0.3
    blend_alpha: float = 0.7
    patch_grid: tuple = (7, 7)


class KnowledgeInfusionPipeline:
    """
    Four-layer knowledge infusion pipeline.
    All layers share the same knowledge_graph — critical for avoiding inter-layer conflict.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        generator: IterativeGenerator,
        config: LayerConfig | None = None,
    ):
        self.K = knowledge_graph       # shared across ALL layers
        self.gen = generator
        self.cfg = config or LayerConfig()
        self.log: list[str] = []

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        print(f"  {msg}")

    # ── Layer 1: Surface (input-side) ─────────────────────────────────────────

    def surface_input(self, prompt: str) -> str:
        violations = self.K.detect_violations(prompt)
        if not violations:
            self._log("[Surface:input] No violations in prompt — pass through")
            return prompt
        neutralized = self.K.neutralize_text(prompt, violations)
        self._log(f"[Surface:input] Neutralized {violations} → '{neutralized}'")
        return neutralized

    # ── Layer 2+3: Trajectory + Latent (mid-generation) ──────────────────────

    def trajectory_latent_hook(
        self, state: GeneratorState, conditioning: str
    ) -> tuple[GeneratorState, str]:
        t_check_step = int(self.cfg.t_check * self.gen.T)
        if state.step != t_check_step:
            return state, conditioning

        preview = self.gen.decode_preview(state)
        score = self.K.score_content(preview)
        self._log(f"[Traj+Latent] Step {state.step}: violation score={score:.2f}")

        if score > self.cfg.violation_threshold:
            # Trajectory: switch to neutral conditioning
            new_conditioning = self.K.get_neutral_conditioning()
            self._log(f"[Trajectory] Switching conditioning → '{new_conditioning}'")

            # Latent: rewind and blend with neutral reference
            t_rewind_step = int(self.cfg.t_rewind * self.gen.T)
            state = self.gen.renoise(state, t_rewind_step)
            neutral_ref = self.K.get_neutral_reference()
            state = self.gen.blend_with_neutral(state, neutral_ref, self.cfg.blend_alpha)
            self._log(f"[Latent] Rewound to step {t_rewind_step}, blended with neutral reference")
            return state, new_conditioning

        return state, conditioning

    # ── Layer 4: Surface (output-side) ────────────────────────────────────────

    def surface_output(self, image_repr: str) -> str:
        patch_scores = self.K.score_patches(image_repr, self.cfg.patch_grid)
        violations = [(p, s) for p, s in patch_scores if s > self.cfg.violation_threshold]
        if not violations:
            self._log("[Surface:output] No patch violations — output clean")
            return image_repr

        self._log(f"[Surface:output] {len(violations)} violating patches — inpainting")
        result = image_repr
        for patch, score in violations:
            neutral_prompt = self.K.get_neutral_inpaint_prompt(patch)
            result = self.gen.inpaint(result, patch, neutral_prompt)
        return result

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> tuple[str, list[str]]:
        self.log = []
        print(f"\n[Pipeline] Generating for prompt: '{prompt}'")

        # Layer 1: Surface input
        p_prime = self.surface_input(prompt) if self.cfg.use_surface_input else prompt

        # Generation loop with trajectory/latent hooks
        state = self.gen.init_state(p_prime)
        conditioning = p_prime

        for _ in range(self.gen.T):
            if self.cfg.use_trajectory_latent:
                state, conditioning = self.trajectory_latent_hook(state, conditioning)
            state = self.gen.step_fn(state, conditioning)

        image = self.gen.finalize(state)

        # Layer 4: Surface output
        if self.cfg.use_surface_output:
            image = self.surface_output(image)

        return image, self.log


# ── Demo: cumulative layer ablation matching Table 3 in the paper ─────────────

if __name__ == "__main__":
    # Build the shared knowledge graph (MMKG-style)
    kg = KnowledgeGraph("safety-mmkg")
    kg.add_concept("hate_slur_A", "person")
    kg.add_concept("hate_trope_B", "community")
    kg.add_concept("violent_imagery", "peaceful scene")

    gen = IterativeGenerator(total_steps=10)

    # Adversarial prompt that contains a hate concept
    prompt = "A photo of hate_slur_A in a crowd"

    configs = [
        ("Vanilla (no infusion)",
         LayerConfig(use_surface_input=False, use_trajectory_latent=False, use_surface_output=False)),
        ("+ Surface input only",
         LayerConfig(use_surface_input=True, use_trajectory_latent=False, use_surface_output=False)),
        ("+ Trajectory–Latent",
         LayerConfig(use_surface_input=True, use_trajectory_latent=True, use_surface_output=False)),
        ("+ Surface output (full stack)",
         LayerConfig(use_surface_input=True, use_trajectory_latent=True, use_surface_output=True)),
    ]

    print("=" * 70)
    print("Cumulative layer ablation — matches paper Table 3 structure")
    print("=" * 70)

    for name, cfg in configs:
        print(f"\n{'─'*60}\nConfig: {name}")
        pipeline = KnowledgeInfusionPipeline(kg, gen, cfg)
        image, log = pipeline.generate(prompt)
        # Score the final output
        final_score = kg.score_content(image)
        print(f"  → Final violation score: {final_score:.2f} ({'CLEAN' if final_score < 0.3 else 'VIOLATING'})")
        print(f"  → Output: {image[:80]}")
