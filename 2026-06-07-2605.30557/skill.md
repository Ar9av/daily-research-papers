---
name: spatial-uncertainty-abstention
description: Evaluate and improve VLM abstention under unreliable visual observations — occlusion (missing info) and perspective ambiguity (misleading info). Pattern for building agents that know when to request a better view instead of answering confidently from insufficient evidence.
trigger: When building vision agents that operate in real environments, when a VLM must decide whether to answer or abstain from a spatial question, when designing multi-view robotic or embodied systems that need observation reliability awareness
---

## When to use

- Vision agents deployed in physical or 3D environments (robotics, AR, drones, embodied AI)
- Any system where a VLM answers spatial questions from a single viewpoint
- Evaluating whether your VLM knows when to say "I can't determine this from this view"
- Designing active perception pipelines: model requests additional views when current observation is unreliable
- Fine-tuning VLMs for calibrated spatial reasoning (not just accuracy, but abstention quality)

## Pattern

1. **Classify the observation condition** — is the visual evidence complete (clean), missing (occlusion), or misleading (perspective ambiguity)? These are distinct failure modes requiring different handling
2. **Design abstention as a first-class output** — include "Cannot determine" as a valid answer option; evaluate it separately from answer accuracy
3. **Track the answer–abstention trade-off** — improving abstention (Unans. accuracy) often hurts answerable accuracy; a robust system improves both simultaneously
4. **For active perception**: implement ViewSel — given a bad view, score candidate alternative viewpoints and request the most informative one
5. **For fine-tuning**: train on *diverse* ambiguity types (both occlusion and perspective), not just one — single-condition fine-tuning fails to generalize across ambiguity types
6. **Avoid relying on visual input under perspective ambiguity** — misleading cues actively hurt abstention; consider text-only fallback or explicit reliability flag before committing to an answer

## Implementation

```python
class ObservationReliabilityEvaluator:
    def classify_observation(self, image, question) -> ObsCondition:
        # Heuristics or a trained classifier
        # ObsCondition: CLEAN | OCCLUDED | PERSPECTIVE_AMBIGUOUS
        ...

    def should_abstain(self, model_output: str) -> bool:
        return "cannot determine" in model_output.lower()

    def score_abstention(self, predictions, ground_truth) -> dict:
        answerable = [(p, g) for p, g, a in zip(predictions, ground_truth, answerability) if a]
        unanswerable = [(p, g) for p, g, a in zip(predictions, ground_truth, answerability) if not a]
        return {
            "ans_acc": accuracy(answerable),
            "unans_acc": sum(self.should_abstain(p) for p, _ in unanswerable) / len(unanswerable),
            "tradeoff": ans_acc - unans_acc,   # lower = better balance
        }

    def viewsel_score(self, selected_view: int, reference_view: int) -> bool:
        return selected_view == reference_view

    def abstain_then_viewsel(self, abstained: bool, selected_view: int, reference_view: int) -> bool:
        # Both must be correct
        return abstained and self.viewsel_score(selected_view, reference_view)

def structured_prompt(question: str) -> str:
    return f"""Before answering, assess:
1. Is the target object clearly visible in this image?
2. Could the viewpoint be distorting geometric properties?
3. If visual evidence is incomplete or unreliable, respond: Cannot determine.

Question: {question}"""
```

See `scripts/spatial_uncertain.py` for a full runnable evaluator with mock VLM.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Abstention token | "Cannot determine" | Must be a selectable multiple-choice option, not free-form |
| LoRA rank | 16 (α=32) | Fine-tuning for abstention; diversity of training conditions matters more than rank |
| Training data mix | Both occlusion + perspective | Single-condition training fails to generalize |
| Prompt strategy | Structured (assess reliability first) | Gains on Unans. at cost to Ans.; fine-tuning resolves trade-off |
| ViewSel candidates | 5 views (1 informative + 4 ambiguous) | More candidates → harder; ensure reference view is included |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Evaluate only answerable accuracy | Track Ans. and Unans. separately — they trade off |
| Assume visual input always helps | Under perspective ambiguity, image actively hurts Unans. accuracy |
| Prompt-only fix for abstention | Structured prompting improves Unans. but hurts Ans.; fine-tuning resolves both |
| Fine-tune on one ambiguity type | LoRA-Occ fails on perspective; LoRA-Pers fails on occlusion; use mixed |
| Single-viewpoint answers as ground truth | Treat VLM spatial answers as provisional; architect for view requests |
| No "I can't see this" output class | "Cannot determine" must be a first-class option, not an afterthought |

## Key numbers

- Perspective ambiguity Unans. accuracy: <10% for all tested models (random = 25%)
- Visual input under perspective ambiguity: GPT-5.4 Unans. -21.7pp, Gemini-3.0-Flash -35.8pp
- GPT-5.4 ViewSel: 70.9 → AbstainViewSel: 22.6 (abstention bottleneck)
- LoRA-Mixed vs base: Occ-Unans 41.0→62.8, Pers-Unans 42.9→76.9, both without accuracy drop
- 10,322 QA pairs across 240 scenes, 43 room types, 4 question types

## Source

- arXiv: https://arxiv.org/abs/2605.30557
- Website: https://spatialuncertain.github.io
