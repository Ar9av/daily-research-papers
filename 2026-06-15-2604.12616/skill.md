---
name: vlm-jailbreak-defense
description: Defensive patterns for VLMs against semantic jailbreak attacks using unmodified natural images. Covers scene-level risk scoring, visual anchor detection, response strategy classification (refusal vs benign reframing), and multi-turn escalation monitoring. Based on MemJack attack findings showing 71% ASR on unmodified COCO images.
trigger: When deploying a VLM that accepts image + text input from untrusted users; when auditing VLM safety alignment; when reviewing safety response strategies (refusal vs redirection); when monitoring multi-turn conversations for escalation patterns.
---

## When to use

- You're deploying a VLM (GPT-5-Mini, Gemini-Flash, Qwen3-VL, etc.) in a user-facing product
- You accept unmodified user-uploaded or externally sourced images alongside text queries
- You want to improve your model's resistance to adaptive multi-turn jailbreak attacks
- You're evaluating whether your VLM's safety responses are training an adversarial attacker

## Key findings from MemJack

| Finding | Implication |
|---------|------------|
| 71.48% ASR on unmodified COCO images | Pixel-based defenses don't cover semantic attacks |
| Memory removal drops ASR 72%→38% | Stateless attacks are much weaker — stateful monitoring matters |
| Direct refusal has lowest resistance | Explicit refusals give attackers a clear gradient to follow |
| Benign reframing has highest resistance | No refusal signal = no attack gradient |
| 90% ASR at R=100 budget | Given enough turns, almost any natural image can be weaponized |
| Office/street scenes highest risk | Scene-level filtering is a tractable first-pass defense |

## Defense Pattern: Response Strategy

**Worst**: Direct refusal ("I can't help with that")
- Tells attacker exactly what failed
- 41.4% of model responses in the wild are direct refusals
- MemJack bypasses these by switching angles

**Best**: Benign reframing
- Proactively reinterpret the request as innocent and answer that version
- Generates no explicit refusal signal
- Hardest for adaptive attackers to defeat (no gradient to follow)
- Example: "Let me address the general safety aspects of handling tools like this..."

**Medium**: Safe answer without acknowledgment of the harmful intent

## Defense Pattern: Input-Side Filtering

```
For each incoming (image, query) pair:
1. Scene risk score: flag high-risk scenes (office, street, transport) for elevated monitoring
2. Anchor detection: identify dual-use objects in image captions or object detection output
3. Query escalation score: check for MemJack angle patterns (hypothetical framing, "for educational purposes", "step-by-step")
4. Multi-turn escalation: track escalation trend across conversation window
```

## Pattern

1. **Scene risk gate**: run scene classifier on input image; apply elevated suspicion threshold for high-risk scenes (office workspace ≥0.80, street/transport ≥0.60)
2. **Anchor detection**: extract objects from image; flag dual-use categories (weapons/tools, chemicals, vehicles, infrastructure)
3. **Query pattern scan**: detect MemJack attack angle signatures before sending to VLM
4. **Response strategy**: configure VLM toward benign reframing over direct refusal for borderline requests
5. **Turn-level monitoring**: track semantic escalation score across conversation; trigger review after 3-turn rising escalation window

## Implementation

See `scripts/memjack_defense.py` for:
- `SceneRiskScorer` — scene label → risk score (calibrated to Figure 6 heatmap)
- `AnchorHeuristicDetector` — caption/object list → dual-use anchor detection
- `DefenseStrategyEvaluator` — VLM response → refusal vs benign-reframe classification
- `MultiTurnEscalationMonitor` — sliding window escalation tracking

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Scene risk threshold | 0.55 = high | Office workspace (0.81) and street (0.62) both high |
| Anchor risk threshold | ≥2 anchors | Single anchor is low signal; 2+ warrants elevated monitoring |
| Escalation window | 3 turns | Matches MemJack's angle-switching cycle |
| Escalation alert | 0.40 avg score | Calibrated to catch "step-by-step + educational purposes" combinations |
| Max query budget | 20 rounds | MemJack's default; at R=100 ASR reaches 90% |

## Pitfalls

| Approach | Problem |
|----------|---------|
| Pixel perturbation detection only | Misses semantic attacks on unmodified images |
| Direct refusal as primary safety behavior | Lowest resistance; provides attack gradient |
| Single-turn evaluation | Misses gradual escalation that crosses safety threshold only after several turns |
| Text-only content filtering | Bypassed when harmful intent is framed as visual analysis |
| Blocking all office/tool images | Too broad; dual-use objects are in the majority of real-world images |

## Key numbers

- 71.48% ASR on 5,000 unmodified COCO val2017 images (R=20)
- 90% ASR under extended budget (R=100)
- Memory ablation: ASR 72%→38% (−34 pp)
- Benign reframing: 20.8% of defense responses; highest resistance
- Direct refusal: 41.4% of defense responses; lowest resistance
- Office workspace non-violent illegal acts: 81% ASR contribution
- Street/traffic scenes: 61–62% ASR contribution
- 83.8% linear separability of safe/unsafe (image, prompt) embeddings via SVM

## Source

- arXiv: https://arxiv.org/abs/2604.12616
- Dataset: MemJack-Bench (113,000+ trajectories, forthcoming)
