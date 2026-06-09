---
name: latent-world-model-planning
description: Decouple world-model pretraining from action-conditioned fine-tuning. Pretrain a representation encoder on cheap, abundant observation data. Freeze it. Train a small action-conditioned predictor on interaction data. Plan in latent space using the cross-entropy method for closed-loop control. Faster and more data-efficient than pixel-space generative approaches.
trigger: When building robot manipulation agents, game-playing agents, or any embodied system that needs to plan ahead; when you have large observation datasets but limited interaction data; when you need real-time closed-loop planning and generative video is too slow.
---

## When to use

- Building robotic or embodied agents that need to generalize zero-shot to new environments
- You have abundant observation data (video, simulation) but limited labeled interaction data
- Planning needs to run in real time — generative video models are too slow for closed-loop control
- You want physics priors from internet-scale data without labeling it
- Multi-stage deployment: learn representations once, fine-tune action conditioning per robot/task

## Pattern

1. **Stage 1 — Observation pretraining**: train a video encoder with a mask-denoising objective on large-scale observation data. Target: representations that capture predictable dynamics (motion, object permanence) without pixel-level reconstruction. Freeze the encoder after this stage.
2. **Stage 2 — Action-conditioned fine-tuning**: train a block-causal transformer predictor on top of the frozen encoder. Input: (past frames, actions, proprioception). Output: predicted representation of the next frame. Use teacher-forcing loss + 2-step rollout loss to reduce error accumulation.
3. **Deployment — MPC planning loop**: at each timestep, encode current frame and goal image. Use cross-entropy method to optimize a sequence of actions by minimizing L1(predicted future latent, goal latent). Execute the first action, observe, re-plan.

```
Observation data (large, unlabeled)
        ↓
[Encoder pretraining]
        ↓
Frozen encoder E(·)
        ↓
Interaction data (small, unlabeled)
        ↓
[Action-conditioned predictor training]
  input:  (z_t, a_t, s_t) for t in past
  output: ẑ_{t+1} = P(z_t, a_t, s_t)
  loss:   L1(ẑ, E(x)) for teacher-forcing + rollout
        ↓
World model E + P
        ↓
MPC: argmin_{a_1:T} L1(P(a_1:T; z_k, s_k), z_goal)
```

## Implementation

```python
class LatentWorldModel:
    def __init__(self, encoder: VideoEncoder, predictor: BlockCausalTransformer):
        self.encoder = encoder  # frozen after pretraining
        self.predictor = predictor

    def predict_next(self, z_current: Tensor, action: Tensor, state: Tensor) -> Tensor:
        return self.predictor(z_current, action, state)

    def rollout(self, z0: Tensor, s0: Tensor, actions: list[Tensor]) -> Tensor:
        z = z0
        for a in actions:
            z = self.predict_next(z, a, s0)
        return z

    def energy(self, actions: list[Tensor], z_current: Tensor,
               s_current: Tensor, z_goal: Tensor) -> float:
        z_pred = self.rollout(z_current, s_current, actions)
        return l1_loss(z_pred, z_goal)


class CEMPlanner:
    def __init__(self, world_model: LatentWorldModel, horizon: int = 1,
                 n_samples: int = 800, n_iter: int = 10, elite_frac: float = 0.1):
        self.model = world_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.n_elite = int(n_samples * elite_frac)

    def plan(self, z_current: Tensor, s_current: Tensor, z_goal: Tensor) -> Tensor:
        mu = zeros(self.horizon, action_dim)
        sigma = ones(self.horizon, action_dim)
        for _ in range(self.n_iter):
            actions = sample_gaussian(mu, sigma, self.n_samples)  # (N, T, A)
            energies = [self.model.energy(a, z_current, s_current, z_goal) for a in actions]
            elite = actions[argsort(energies)[:self.n_elite]]
            mu, sigma = elite.mean(0), elite.std(0)
        return mu[0]  # first action of best trajectory


def mpc_loop(model: LatentWorldModel, planner: CEMPlanner,
             camera, robot, goal_image, n_steps: int):
    z_goal = model.encoder(goal_image)
    for _ in range(n_steps):
        obs = camera.capture()
        z_current = model.encoder(obs)
        s_current = robot.end_effector_state()
        action = planner.plan(z_current, s_current, z_goal)
        robot.execute(action)
```

See `scripts/vjepa2_planner.py` for a full runnable implementation.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| CEM samples | 800 | More → better plans, slower; 800 achieves 16 sec/action on RTX 4090 |
| CEM iterations | 10 | Convergence typically within 5–7 |
| Planning horizon T | 1 | Short horizon + re-plan beats long horizon; error accumulates in latent rollouts |
| Rollout loss steps | 2 | Differentiating through 1 recurrent step; longer degrades training stability |
| Action constraint | L1-ball radius 0.075 | Prevents out-of-distribution actions; max 13cm displacement/step |
| Fine-tuning data | 62 hours | Works at this scale given frozen encoder; less data → weaker action-state mapping |
| Encoder size | 1B (ViT-g) | Each +500M params adds ~1.5pp on understanding tasks; worth it for robotics |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Train world model end-to-end on robot data | Pretrain on internet video, freeze encoder, fine-tune predictor only |
| Generative video for planning (pixel-space rollout) | Latent-space rollout — 15× faster, better zero-shot transfer |
| Fine-tune on task-specific demonstrations | No task labels, no reward; trains on raw unlabeled interaction video |
| Per-lab fine-tuning for new environments | Zero-shot deployment with same weights |
| More robot data to generalize | More observation data (video) to build physics priors; robot data only for action mapping |
| Single-stage training (perception + action together) | Two-stage: observation pretraining then action post-training |

## Key numbers

- V-JEPA 2 encoder: 1B params (ViT-g), pretrained on 22M videos / 1M+ hours
- V-JEPA 2-AC predictor: 300M params, fine-tuned on 62 hours of Droid data
- Pick-and-place: 75% (V-JEPA 2-AC) vs. 15% (Octo trained on 1M+ trajectories + full Droid)
- Planning speed: 16 sec/action (V-JEPA 2-AC) vs. 4 min/action (Cosmos)
- Progressive resolution: 8.4× speedup vs. full-resolution training throughout
- Epic-Kitchens-100 anticipation: 39.7 recall@5 (+44% relative over prior best)
- PerceptionTest (8B): 84.0 — SOTA at that scale
- Scaling: +4.0pp cumulative from data + model + training + resolution

## Source

- arXiv: https://arxiv.org/abs/2506.09985
- Code: https://github.com/facebookresearch/vjepa2
