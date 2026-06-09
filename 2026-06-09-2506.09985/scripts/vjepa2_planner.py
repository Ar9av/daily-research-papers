"""
V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
arxiv: 2506.09985  —  Assran, Bardes, Fan, et al., FAIR Meta, 2025

Install: pip install numpy  (optional — demo runs with stdlib only)
Run:     python scripts/vjepa2_planner.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ── Tensor mock (avoid numpy dep for demo) ───────────────────────────────────

class Tensor:
    """Minimal n-d array for demo purposes."""
    def __init__(self, data: list):
        self.data = data
        self.shape = self._shape(data)

    def _shape(self, d):
        if isinstance(d, list):
            return (len(d),) + (self._shape(d[0]) if d else ())
        return ()

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    @staticmethod
    def zeros(*shape) -> "Tensor":
        def _fill(s):
            if len(s) == 1:
                return [0.0] * s[0]
            return [_fill(s[1:]) for _ in range(s[0])]
        return Tensor(_fill(shape))

    @staticmethod
    def randn(*shape) -> "Tensor":
        def _fill(s):
            if len(s) == 1:
                return [random.gauss(0, 1) for _ in range(s[0])]
            return [_fill(s[1:]) for _ in range(s[0])]
        return Tensor(_fill(shape))

    def flat(self) -> list[float]:
        def _flat(d):
            if isinstance(d, list):
                out = []
                for x in d:
                    out.extend(_flat(x))
                return out
            return [d]
        return _flat(self.data)

    def l1_distance(self, other: "Tensor") -> float:
        a, b = self.flat(), other.flat()
        return sum(abs(x - y) for x, y in zip(a, b))


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class Frame:
    pixels: list      # H x W x 3 (mocked as flat list)
    timestamp: float


@dataclass
class EndEffectorState:
    position: list[float]    # [x, y, z]
    orientation: list[float] # [roll, pitch, yaw]
    gripper: float           # 0=open, 1=closed

    def as_vector(self) -> list[float]:
        return self.position + self.orientation + [self.gripper]


@dataclass
class Action:
    delta_position: list[float]    # [dx, dy, dz]
    delta_orientation: list[float] # [dr, dp, dy]
    delta_gripper: float

    def as_vector(self) -> list[float]:
        return self.delta_position + self.delta_orientation + [self.delta_gripper]

    @staticmethod
    def zero() -> "Action":
        return Action([0.0]*3, [0.0]*3, 0.0)

    @staticmethod
    def random_bounded(l1_radius: float = 0.075) -> "Action":
        v = [random.gauss(0, 0.03) for _ in range(7)]
        # project onto L1-ball
        norm = sum(abs(x) for x in v)
        if norm > l1_radius:
            v = [x * l1_radius / norm for x in v]
        return Action(v[:3], v[3:6], v[6])


# ── Video Encoder (frozen after pretraining) ──────────────────────────────────

class VideoEncoder:
    """
    Mock of V-JEPA 2 ViT-g encoder.
    In production: 1B-param ViT-g trained on 22M videos with mask-denoising objective.
    Outputs (H/16 x W/16) patch features, dim=1408.
    """
    LATENT_DIM = 16  # compressed for demo (real: 16 x 16 x 1408)

    def __init__(self, seed: int = 42):
        random.seed(seed)
        # Mock: fixed random projection as "pretrained" weights
        self._weights = [[random.gauss(0, 0.1) for _ in range(self.LATENT_DIM)]
                         for _ in range(256)]  # 256 = mock pixel dim

    def encode(self, frame: Frame) -> Tensor:
        """Encode a single frame to a patch-level representation."""
        pixels = frame.pixels[:256] + [0.0] * max(0, 256 - len(frame.pixels))
        latent = [sum(p * w for p, w in zip(pixels, row)) for row in self._weights]
        # Normalize
        norm = math.sqrt(sum(x**2 for x in latent)) + 1e-6
        return Tensor([x / norm for x in latent])

    def encode_goal(self, goal_frame: Frame) -> Tensor:
        return self.encode(goal_frame)


# ── Action-Conditioned Predictor ──────────────────────────────────────────────

class BlockCausalPredictor:
    """
    Mock of V-JEPA 2-AC 300M block-causal transformer.
    In production: 24-layer transformer, 1024 hidden dim, block-causal attention.
    Fine-tuned on 62 hours of Droid robot data (unlabeled).
    """
    LATENT_DIM = VideoEncoder.LATENT_DIM
    ACTION_DIM = 7
    STATE_DIM = 7

    def __init__(self, seed: int = 0):
        random.seed(seed)
        # Mock learned dynamics: action affects latent via learned mixing
        self._action_weights = [[random.gauss(0, 0.05) for _ in range(self.LATENT_DIM)]
                                for _ in range(self.ACTION_DIM)]
        self._state_weights = [[random.gauss(0, 0.02) for _ in range(self.LATENT_DIM)]
                               for _ in range(self.STATE_DIM)]

    def predict_next(self, z: Tensor, action: Action, state: EndEffectorState) -> Tensor:
        """Predict representation of next frame given current latent, action, and state."""
        z_flat = z.flat()
        a_vec = action.as_vector()
        s_vec = state.as_vector()

        # z_{t+1} ≈ z_t + f(a_t) + g(s_t)  (simplified linear dynamics)
        action_effect = [sum(a * w for a, w in zip(a_vec, row)) for row in self._action_weights]
        state_effect = [sum(s * w for s, w in zip(s_vec, row)) for row in self._state_weights]

        next_z = [zv + ae + se for zv, ae, se in zip(z_flat, action_effect, state_effect)]
        norm = math.sqrt(sum(x**2 for x in next_z)) + 1e-6
        return Tensor([x / norm for x in next_z])

    def rollout(self, z0: Tensor, s0: EndEffectorState,
                actions: list[Action]) -> Tensor:
        """Autoregressively predict T steps ahead."""
        z = z0
        for action in actions:
            z = self.predict_next(z, action, s0)
        return z


# ── Latent World Model ────────────────────────────────────────────────────────

class LatentWorldModel:
    def __init__(self, encoder: VideoEncoder, predictor: BlockCausalPredictor):
        self.encoder = encoder
        self.predictor = predictor

    def energy(self, actions: list[Action], z_current: Tensor,
               s_current: EndEffectorState, z_goal: Tensor) -> float:
        """Goal-conditioned energy: L1(predicted future latent, goal latent)."""
        z_pred = self.predictor.rollout(z_current, s_current, actions)
        return z_pred.l1_distance(z_goal)


# ── Cross-Entropy Method Planner ──────────────────────────────────────────────

@dataclass
class CEMConfig:
    horizon: int = 1          # planning steps; 1 + replan beats long horizon
    n_samples: int = 100      # 800 in paper; reduced for demo
    n_iter: int = 10          # refinement iterations
    elite_frac: float = 0.1   # top-k fraction
    action_l1_radius: float = 0.075  # max displacement per step (~13cm)


class CEMPlanner:
    """
    Cross-Entropy Method planning over action sequences.
    At each MPC step: sample → score → refit → return best first action.
    """
    def __init__(self, world_model: LatentWorldModel, config: CEMConfig = CEMConfig()):
        self.model = world_model
        self.cfg = config
        self.n_elite = max(1, int(config.n_samples * config.elite_frac))

    def _sample_actions(self, mu: list[float], sigma: list[float]) -> list[Action]:
        """Sample one action trajectory from per-dimension Gaussians."""
        v = [random.gauss(m, s) for m, s in zip(mu, sigma)]
        norm = sum(abs(x) for x in v)
        if norm > self.cfg.action_l1_radius:
            v = [x * self.cfg.action_l1_radius / norm for x in v]
        return [Action(v[:3], v[3:6], v[6])]

    def plan(self, z_current: Tensor, s_current: EndEffectorState,
             z_goal: Tensor) -> Action:
        """Return the best first action for MPC re-planning."""
        action_dim = 7
        mu = [0.0] * action_dim
        sigma = [1.0] * action_dim

        for iteration in range(self.cfg.n_iter):
            candidates = [self._sample_actions(mu, sigma) for _ in range(self.cfg.n_samples)]
            energies = [self.model.energy(c, z_current, s_current, z_goal) for c in candidates]

            # Select elite trajectories
            ranked = sorted(zip(energies, candidates), key=lambda x: x[0])
            elite = [c for _, c in ranked[:self.n_elite]]

            # Refit Gaussian from elite set
            elite_vecs = [[a.as_vector() for a in traj] for traj in elite]
            flat_elite = [v[0] for v in elite_vecs]  # horizon=1

            mu = [sum(e[i] for e in flat_elite) / len(flat_elite) for i in range(action_dim)]
            variance = [sum((e[i] - mu[i])**2 for e in flat_elite) / len(flat_elite)
                        for i in range(action_dim)]
            sigma = [math.sqrt(v) + 1e-4 for v in variance]

            best_energy = ranked[0][0]

        best_vec = mu
        return Action(best_vec[:3], best_vec[3:6], best_vec[6])


# ── MPC Control Loop ──────────────────────────────────────────────────────────

@dataclass
class Robot:
    """Mock Franka Emika Panda."""
    _ee_state: EndEffectorState = field(default_factory=lambda: EndEffectorState(
        position=[0.5, 0.0, 0.4],
        orientation=[0.0, 0.0, 0.0],
        gripper=0.0
    ))

    def end_effector_state(self) -> EndEffectorState:
        return self._ee_state

    def execute(self, action: Action) -> None:
        self._ee_state = EndEffectorState(
            position=[p + d for p, d in zip(self._ee_state.position, action.delta_position)],
            orientation=[o + d for o, d in zip(self._ee_state.orientation, action.delta_orientation)],
            gripper=max(0.0, min(1.0, self._ee_state.gripper + action.delta_gripper))
        )

    def capture_frame(self) -> Frame:
        pos = self._ee_state.position
        pixels = [pos[0] * 0.3 + pos[1] * 0.5 + pos[2] * 0.2 + random.gauss(0, 0.01)
                  for _ in range(256)]
        return Frame(pixels=pixels, timestamp=0.0)


def mpc_loop(model: LatentWorldModel, planner: CEMPlanner,
             robot: Robot, goal_frame: Frame, n_steps: int = 5) -> list[float]:
    """Closed-loop MPC: observe → plan → execute → repeat."""
    z_goal = model.encoder.encode_goal(goal_frame)
    energies = []
    print(f"\n  Goal latent encoded. Running {n_steps}-step MPC loop...")

    for step in range(n_steps):
        obs = robot.capture_frame()
        z_current = model.encoder.encode(obs)
        s_current = robot.end_effector_state()

        energy = model.energy([Action.zero()], z_current, s_current, z_goal)
        energies.append(energy)

        action = planner.plan(z_current, s_current, z_goal)
        robot.execute(action)

        print(f"  step {step+1}: energy={energy:.4f}, "
              f"ee_pos={[f'{x:.3f}' for x in s_current.position]}")

    return energies


# ── Pretraining mock ──────────────────────────────────────────────────────────

def pretrain_encoder(n_videos: int = 22_000_000) -> VideoEncoder:
    """
    Stage 1: mask-denoising pretraining on internet video.
    Real: V-JEPA 2 mask-denoising objective, ViT-g/16, VideoMix22M + ImageNet.
    Here: returns a mock encoder with fixed random weights.
    """
    print(f"  [Stage 1] Pretraining encoder on {n_videos:,} videos (internet-scale)")
    print(f"  [Stage 1] Architecture: ViT-g, 1B params, 3D-RoPE")
    print(f"  [Stage 1] Progressive resolution: 16→64 frames, 256→384px")
    print(f"  [Stage 1] Training: 252K iterations, warmup-constant-decay schedule")
    print(f"  [Stage 1] Encoder frozen after pretraining.")
    return VideoEncoder()


def finetune_predictor(encoder: VideoEncoder,
                       interaction_hours: float = 62.0) -> BlockCausalPredictor:
    """
    Stage 2: action-conditioned post-training on robot data.
    Real: 300M block-causal transformer, teacher-forcing + 2-step rollout loss, Droid dataset.
    """
    print(f"\n  [Stage 2] Post-training action-conditioned predictor")
    print(f"  [Stage 2] Data: {interaction_hours:.0f} hours of unlabeled Droid robot videos")
    print(f"  [Stage 2] No task labels. No reward. No success annotations.")
    print(f"  [Stage 2] Loss: teacher-forcing + rollout (T=2) over frozen encoder representations")
    print(f"  [Stage 2] Architecture: 300M transformer, block-causal attention")
    return BlockCausalPredictor()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA 2 — Latent World Model Planning Demo")
    print("arxiv: 2506.09985")
    print("=" * 60)

    # Stage 1: pretrain encoder on internet video
    encoder = pretrain_encoder(n_videos=22_000_000)

    # Stage 2: fine-tune action-conditioned predictor on robot data
    predictor = finetune_predictor(encoder, interaction_hours=62.0)

    world_model = LatentWorldModel(encoder, predictor)
    planner = CEMPlanner(world_model, CEMConfig(n_samples=50, n_iter=5))

    robot = Robot()

    # Task 1: single-goal reaching
    print("\n[Task 1: Single-Goal Reaching]")
    print("  Goal: move end-effector to target position from monocular RGB camera")
    goal_frame = Frame(pixels=[0.8 + random.gauss(0, 0.01) for _ in range(256)], timestamp=0.0)

    energies = mpc_loop(world_model, planner, robot, goal_frame, n_steps=5)
    converged = energies[-1] < energies[0]
    print(f"  Energy trend: {energies[0]:.4f} → {energies[-1]:.4f} "
          f"({'converging' if converged else 'not converged — more steps needed'})")

    # Task 2: multi-subgoal pick-and-place
    print("\n[Task 2: Multi-Subgoal Pick-and-Place]")
    print("  Subgoals: (1) grasp object, (2) lift, (3) place at goal position")
    subgoals = [
        Frame(pixels=[0.6 + i * 0.1 + random.gauss(0, 0.01) for _ in range(256)], timestamp=float(i))
        for i in range(3)
    ]
    robot2 = Robot()
    subgoal_steps = [4, 10, 4]  # steps per subgoal (from paper)
    for idx, (subgoal, n_steps) in enumerate(zip(subgoals, subgoal_steps)):
        print(f"\n  Subgoal {idx+1}/{len(subgoals)} ({n_steps} steps):")
        mpc_loop(world_model, planner, robot2, subgoal, n_steps=n_steps)

    # Comparison
    print("\n" + "=" * 60)
    print("Planning comparison (from paper, Lab 2):")
    print(f"  Cosmos (video diffusion, 20M hr pretraining):  4 min/action, 20% pick-&-place")
    print(f"  V-JEPA 2-AC (latent world model, 62hr ft):    16 sec/action, 65% pick-&-place")
    print(f"  Speed gain: 15× faster. Success gain: 3.25×.")
    print(f"\nZero-shot robot manipulation vs. Octo (avg across both labs):")
    print(f"  Octo (1M+ trajectories): Reach=100%, Grasp=15%, Pick-&-Place=15%")
    print(f"  V-JEPA 2-AC (62 hours):  Reach=100%, Grasp=72.5%, Pick-&-Place=75%")
