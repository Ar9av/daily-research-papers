"""
SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning
arXiv: 2606.13673
NVIDIA, 2026

Install:
    pip install numpy scipy matplotlib pdfminer.six

Run:
    python scripts/spatialclaw_kernel.py
"""

import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stdout

BLOCKED_MODULES = {"os", "subprocess", "sys", "shutil", "socket"}
BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "open"}


def ast_safety_check(code: str) -> tuple[bool, str]:
    """Parse code AST and reject disallowed modules/builtins before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    return False, f"Blocked import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BLOCKED_MODULES:
                return False, f"Blocked from-import: {node.module}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                return False, f"Blocked builtin: {node.func.id}"
    return True, ""


class PersistentKernel:
    """
    Persistent Python workspace for one spatial reasoning example.
    Exposes InputImages, tools, show(), ReturnAnswer(), numpy, scipy, matplotlib.
    All variables survive across execute() calls.
    """

    def __init__(self, frames, tools, metadata=None):
        self._answer = None
        self._shown_images: list = []
        self._ns: dict = {
            "InputImages": frames,
            "tools": tools,
            "Metadata": metadata or {},
            "show": self._show,
            "ReturnAnswer": self._return_answer,
        }
        for pkg, alias in [("numpy", "np"), ("scipy", "scipy"), ("matplotlib.pyplot", "plt")]:
            try:
                import importlib
                mod = importlib.import_module(pkg)
                if pkg == "matplotlib.pyplot":
                    importlib.import_module("matplotlib").use("Agg")
                self._ns[alias] = mod
            except ImportError:
                pass

    def _show(self, img, label: str = "") -> None:
        self._shown_images.append((label, img))

    def _return_answer(self, answer) -> None:
        self._answer = answer

    def execute(self, code: str) -> dict:
        ok, err = ast_safety_check(code)
        if not ok:
            return {"status": "blocked", "error": err, "stdout": "", "images": [], "answer": None, "variables": {}}

        buf = io.StringIO()
        self._shown_images = []
        try:
            with redirect_stdout(buf):
                exec(code, self._ns)  # noqa: S102
            status = "ok"
            error = ""
        except Exception:
            status = "error"
            error = traceback.format_exc()

        var_summary = {
            k: (f"{type(v).__name__} shape={v.shape}" if hasattr(v, "shape")
                else f"{type(v).__name__} len={len(v)}" if hasattr(v, "__len__") and not isinstance(v, str)
                else type(v).__name__)
            for k, v in self._ns.items()
            if not k.startswith("_") and k not in (
                "InputImages", "tools", "Metadata", "show", "ReturnAnswer", "np", "scipy", "plt"
            )
        }

        return {
            "status": status,
            "stdout": buf.getvalue(),
            "error": error,
            "images": list(self._shown_images),
            "answer": self._answer,
            "variables": var_summary,
        }

    @property
    def answered(self) -> bool:
        return self._answer is not None


SYSTEM_PROMPT = """You are a spatial reasoning agent with access to a persistent Python kernel.
At each step, write exactly one Python cell. Available in the kernel:
  InputImages    — list of frames (numpy arrays)
  tools          — perception toolkit (Reconstruct, SAM3, and geometry utilities)
  Metadata       — frame rate, duration, frame indices for video inputs
  show(img)      — registers an image into your next context for visual inspection
  ReturnAnswer(x)— submits the final answer and terminates the loop
  np, scipy, plt — scientific computing libraries

Spatial reasoning discipline:
  - Prefer metric computation (3D points, camera geometry) over pixel-level estimates
  - Verify segmentation masks with show() before computing distances or directions
  - Cross-check numerical results against visualizations before calling ReturnAnswer()
  - Use scipy.spatial.KDTree for nearest-neighbor queries, not centroid-to-centroid distance
"""


def extract_code_block(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def build_messages(question: str, history: list[dict], plan: str = "") -> list[dict]:
    system = SYSTEM_PROMPT
    if plan:
        system += f"\n\nAnalysis plan:\n{plan}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for turn in history:
        msgs.append({"role": "assistant", "content": f"```python\n{turn['code']}\n```"})
        fb = turn["result"]
        parts = [f"Status: {fb['status']}"]
        if fb["stdout"]:
            parts.append(f"Stdout:\n{fb['stdout'].rstrip()}")
        if fb["error"]:
            parts.append(f"Error:\n{fb['error'].rstrip()}")
        if fb["variables"]:
            parts.append("New variables: " + ", ".join(f"{k}: {v}" for k, v in fb["variables"].items()))
        msgs.append({"role": "user", "content": "\n".join(parts)})
    return msgs


def run_spatial_agent(frames, tools, metadata, question, vlm_fn, max_steps: int = 30):
    """
    Execute a spatial reasoning question with SpatialClaw's persistent kernel loop.

    Args:
        frames:    list of numpy arrays (video frames or multi-view images)
        tools:     object with .Reconstruct() and .SAM3() callables
        metadata:  dict with fps, duration, frame_indices (None for images)
        question:  the spatial question string
        vlm_fn:    callable(messages: list[dict]) -> str  (your VLM inference)
        max_steps: maximum agentic loop iterations

    Returns:
        answer (str/float/int) or None if max_steps exhausted
    """
    kernel = PersistentKernel(frames, tools, metadata)
    history = []

    for _ in range(max_steps):
        msgs = build_messages(question, history)
        response = vlm_fn(msgs)
        code = extract_code_block(response)
        if not code:
            break

        result = kernel.execute(code)
        history.append({"code": code, "result": result})

        if kernel.answered:
            return kernel._answer

    return None


# ---------------------------------------------------------------------------
# Demo — runs without any VLM or perception tools
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    class MockTools:
        def Reconstruct(self, frames):
            class Recon:
                points = np.random.default_rng(0).random((4096, 3)) * 5.0
                depth = np.random.default_rng(0).random((4, 64, 64))
                intrinsics = np.eye(3)
                extrinsics = np.eye(4)
            return Recon()

        def SAM3(self, frame, prompt: str):
            mask = np.zeros((64, 64), dtype=bool)
            mask[:32, :32] = True
            return mask

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    tools = MockTools()
    kernel = PersistentKernel(frames, tools, metadata={"fps": 30, "duration": 4.0})

    # Simulate: agent finds nearest-neighbor distance using brute-force norms (scipy optional)
    code = """\
recon = tools.Reconstruct(InputImages)
pts = recon.points
seg_a = tools.SAM3(InputImages[0], "heater")
seg_b = tools.SAM3(InputImages[0], "door")
# two subsets as stand-ins for segmented point clouds
pts_a = pts[:128]
pts_b = pts[256:384]
# compute pairwise distances with numpy (scipy.spatial.KDTree preferred in real use)
diffs = pts_a[:, None, :] - pts_b[None, :, :]   # (128, 128, 3)
dists = np.sqrt((diffs ** 2).sum(axis=-1))       # (128, 128)
min_dist = float(dists.min())
print(f"Minimum distance between objects: {min_dist:.4f} m")
ReturnAnswer(round(min_dist, 2))
"""

    result = kernel.execute(code)
    print("Status  :", result["status"])
    print("Stdout  :", result["stdout"].strip())
    print("Answer  :", result["answer"])
    print("Variables:", result["variables"])

    # Demonstrate AST safety check
    bad_code = "import os; print(os.listdir('.'))"
    ok, err = ast_safety_check(bad_code)
    print(f"\nAST check on blocked import: ok={ok}, error='{err}'")
