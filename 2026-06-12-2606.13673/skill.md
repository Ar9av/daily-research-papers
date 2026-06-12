---
name: persistent-kernel-spatial-agent
description: Build tool-augmented spatial reasoning agents using a persistent Python kernel as the action interface. The VLM writes one executable cell per step, conditioned on intermediate text output, variable state, and rendered images from prior steps. Outperforms structured tool-call and single-pass code interfaces across all spatial reasoning categories without model-specific tuning.
trigger: When building VLM agents that need to answer spatial, geometric, or 3D/4D questions from images or video; when structured tool-call APIs fail at test-time compositions (KDTree queries, RANSAC fits, custom geometry); when you need the same agent to generalize across diverse spatial task types without per-task prompt engineering.
---

## When to use

- You're building a spatial reasoning agent (distance, direction, camera motion, temporal ordering, depth) on top of a VLM
- Structured tool-calls can't express the composition your task needs at test time
- You want one agent to handle single-image, multi-view, and video spatial questions without retraining
- Post-processing tool outputs requires task-specific numerical computation that no predefined API can anticipate

## Pattern

1. **Init persistent kernel**: create one Python execution context per sample; pre-load InputImages, perception tools, and scientific libraries
2. **Front-load planning**: run a separate planner LLM (no images) to produce an analysis plan; append it to the main agent's system prompt
3. **Cell generation loop**: main VLM writes one Python cell per step targeting the next evidence acquisition
4. **AST safety check**: parse cell AST before execution; reject unsafe builtins/imports; return traceback if invalid
5. **Execute + assemble feedback**: run cell in persistent kernel; collect stdout, variable summaries, show() images; append to next model context
6. **Answer submission**: agent calls ReturnAnswer() when evidence is sufficient; loop terminates

## Implementation

```python
# spatialclaw_kernel.py — persistent kernel + agentic loop scaffold

import ast, io, sys, traceback
from contextlib import redirect_stdout

BLOCKED_MODULES = {"os", "subprocess", "sys", "shutil", "socket"}
BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "open"}

def ast_safety_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    return False, f"Blocked import: {alias.name}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                return False, f"Blocked builtin: {node.func.id}"
    return True, ""


class PersistentKernel:
    def __init__(self, frames, tools, metadata=None):
        self._ns = {
            "InputImages": frames,
            "tools": tools,
            "Metadata": metadata or {},
            "show": self._show,
            "ReturnAnswer": self._return_answer,
        }
        try:
            import numpy as np
            import scipy
            import matplotlib.pyplot as plt
            self._ns.update({"np": np, "scipy": scipy, "plt": plt})
        except ImportError:
            pass
        self._shown_images = []
        self._answer = None

    def _show(self, img, label=""):
        self._shown_images.append((label, img))

    def _return_answer(self, answer):
        self._answer = answer

    def execute(self, code: str) -> dict:
        ok, err = ast_safety_check(code)
        if not ok:
            return {"status": "blocked", "error": err, "stdout": "", "images": []}

        buf = io.StringIO()
        self._shown_images = []
        try:
            with redirect_stdout(buf):
                exec(code, self._ns)
            status = "ok"
            error = ""
        except Exception:
            status = "error"
            error = traceback.format_exc()

        # Variable summary: type + shape/len for new numpy arrays and lists
        var_summary = {
            k: f"{type(v).__name__} shape={v.shape}" if hasattr(v, "shape")
               else f"{type(v).__name__} len={len(v)}" if hasattr(v, "__len__") and not isinstance(v, str)
               else type(v).__name__
            for k, v in self._ns.items()
            if not k.startswith("_") and k not in ("InputImages", "tools", "Metadata",
                                                     "show", "ReturnAnswer", "np", "scipy", "plt")
        }

        return {
            "status": status,
            "stdout": buf.getvalue(),
            "error": error,
            "images": list(self._shown_images),
            "answer": self._answer,
            "variables": var_summary,
        }

    def answered(self):
        return self._answer is not None


def run_spatial_agent(frames, tools, metadata, question, vlm_generate_fn, max_steps=30):
    """
    vlm_generate_fn(messages: list[dict]) -> str
    Returns the submitted answer or None if max_steps reached.
    """
    kernel = PersistentKernel(frames, tools, metadata)
    history = []

    for step in range(max_steps):
        prompt = build_prompt(question, history)
        response = vlm_generate_fn(prompt)
        code = extract_code_block(response)

        if code is None:
            break

        result = kernel.execute(code)
        history.append({"code": code, "result": result})

        if kernel.answered():
            return kernel._answer

    return None


def build_prompt(question, history):
    # Assemble: system prompt + question + execution trajectory
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.append({"role": "user", "content": question})
    for turn in history:
        msgs.append({"role": "assistant", "content": f"```python\n{turn['code']}\n```"})
        fb = turn["result"]
        feedback = f"Status: {fb['status']}\n"
        if fb["stdout"]:
            feedback += f"Stdout:\n{fb['stdout']}\n"
        if fb["error"]:
            feedback += f"Error:\n{fb['error']}\n"
        if fb["variables"]:
            feedback += "Variables: " + str(fb["variables"]) + "\n"
        msgs.append({"role": "user", "content": feedback})
    return msgs


def extract_code_block(text: str) -> str | None:
    import re
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


SYSTEM_PROMPT = """
You are a spatial reasoning agent. You have access to a persistent Python kernel.
At each step, write one Python cell. Use show() to inspect intermediate results visually.
Call ReturnAnswer(answer) when you have sufficient evidence.
Prefer metric computation over pixel-level estimates for geometric questions.
Verify segmentation masks visually before computing distances or directions.
""".strip()


if __name__ == "__main__":
    # Demo with mock tools and frames
    import numpy as np

    class MockTools:
        def Reconstruct(self, frames):
            class R:
                depth = np.random.rand(4, 64, 64)
                points = np.random.rand(4096, 3)
                intrinsics = np.eye(3)
                extrinsics = np.eye(4)
            return R()

        def SAM3(self, frame, prompt):
            return np.ones((64, 64), dtype=bool)

    frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 4
    kernel = PersistentKernel(frames, MockTools())

    result = kernel.execute("""
import scipy.spatial
pts = tools.Reconstruct(InputImages).points
tree = scipy.spatial.KDTree(pts[:100])
dist, _ = tree.query(pts[200:201], k=1)
print(f"Nearest distance: {dist[0]:.4f}")
""")
    print("Status:", result["status"])
    print("Stdout:", result["stdout"])
    print("Variables:", result["variables"])
