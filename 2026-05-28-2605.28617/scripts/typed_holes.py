"""
LACUNA — Typed holes: fill, type-check, revision loop
arxiv: 2605.28617

Run:
    python scripts/typed_holes.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Type, Callable
import re


# ── Domain-specific safe types ───────────────────────────────────────────────

class AllowedPath(str):
    ALLOWED_PREFIXES = ("/tmp/", "/home/", "./")

    @classmethod
    def validate(cls, value: str) -> "AllowedPath":
        if not any(value.startswith(p) for p in cls.ALLOWED_PREFIXES):
            raise TypeError(
                f"Path {value!r} not in allowed prefixes {cls.ALLOWED_PREFIXES}. "
                f"Only read/write within: {', '.join(cls.ALLOWED_PREFIXES)}"
            )
        return cls(value)


class BoundedInt(int):
    def __new__(cls, value: int, min_val: int = 0, max_val: int = 1000):
        if not (min_val <= value <= max_val):
            raise TypeError(f"Value {value} out of bounds [{min_val}, {max_val}]")
        return super().__new__(cls, value)


class ApprovedURL(str):
    BLOCKED_DOMAINS = ("evil.com", "malware.io", "phishing.net")

    @classmethod
    def validate(cls, value: str) -> "ApprovedURL":
        for domain in cls.BLOCKED_DOMAINS:
            if domain in value:
                raise TypeError(f"URL {value!r} contains blocked domain {domain!r}")
        return cls(value)


# ── Typed Hole infrastructure ─────────────────────────────────────────────────

@dataclass
class TypedHole:
    """
    A placeholder in a program that the LLM must fill.
    Before execution, the filled value is type-checked.
    """
    name: str
    expected_type: Type
    description: str
    validator: Callable | None = None  # optional additional runtime check

    def fill(self, value: Any) -> Any:
        """Fill this hole and type-check the result. Raises TypeError on failure."""
        # Basic type coercion attempt
        if self.validator:
            return self.validator(value)
        if not isinstance(value, self.expected_type):
            try:
                value = self.expected_type(value)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Hole {self.name!r} expects {self.expected_type.__name__}, "
                    f"got {type(value).__name__}: {e}"
                ) from e
        return value


@dataclass
class ProgramWithHoles:
    """
    A well-typed program skeleton with holes the LLM fills in.
    Ensures type safety before any execution.
    """
    name: str
    holes: list[TypedHole]
    execute_fn: Callable  # called with filled hole values on success

    def fill_and_run(self, filled: dict[str, Any], revision_budget: int = 3) -> Any:
        """
        Attempt to fill all holes and type-check. On failure, report the error
        and allow the caller to revise (simulates the agent revision loop).
        """
        for attempt in range(1, revision_budget + 1):
            errors = []
            filled_values = {}

            for hole in self.holes:
                raw = filled.get(hole.name)
                if raw is None:
                    errors.append(f"Hole {hole.name!r} not filled ({hole.description})")
                    continue
                try:
                    filled_values[hole.name] = hole.fill(raw)
                except TypeError as e:
                    errors.append(str(e))

            if not errors:
                print(f"  [✓] All holes filled and type-checked on attempt {attempt}")
                return self.execute_fn(**filled_values)
            else:
                print(f"  [✗] Attempt {attempt} type errors:")
                for err in errors:
                    print(f"      • {err}")
                if attempt < revision_budget:
                    # In production: send errors back to LLM for revision
                    filled = _simulate_revision(filled, errors)
                else:
                    raise RuntimeError(f"Failed to fill {self.name!r} after {revision_budget} attempts")


def _simulate_revision(filled: dict, errors: list[str]) -> dict:
    """Simulate the agent revising its output after seeing type errors."""
    revised = dict(filled)
    for err in errors:
        if "path" in err.lower() or "AllowedPath" in err:
            revised["path"] = "/tmp/safe_output.txt"  # agent corrects to allowed path
        if "url" in err.lower() or "ApprovedURL" in err:
            revised["url"] = "https://api.example.com/data"  # agent corrects URL
        if "out of bounds" in err.lower():
            revised["count"] = 10  # agent corrects to in-bounds value
    return revised


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LACUNA typed holes demo\n")

    # Define a program with typed holes
    file_write_program = ProgramWithHoles(
        name="WriteFile",
        holes=[
            TypedHole("path", AllowedPath, "file path to write to", AllowedPath.validate),
            TypedHole("count", int, "number of lines to write (0-100)"),
        ],
        execute_fn=lambda path, count: print(f"  → Would write {count} lines to {path!r}"),
    )

    print("=== Test 1: Agent tries to write to /etc/passwd (unsafe path) ===")
    try:
        file_write_program.fill_and_run({"path": "/etc/passwd", "count": 5})
    except RuntimeError as e:
        print(f"  [blocked] {e}")

    print("\n=== Test 2: Agent provides out-of-bounds count, then corrects ===")
    file_write_program.fill_and_run({"path": "/tmp/output.txt", "count": 9999})

    print("\n=== Test 3: URL safety check ===")
    fetch_program = ProgramWithHoles(
        name="FetchURL",
        holes=[
            TypedHole("url", ApprovedURL, "URL to fetch", ApprovedURL.validate),
        ],
        execute_fn=lambda url: print(f"  → Would fetch {url!r}"),
    )
    fetch_program.fill_and_run({"url": "https://evil.com/steal-data"})
