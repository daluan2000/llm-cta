from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

from src.config.config import ModelConfig


STATE_BLOCKS = [
    ("dep", "Global dependency status"),
    ("skl", "Global skill supply"),
    ("tas", "Global task state"),
    ("wok", "Global worker state"),
]
OBSERVATION_BLOCKS = [
    ("dep", "Local dependency status"),
    ("skl", "Local skill status"),
    ("wok", "Local spatiotemporal feasibility"),
]
CONTRIBUTION_BLOCKS = [
    ("rev", "Revenue contribution"),
    ("dep", "Dependency satisfaction"),
    ("skl", "Skill utilization"),
    ("com", "Competition impact"),
]

STATE_FUNCTION_NAME = "build_state_aug"
OBSERVATION_FUNCTION_NAME = "build_observation_aug"
CONTRIBUTION_FUNCTION_NAME = "build_contribution_repr"


class FunctionExecutionError(RuntimeError):
    """Raised when a generated function cannot be compiled or executed."""


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"non-finite numeric value: {value!r}")
        return out
    raise ValueError(f"unsupported numeric value type: {type(value).__name__}")


def _normalize_vector(value: Any, expected_dim: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must return a list or tuple")
    raw = list(value)
    if len(raw) != expected_dim:
        raise ValueError(f"{label} must return length {expected_dim}, got {len(raw)}")
    return [_to_float(item) for item in raw]


def _block_dimensions(blocks: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    for idx, (name, _) in enumerate(blocks):
        mapping[name] = (idx * 2, idx * 2 + 1)
    return mapping


SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    },
    "math": math,
}


def _compile_function(code: str, function_name: str) -> Callable[[Any], list[float]]:
    namespace: dict[str, Any] = {}
    try:
        exec(code, dict(SAFE_GLOBALS), namespace)
    except Exception as exc:
        raise FunctionExecutionError(f"compile {function_name} failed: {exc}") from exc

    fn = namespace.get(function_name)
    if not callable(fn):
        raise FunctionExecutionError(f"compiled code does not define callable '{function_name}'")
    return fn


def _state_dim() -> int:
    return ModelConfig.llm_state_aug_dim


def _observation_dim() -> int:
    return ModelConfig.llm_observation_aug_dim


def _contribution_dim() -> int:
    return ModelConfig.llm_contribution_dim


def _default_state_code() -> str:
    return """
def build_state_aug(query):
    return [0.0] * 8
""".strip()


def _default_observation_code() -> str:
    return """
def build_observation_aug(query):
    return [0.0] * 6
""".strip()


def _default_contribution_code() -> str:
    return """
def build_contribution_repr(query):
    return [0.0] * 8
""".strip()


@dataclass
class TemplateFunctionSet:
    """Executable container for one generated auxiliary function set."""

    name: str
    kind: str
    round_idx: int
    state_function_code: str
    observation_function_code: str
    contribution_function_code: str
    _state_fn: Callable[[Any], list[float]] | None = field(default=None, init=False, repr=False)
    _observation_fn: Callable[[Any], list[float]] | None = field(default=None, init=False, repr=False)
    _contribution_fn: Callable[[Any], list[float]] | None = field(default=None, init=False, repr=False)

    @staticmethod
    def state_spec() -> dict[str, Any]:
        return {
            "symbol": "F_s",
            "function_name": STATE_FUNCTION_NAME,
            "description": (
                "Generate the global state augmentation function that calls state-query methods "
                "on demand and transforms the returned statistics into a compact semantic vector for the critic."
            ),
            "output_dim": _state_dim(),
            "blocks": [
                {
                    "name": block,
                    "description": desc,
                    "dims": list(_block_dimensions(STATE_BLOCKS)[block]),
                }
                for block, desc in STATE_BLOCKS
            ],
        }

    @staticmethod
    def observation_spec() -> dict[str, Any]:
        return {
            "symbol": "F_o",
            "function_name": OBSERVATION_FUNCTION_NAME,
            "description": (
                "Generate the local observation augmentation function that calls observation-query methods "
                "on demand and transforms the returned statistics into a compact semantic vector for the policy."
            ),
            "output_dim": _observation_dim(),
            "blocks": [
                {
                    "name": block,
                    "description": desc,
                    "dims": list(_block_dimensions(OBSERVATION_BLOCKS)[block]),
                }
                for block, desc in OBSERVATION_BLOCKS
            ],
        }

    @staticmethod
    def contribution_spec() -> dict[str, Any]:
        return {
            "symbol": "G",
            "function_name": CONTRIBUTION_FUNCTION_NAME,
            "description": (
                "Generate the contribution representation function that calls post-decision query methods "
                "on demand and maps the returned statistics into a compact semantic vector for credit assignment."
            ),
            "output_dim": _contribution_dim(),
            "blocks": [
                {
                    "name": block,
                    "description": desc,
                    "dims": list(_block_dimensions(CONTRIBUTION_BLOCKS)[block]),
                }
                for block, desc in CONTRIBUTION_BLOCKS
            ],
        }

    @staticmethod
    def prompt_contract() -> dict[str, Any]:
        return {
            "state": TemplateFunctionSet.state_spec(),
            "observation": TemplateFunctionSet.observation_spec(),
            "contribution": TemplateFunctionSet.contribution_spec(),
        }

    @staticmethod
    def default() -> "TemplateFunctionSet":
        return TemplateFunctionSet(
            name="default_code_set",
            kind="default",
            round_idx=0,
            state_function_code=_default_state_code(),
            observation_function_code=_default_observation_code(),
            contribution_function_code=_default_contribution_code(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "round_idx": self.round_idx,
            "state_function_code": self.state_function_code,
            "observation_function_code": self.observation_function_code,
            "contribution_function_code": self.contribution_function_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateFunctionSet":
        base = cls.default()
        if not isinstance(data, dict):
            return base

        raw = data.get("spec") if isinstance(data.get("spec"), dict) else data
        if not isinstance(raw, dict):
            return base

        name = raw.get("name")
        kind = raw.get("kind")
        round_idx = raw.get("round_idx")

        def _string_field(key: str, fallback: str) -> str:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
            return fallback

        return cls(
            name=name.strip() if isinstance(name, str) and name.strip() else base.name,
            kind=kind.strip() if isinstance(kind, str) and kind.strip() else base.kind,
            round_idx=round_idx if isinstance(round_idx, int) else base.round_idx,
            state_function_code=_string_field("state_function_code", base.state_function_code),
            observation_function_code=_string_field("observation_function_code", base.observation_function_code),
            contribution_function_code=_string_field("contribution_function_code", base.contribution_function_code),
        )

    def _ensure_compiled(self) -> None:
        if self._state_fn is None:
            self._state_fn = _compile_function(self.state_function_code, STATE_FUNCTION_NAME)
        if self._observation_fn is None:
            self._observation_fn = _compile_function(self.observation_function_code, OBSERVATION_FUNCTION_NAME)
        if self._contribution_fn is None:
            self._contribution_fn = _compile_function(self.contribution_function_code, CONTRIBUTION_FUNCTION_NAME)

    def validate(self) -> None:
        self._ensure_compiled()

    def build_state_aug(self, state_query: Any) -> list[float]:
        self._ensure_compiled()
        assert self._state_fn is not None
        try:
            return _normalize_vector(self._state_fn(state_query), _state_dim(), STATE_FUNCTION_NAME)
        except Exception as exc:
            raise FunctionExecutionError(f"{self.name}.{STATE_FUNCTION_NAME} failed: {exc}") from exc

    def build_observation_aug(self, observation_query: Any) -> list[float]:
        self._ensure_compiled()
        assert self._observation_fn is not None
        try:
            return _normalize_vector(
                self._observation_fn(observation_query),
                _observation_dim(),
                OBSERVATION_FUNCTION_NAME,
            )
        except Exception as exc:
            raise FunctionExecutionError(f"{self.name}.{OBSERVATION_FUNCTION_NAME} failed: {exc}") from exc

    def build_contribution_repr(self, contribution_query: Any) -> list[float]:
        self._ensure_compiled()
        assert self._contribution_fn is not None
        try:
            return _normalize_vector(
                self._contribution_fn(contribution_query),
                _contribution_dim(),
                CONTRIBUTION_FUNCTION_NAME,
            )
        except Exception as exc:
            raise FunctionExecutionError(f"{self.name}.{CONTRIBUTION_FUNCTION_NAME} failed: {exc}") from exc
