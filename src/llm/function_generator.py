from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from src.config.config import LLMConfig, MetaConfig
from src.llm.client import LLMClient
from src.llm.function_templates import TemplateFunctionSet
from src.utils.query_functions import ContributionQuerySet, ObservationQuerySet, StateQuerySet


@dataclass
class HistoryItem:
    name: str
    score: float
    avg_revenue: float
    avg_success_tasks: float
    stability: float
    spec: dict[str, Any]


FUNCTION_ORDER = ["state", "observation", "contribution"]


class FunctionGenerator:
    """LLM-based candidate function set generator."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(
                model=LLMConfig.OPENAI_MODEL,
                api_key=LLMConfig.OPENAI_API_KEY,
                base_url=LLMConfig.OPENAI_BASE_URL,
            )
        return self._client

    def _history_text(self, history: list[HistoryItem]) -> str:
        if not history:
            return (
                "Current round type: initial generation.\n"
                "There are no historical function sets available for reference in this round."
            )

        contract = TemplateFunctionSet.prompt_contract()
        lines = [
            "Current round type: iterative optimization.",
            "Historical function sets and their performance records:",
        ]
        for idx, item in enumerate(history, start=1):
            spec = TemplateFunctionSet.from_dict(item.spec)
            lines.extend(
                [
                    f"History #{idx}: name={item.name}",
                    (
                        "Performance: "
                        f"score={item.score:.4f}, revenue={item.avg_revenue:.4f}, "
                        f"success={item.avg_success_tasks:.4f}, stability={item.stability:.4f}"
                    ),
                    "Full function set:",
                    f"{contract['state']['symbol']} / {contract['state']['function_name']}:",
                    spec.state_function_code,
                    f"{contract['observation']['symbol']} / {contract['observation']['function_name']}:",
                    spec.observation_function_code,
                    f"{contract['contribution']['symbol']} / {contract['contribution']['function_name']}:",
                    spec.contribution_function_code,
                ]
            )
        return "\n".join(lines)

    def _query_schema_text(self) -> str:
        examples = {
            "state": StateQuerySet.prompt_example(),
            "observation": ObservationQuerySet.prompt_example(),
            "contribution": ContributionQuerySet.prompt_example(),
        }
        lines: list[str] = []
        for name, example in examples.items():
            methods = example.get("methods", {}) if isinstance(example, dict) else {}
            lines.append(f"- {name} query methods: {list(methods.keys())}")
            lines.append(f"  query interface schema by method: {json.dumps(example, ensure_ascii=True)}")
        return "\n".join(lines)

    def build_user_prompt(
        self,
        history: list[HistoryItem],
        kind: str,
        round_idx: int,
        target_key: str,
        generated_functions: dict[str, dict[str, str]],
    ) -> str:
        contract = TemplateFunctionSet.prompt_contract()

        target_lines: list[str] = []
        for key in FUNCTION_ORDER:
            section = contract[key]
            target_lines.append(
                f"- {section['symbol']} / `{section['function_name']}`: {section['description']}"
            )

        status_lines = [
            f"This prompt must generate only {contract[target_key]['symbol']} "
            f"({contract[target_key]['function_name']})."
        ]
        if not generated_functions:
            status_lines.append("No functions in the current set have been generated yet.")
        else:
            generated_names = ", ".join(contract[key]["symbol"] for key in FUNCTION_ORDER if key in generated_functions)
            status_lines.append(f"Already generated functions in the current set: {generated_names}.")
            status_lines.append("Generated function context for the current set:")
            for key in FUNCTION_ORDER:
                if key not in generated_functions:
                    continue
                generated = generated_functions[key]
                status_lines.extend(
                    [
                        f"{contract[key]['symbol']} / {contract[key]['function_name']}:",
                        generated["description"],
                        generated["code"],
                    ]
                )

        pending_names = ", ".join(
            contract[key]["symbol"] for key in FUNCTION_ORDER if key not in generated_functions and key != target_key
        )
        if pending_names:
            status_lines.append(f"Other functions not yet generated in the current set: {pending_names}.")

        candidate_name = ""
        for key in FUNCTION_ORDER:
            if key in generated_functions and generated_functions[key]["candidate_name"].strip():
                candidate_name = generated_functions[key]["candidate_name"].strip()
                break
        if candidate_name:
            status_lines.append(
                f"The candidate set name for this round has already been fixed as `{candidate_name}`; reuse it exactly."
            )
        else:
            status_lines.append("Choose one concise candidate set name and reuse it for later functions in this set.")

        schema = json.dumps(
            {
                "candidate_name": "string",
                "kind": kind,
                "round_idx": round_idx,
                "target_function": contract[target_key]["symbol"],
                "function_name": contract[target_key]["function_name"],
                "function_description": "string",
                "function_code": "python source code",
            },
            ensure_ascii=True,
            indent=2,
        )

        return (
            "Historical function set performance context:\n"
            f"{self._history_text(history)}\n\n"

            "Output policy:\n"
            "Return only the final answer.\n"
            "Do not output any thinking, reasoning, analysis, scratchpad, or intermediate process.\n"
            "Do not wrap the answer with any extra text before or after the final JSON object.\n\n"
            "Task description:\n"
            "You are helping a CTA (Complex Task Assignment) reinforcement learning framework by generating the auxiliary function set.\n"
            "The auxiliary function set has three functions and each prompt must still consider all three functions.\n"
            "Their goal is to improve representation learning and credit assignment by reorganizing structured query statistics into semantic vectors.\n"
            "The three target functions are:\n"
            f"{chr(10).join(target_lines)}\n\n"
            f"""
            CTA framework overview:
            The CTA (Complex Task Assignment) problem considers assigning workers to complex tasks composed of multiple interdependent subtasks.
            Each subtask requires specific skills and may have prerequisite dependencies that constrain its execution order.
            A task can be successfully completed only when all required subtasks are finished while respecting dependency, skill, and spatio-temporal constraints.

            The reinforcement learning framework learns assignment policies for maximizing long-term task completion quality and overall revenue.
            Since task rewards are observed only after multiple related subtasks are completed, the framework faces a challenging temporal credit assignment problem.
            To address this issue, the framework constructs semantic representations from structured task-worker statistics and uses them to support both representation learning and reward attribution.

            The generated functions are components of this semantic representation module.
            They transform low-level query statistics into fixed-length feature vectors that are subsequently consumed by neural networks for state encoding and credit estimation.
            Therefore, the quality of generated functions directly affects how effectively the framework captures task dependencies, worker-skill compatibility, supply-demand dynamics, execution feasibility, and long-term task contributions.
            
            The generated functions are evolved as part of an LLM-guided function optimization process.
            Their effectiveness is ultimately evaluated by the downstream reinforcement learning performance, including task revenue, successful task completion, policy stability, and credit assignment quality.
            Therefore, functions should focus on generating useful semantic abstractions rather than simply exposing raw statistics.
            
            If historical function sets are available, use their performance records as empirical feedback, and generate improved functions.
            Identify potentially beneficial or detrimental feature-construction patterns, preserve useful behaviors when appropriate, and explore improved semantic representations.
            When no historical functions are available, focus on generating diverse and meaningful semantic abstractions guided by the CTA framework description.            
            
            High-quality functions should:
            (1) capture informative interactions among task structure, worker skills, and resource availability;
            (2) produce stable and discriminative representations across diverse assignment states;
            (3) expose signals useful for estimating subtask importance and delayed rewards;
            (4) improve downstream policy learning and credit assignment rather than merely encoding redundant statistics;
            (5) generalize across different tasks and environments.
            """
            "Current generation status for this candidate set:\n"
            f"{chr(10).join(status_lines)}\n\n"
            "Semantic block contract:\n"
            f"{json.dumps(contract, ensure_ascii=True, indent=2)}\n\n"
            "Real query interfaces and examples:\n"
            f"{self._query_schema_text()}\n\n"
            "Query calling contract:\n"
            "The `query` argument is a live query object, not a precomputed dictionary.\n"
            "Call only the specific query methods you need, such as `query.skill_supply_demand_ratio()`.\n"
            "Do not use `query_all`, do not prefetch every method, and do not access query results by dictionary indexing.\n\n"
            "Output specification:\n"
            "Return exactly one JSON object with the following schema:\n"
            f"{schema}\n\n"
            "Constraints:\n"
            "1) Output JSON only.\n"
            "2) Generate only the current target function for this prompt; do not generate the other two function codes.\n"
            "3) Use only query method calls and local computation, no imports, no files, no network, no randomness.\n"
            "4) Return numeric vectors with exact target lengths.\n"
            "5) Every returned vector element must be finite; never return inf, -inf, or nan.\n"
            "6) When computing ratios or averages, use finite values and guard empty denominators.\n"
            "7) Add concise inline comments in the generated code when helpful.\n"
            "8) Do not include any reasoning, analysis, or intermediate explanation outside the final JSON.\n"
        )

    def generate_function_set(
        self,
        history: list[HistoryItem],
        kind: str,
        round_idx: int,
    ) -> TemplateFunctionSet:
        total_attempts = MetaConfig.generator_retry + 1
        last_error: Exception | None = None

        for _ in range(total_attempts):
            try:
                generated_functions: dict[str, dict[str, str]] = {}

                for target_key in FUNCTION_ORDER:
                    prompt = self.build_user_prompt(history, kind, round_idx, target_key, generated_functions)
                    text = self._get_client().chat(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return exactly one JSON object for the current CTA auxiliary function generation step. "
                                    "Output final result only. Do not output any reasoning or intermediate process."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_new_tokens=LLMConfig.MAX_NEW_TOKENS,
                        do_sample=MetaConfig.generator_temperature > 0.0,
                        temperature=MetaConfig.generator_temperature,
                    )
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        raise ValueError("LLM response must be a JSON object")

                    generated_functions[target_key] = {
                        "candidate_name": str(payload.get("candidate_name", "")).strip(),
                        "description": str(payload.get("function_description", "")).strip(),
                        "code": str(payload["function_code"]).strip(),
                    }

                candidate_name = ""
                for key in FUNCTION_ORDER:
                    candidate_name = generated_functions[key]["candidate_name"]
                    if candidate_name:
                        break
                if not candidate_name:
                    candidate_name = f"{kind}_candidate_{round_idx}"

                spec = TemplateFunctionSet(
                    name=candidate_name,
                    kind=kind,
                    round_idx=round_idx,
                    state_function_code=generated_functions["state"]["code"],
                    observation_function_code=generated_functions["observation"]["code"],
                    contribution_function_code=generated_functions["contribution"]["code"],
                )
                spec.validate()
                return spec
            except Exception as exc:
                last_error = exc
                continue

        if last_error is None:
            raise RuntimeError(f"LLM generation failed for {kind} round {round_idx}")
        raise RuntimeError(
            f"LLM generation failed for {kind} round {round_idx}: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def generate_initial_candidate(self, idx: int) -> TemplateFunctionSet:
        return self.generate_function_set([], "initial", idx)

    def generate_next_candidate(self, history: list[HistoryItem], round_idx: int) -> TemplateFunctionSet:
        return self.generate_function_set(history, "iter", round_idx)
