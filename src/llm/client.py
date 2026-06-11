import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config.config import LLMConfig


class LLMOutputValidationError(RuntimeError):
    """Raised when the model returns reasoning/process text instead of final-only output."""


class LLMClient:
    _raw_io_root_override: ContextVar[Path | None] = ContextVar(
        "llm_raw_io_root_override",
        default=None,
    )
    _raw_io_dir_name = "llm_raw_io"


    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ) -> None:

        self.model_name = model or LLMConfig.OPENAI_MODEL
        self.api_key = api_key or LLMConfig.OPENAI_API_KEY
        self.base_url = base_url or LLMConfig.OPENAI_BASE_URL

        if not self.api_key:
            raise ValueError("未配置 LLMConfig.OPENAI_API_KEY，无法调用远程 LLM 接口")

        client_kwargs: Dict[str, object] = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)
        self._raw_io_dir = Path()
        self._raw_io_counter = 1
        self._sync_raw_io_state()

    @classmethod
    @contextmanager
    def raw_io_log_root(cls, root: str | Path | None):
        token = cls._raw_io_root_override.set(Path(root) if root is not None else None)
        try:
            yield
        finally:
            cls._raw_io_root_override.reset(token)

    @classmethod
    def _resolve_raw_io_dir(cls) -> Path:
        root = cls._raw_io_root_override.get()
        if root is None:
            return Path()
        return Path(root) / cls._raw_io_dir_name

    @staticmethod
    def _next_raw_io_index(raw_io_dir: Path) -> int:
        max_idx = 0
        if raw_io_dir.exists():
            for path in raw_io_dir.glob("*.txt"):
                try:
                    idx = int(path.stem)
                except ValueError:
                    continue
                max_idx = max(max_idx, idx)
        return max_idx + 1

    def _sync_raw_io_state(self) -> None:
        raw_io_dir = self._resolve_raw_io_dir()
        if raw_io_dir != self._raw_io_dir:
            self._raw_io_dir = raw_io_dir
            if self._raw_io_dir:
                self._raw_io_dir.mkdir(parents=True, exist_ok=True)
                self._raw_io_counter = self._next_raw_io_index(self._raw_io_dir)
            else:
                self._raw_io_counter = 1

    @staticmethod
    def _inject_output_policy(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return messages

        out = [dict(message) for message in messages]
        directives: list[str] = []
        if LLMConfig.DISABLE_THINKING:
            directives.append(LLMConfig.NO_THINK_DIRECTIVE)
        if LLMConfig.REQUIRE_FINAL_ONLY_OUTPUT:
            directives.append(
                "Output final result only. Do not output any reasoning, thinking process, analysis, "
                "scratchpad, or intermediate explanation."
            )
        if not directives:
            return out

        policy_text = "\n".join(directives)
        if out[0].get("role") == "system":
            original = out[0].get("content", "")
            out[0]["content"] = f"{policy_text}\n{original}".strip()
            return out

        out.insert(0, {"role": "system", "content": policy_text})
        return out

    @staticmethod
    def _validate_final_only_output(text: str) -> None:
        if not LLMConfig.STRICT_REASONING_TEXT_VALIDATION:
            return

        patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"<reasoning>.*?</reasoning>",
            r"思考过程\s*[:：]",
            r"推理过程\s*[:：]",
            r"chain of thought\s*[:：]",
            r"reasoning\s*[:：]",
            r"analysis\s*[:：]",
        ]
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                raise LLMOutputValidationError(
                    "Model output contains reasoning/process text; final-only output required."
                )

    def _write_raw_io_log(self, messages: List[Dict[str, str]], response_text: str) -> None:
        self._sync_raw_io_state()
        if not self._raw_io_dir:
            return
        index = self._raw_io_counter
        self._raw_io_counter += 1
        path = self._raw_io_dir / f"{index}.txt"
        lines = ["[INPUT]"]
        for message in messages:
            lines.append(f"{message.get('role', '')}: {message.get('content', '')}")
        lines.append("")
        lines.append("[OUTPUT]")
        lines.append(response_text)
        path.write_text("\n".join(lines), encoding="utf-8")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        print_token_usage: bool = True,
    ) -> str:
        if not do_sample:
            temperature = 0.0

        messages = self._inject_output_policy(messages)

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if LLMConfig.ENABLE_JSON_MODE:
            request_kwargs["response_format"] = {"type": "json_object"}
        if not LLMConfig.ENABLE_THINKING:
            request_kwargs["extra_body"] = {"enable_thinking": False}

        response = self.client.chat.completions.create(**request_kwargs)

        content = response.choices[0].message.content
        text = (content or "").strip()
        self._validate_final_only_output(text)
        self._write_raw_io_log(messages, text)

        if print_token_usage:
            usage = response.usage

            def _uget(obj: object, key: str) -> Optional[int]:
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    value = obj.get(key)
                else:
                    value = getattr(obj, key, None)
                if isinstance(value, int):
                    return value
                return None

            input_tokens = _uget(usage, "prompt_tokens")
            output_tokens = _uget(usage, "completion_tokens")
            total_tokens = _uget(usage, "total_tokens")

            completion_details = None
            if usage is not None:
                if isinstance(usage, dict):
                    completion_details = usage.get("completion_tokens_details")
                else:
                    completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = _uget(completion_details, "reasoning_tokens")

            print(
                "[LLMClient] token usage - "
                f"input: {input_tokens if input_tokens is not None else 'N/A'}, "
                f"output: {output_tokens if output_tokens is not None else 'N/A'}, "
                f"reasoning: {reasoning_tokens if reasoning_tokens is not None else 'N/A'}, "
                f"total: {total_tokens if total_tokens is not None else 'N/A'}, "
                f"visible_chars: {len(text)}"
            )

        return text
