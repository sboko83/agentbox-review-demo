"""Immutable, content-addressed agent configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_ROLE = "Ты — полезный ассистент. Отвечай по существу, без воды."
DEFAULT_API_RESPONSE_FORMAT = json.dumps(
    {
        "type": "json_schema",
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "points": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "points"],
            "additionalProperties": False,
        },
    },
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
DEFAULT_PROMPT_RESPONSE_RULES = (
    "Ответь ровно 3 пунктами, каждый с новой строки и начинается с «- ». "
    "Весь ответ не длиннее 60 слов. Последняя строка — «### КОНЕЦ»."
)

ApprovalMode = Literal["read-only", "ask", "auto"]
ExecutorKind = Literal["local", "docker"]
ReasoningEffort = Literal["low", "medium", "high"]
ReasoningStrategy = Literal["direct", "step_by_step", "self_prompt", "experts"]
JudgeFailureAction = Literal["as_is", "retry", "fallback"]
ContextOverflow = Literal["fail", "drop", "send"]
LongTermWrite = Literal["auto", "confirm", "manual"]


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSection(Section):
    provider: str = Field(
        default="zen", min_length=1, max_length=50, description="LLM-провайдер"
    )
    model: str = Field(default=DEFAULT_MODEL, max_length=100, description="Модель")
    role: str = Field(default=DEFAULT_ROLE, max_length=8000, description="Системная роль")
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Бюджет рассуждений reasoning-модели",
        json_schema_extra={"capability": "reasoning_effort"},
    )


class SamplingSection(Section):
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Разброс выбора токена; пусто — настройка модели",
        json_schema_extra={"capability": "temperature"},
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Доля вероятностной массы для отбора токенов",
        json_schema_extra={"capability": "top_p"},
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Сколько самых вероятных токенов рассматривать",
        json_schema_extra={"capability": "top_k", "advanced": True},
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Штраф за повтор токена пропорционально частоте",
        json_schema_extra={"capability": "frequency_penalty", "advanced": True},
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Штраф за уже встречавшийся токен",
        json_schema_extra={"capability": "presence_penalty", "advanced": True},
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Зерно генератора для воспроизводимого ответа",
        json_schema_extra={"capability": "seed", "advanced": True},
    )
    stop: str = Field(
        default="",
        max_length=400,
        description="Стоп-последовательности, по одной в строке",
        json_schema_extra={"capability": "stop", "advanced": True, "ui": "textarea"},
    )

    def stop_sequences(self) -> tuple[str, ...]:
        return tuple(line for line in self.stop.splitlines() if line)


class OutputSection(Section):
    mode: Literal["free", "api", "prompt"] = Field(
        default="free", description="Как ограничивать формат и длину ответа"
    )
    max_output_tokens: int = Field(
        default=200,
        ge=1,
        le=100_000,
        description="Лимит вывода",
        json_schema_extra={"when": {"field": "mode", "values": ["api", "prompt"]}},
    )
    format_json: str = Field(
        default=DEFAULT_API_RESPONSE_FORMAT,
        max_length=8000,
        description="Значение text.format",
        json_schema_extra={"when": {"field": "mode", "values": ["api"]}, "ui": "textarea"},
    )
    rules: str = Field(
        default=DEFAULT_PROMPT_RESPONSE_RULES,
        max_length=8000,
        description="Инструкции для контроля ответа промптом",
        json_schema_extra={"when": {"field": "mode", "values": ["prompt"]}, "ui": "textarea"},
    )

    @field_validator("format_json")
    @classmethod
    def canonical_format_json(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("text.format должен быть JSON-объектом") from error
        if not isinstance(parsed, dict):
            raise ValueError("text.format должен быть JSON-объектом")
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ReasoningSection(Section):
    strategy: ReasoningStrategy = Field(default="direct", description="Стратегия решения")


class ContextBase(Section):
    window_tokens: int | None = Field(
        default=None,
        ge=100,
        le=2_000_000,
        description="Искусственное окно; пусто — окно модели",
    )
    overflow: ContextOverflow = Field(
        default="drop", description="Действие при переполнении контекста"
    )
    budget_usd: float | None = Field(
        default=None, gt=0, description="Суммарный бюджет сессии в долларах"
    )

    @field_validator("budget_usd", mode="before")
    @classmethod
    def zero_budget_means_unlimited(cls, value: object) -> object:
        return normalize_budget_usd(value)


class NoneContextSection(ContextBase):
    mode: Literal["none"] = "none"


class WindowContextSection(ContextBase):
    mode: Literal["window"] = "window"
    keep_last_messages: int = Field(default=6, ge=1, le=100)


class TokensContextSection(ContextBase):
    mode: Literal["tokens"] = "tokens"
    memory_limit_tokens: int = Field(default=4000, ge=100, le=200_000)


class SummaryContextSection(ContextBase):
    mode: Literal["summary"] = "summary"
    keep_last_messages: int = Field(default=6, ge=1, le=100)
    summary_every: int = Field(default=6, ge=1, le=100)
    summary_model: str = Field(default=DEFAULT_MODEL, max_length=100)
    summary_max_tokens: int = Field(default=300, ge=32, le=4000)



ContextSection = Annotated[
    NoneContextSection
    | WindowContextSection
    | TokensContextSection
    | SummaryContextSection,
    Field(discriminator="mode"),
]


class MemorySection(Section):
    """Layers that outlive the context window, and the router that fills them.

    Flat on purpose: the settings form renders one collapsible per section and
    one control per field, so a nested object would degrade into a JSON
    textarea. `when` hides the long-term controls until the layer is on.
    """

    working_enabled: bool = Field(
        default=False, description="Рабочая память: данные текущей задачи"
    )
    working_max_tokens: int = Field(
        default=400,
        ge=32,
        le=4000,
        description="Бюджет блока рабочей памяти",
        json_schema_extra={"when": {"field": "working_enabled", "truthy": True}},
    )
    longterm_enabled: bool = Field(
        default=False, description="Долговременная память: профиль, решения, знания"
    )
    longterm_scope: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Пространство памяти, общее для сессий",
        json_schema_extra={"when": {"field": "longterm_enabled", "truthy": True}},
    )
    longterm_max_tokens: int = Field(
        default=400,
        ge=32,
        le=4000,
        description="Бюджет блока долговременной памяти",
        json_schema_extra={"when": {"field": "longterm_enabled", "truthy": True}},
    )
    longterm_write: LongTermWrite = Field(
        default="confirm",
        description="Запись в долговременную память",
        json_schema_extra={"when": {"field": "longterm_enabled", "truthy": True}},
    )
    model: str = Field(
        default=DEFAULT_MODEL, max_length=100, description="Модель роутера памяти"
    )
    max_tokens: int = Field(
        default=300, ge=32, le=4000, description="Бюджет ответа роутера"
    )


class TaskSection(Section):
    """The tracker: who marks the steps while the task is being worked on.

    Flat like `memory` for the same reason — the settings form renders one
    control per field. The tracker only ever proposes a stage; `auto_transitions`
    lets it move completed execution into validation. A failed validation needs
    a human to name the failed criteria before returning to execution.
    """

    tracker_enabled: bool = Field(
        default=False, description="Трекер задачи: отмечает шаги после хода"
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        max_length=100,
        description="Модель трекера",
        json_schema_extra={"when": {"field": "tracker_enabled", "truthy": True}},
    )
    max_tokens: int = Field(
        default=200,
        ge=32,
        le=4000,
        description="Бюджет ответа трекера",
        json_schema_extra={"when": {"field": "tracker_enabled", "truthy": True}},
    )
    auto_transitions: bool = Field(
        default=False,
        description="Трекер сам переводит завершённое выполнение в проверку",
        json_schema_extra={"when": {"field": "tracker_enabled", "truthy": True}},
    )


class ToolsSection(Section):
    workspace_path: str | None = Field(
        default=None, max_length=4096, description="Рабочая папка и граница инструментов"
    )
    executor: ExecutorKind = Field(
        default="local",
        description="Среда выполнения инструментов",
        json_schema_extra={"when": {"field": "workspace_path", "truthy": True}},
    )
    approval_mode: ApprovalMode = Field(
        default="ask",
        description="Режим подтверждения",
        json_schema_extra={"when": {"field": "workspace_path", "truthy": True}},
    )
    max_tool_steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Лимит шагов инструментов",
        json_schema_extra={"when": {"field": "workspace_path", "truthy": True}},
    )
    command_timeout_s: float = Field(
        default=30.0,
        gt=0,
        le=600,
        description="Таймаут команды в секундах",
        json_schema_extra={"when": {"field": "workspace_path", "truthy": True}},
    )


class JudgeSection(Section):
    model: str = Field(default=DEFAULT_MODEL, max_length=100, description="Модель судьи")
    criteria: str = Field(
        default="Корректность, полнота и соблюдение требований пользователя.",
        min_length=1,
        max_length=8000,
        description="Критерии оценки",
        json_schema_extra={"ui": "textarea"},
    )
    threshold: int = Field(default=7, ge=0, le=10)
    failure_action: JudgeFailureAction = Field(default="as_is")
    fallback_model: str | None = Field(
        default=None,
        max_length=100,
        json_schema_extra={"when": {"field": "failure_action", "values": ["fallback"]}},
    )

    @model_validator(mode="after")
    def fallback_for_fallback_action(self) -> "JudgeSection":
        if self.failure_action == "fallback" and not self.fallback_model:
            raise ValueError("Для fallback нужна запасная модель")
        return self


def normalize_budget_usd(value: object) -> object:
    """A zero budget is the API spelling of an unlimited budget."""
    if value is None:
        return None
    try:
        return None if float(value) == 0 else value
    except (TypeError, ValueError):
        return value


def migrate_spec_facts(raw: Mapping[str, object]) -> dict:
    """Translate the retired `context.mode=facts` into a window plus a layer.

    The mode was memory and truncation welded together. Split apart, the same
    configuration is a window over the tail and a working-memory layer beside
    it, so the resulting payload is unchanged.
    """
    data = dict(raw)
    context = data.get("context")
    if not isinstance(context, Mapping) or context.get("mode") != "facts":
        return data
    old = dict(context)
    data["context"] = {
        "mode": "window",
        "keep_last_messages": old.get("keep_last_messages", 6),
        "window_tokens": old.get("window_tokens"),
        "overflow": old.get("overflow", "drop"),
        "budget_usd": old.get("budget_usd"),
    }
    memory = dict(data.get("memory") or {})
    memory.setdefault("working_enabled", True)
    memory.setdefault("model", old.get("facts_model", DEFAULT_MODEL))
    memory.setdefault("max_tokens", old.get("facts_max_tokens", 300))
    data["memory"] = memory
    return data


def migrate_spec_v1(raw: Mapping[str, object]) -> dict:
    """Translate one flat v1 payload into the canonical nested v2 shape."""
    data = dict(raw)
    if isinstance(data.get("model"), Mapping) or any(
        key in data
        for key in (
            "sampling", "output", "reasoning", "context", "memory", "tools", "task", "judge"
        )
    ):
        return data

    model = {
        "provider": "zen",
        "model": data.get("model", DEFAULT_MODEL),
        "role": data.get("role", DEFAULT_ROLE),
        "reasoning_effort": data.get("reasoning_effort"),
    }
    sampling = {
        key: data[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
        )
        if key in data
    }
    control = data.get("response_control", "unrestricted")
    output = {
        "mode": {"unrestricted": "free", "api": "api", "prompt": "prompt"}.get(
            control, "free"
        )
    }
    if data.get("max_output_tokens") is not None:
        output["max_output_tokens"] = data["max_output_tokens"]
    if data.get("api_response_format") is not None:
        output["format_json"] = data["api_response_format"]
    if data.get("prompt_response_rules") is not None:
        output["rules"] = data["prompt_response_rules"]

    legacy_context = data.get("context_strategy", "trim")
    common_context = {
        "window_tokens": data.get("context_limit_tokens"),
        "overflow": {"trim": "drop", "fail": "fail"}.get(
            data.get("context_overflow", "trim"), "drop"
        ),
        "budget_usd": data.get("budget_usd"),
    }
    if legacy_context == "summary":
        context = {
            "mode": "summary",
            **common_context,
            "keep_last_messages": data.get("keep_last_messages", 6),
            "summary_every": data.get("summary_every", 6),
            "summary_model": data.get("summary_model", DEFAULT_MODEL),
            "summary_max_tokens": data.get("summary_max_tokens", 300),
        }
    else:
        context = {
            "mode": "tokens",
            **common_context,
            "memory_limit_tokens": data.get("memory_limit_tokens", 4000),
        }

    tools = {
        "workspace_path": data.get("workspace_path"),
        "executor": data.get("executor_kind", "local"),
        "approval_mode": data.get("approval_mode", "ask"),
        "max_tool_steps": data.get("max_tool_steps", 8),
        "command_timeout_s": data.get("command_timeout_s", 30.0),
    }
    judge = {
        "model": data.get("judge_model", DEFAULT_MODEL),
        "criteria": data.get(
            "judge_criteria", "Корректность, полнота и соблюдение требований пользователя."
        ),
        "threshold": data.get("judge_threshold", 7),
        "failure_action": data.get("judge_failure_action", "as_is"),
        "fallback_model": data.get("judge_fallback_model"),
    }
    return {
        "model": model,
        "sampling": sampling,
        "output": output,
        "reasoning": {"strategy": data.get("strategy", "direct")},
        "context": context,
        "tools": tools,
        "judge": judge,
    }


class AgentSpec(BaseModel):
    """One immutable v2 configuration, addressed by canonical JSON hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelSection = Field(default_factory=ModelSection)
    sampling: SamplingSection = Field(default_factory=SamplingSection)
    output: OutputSection = Field(default_factory=OutputSection)
    reasoning: ReasoningSection = Field(default_factory=ReasoningSection)
    context: ContextSection = Field(default_factory=TokensContextSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    task: TaskSection = Field(default_factory=TaskSection)
    judge: JudgeSection = Field(default_factory=JudgeSection)

    @model_validator(mode="before")
    @classmethod
    def read_older_payloads(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        return migrate_spec_facts(migrate_spec_v1(value))

    @property
    def hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
