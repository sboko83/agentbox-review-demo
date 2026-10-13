"""Configurable input and output policies for the agent pipeline.

Both policy kinds implement ``apply(context)``. Input policies turn a user
message into a strategy plan; output policies shape a provider request before
it runs and inspect the completed response afterwards. Keeping the context
explicit lets plugins stay independent from HTTP, SQLite and the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Literal, Protocol

from agentbox.core.ports import LLMRequest, ToolCall
from agentbox.core.spec import AgentSpec
from agentbox.core.user_profile import UserProfile


@dataclass(slots=True)
class PromptStage:
    """One hidden preparatory model call of an input strategy."""

    name: str
    prompt: str


@dataclass(slots=True)
class StrategyPlan:
    """Preparatory calls and the final prompt assembled from their output."""

    stages: list[PromptStage] = field(default_factory=list)
    strategy: str = "direct"

    def stage_prompt(self, stage: PromptStage, answers: dict[str, str]) -> str:
        """Give the critic its peers' work without leaking it into memory."""
        if self.strategy == "experts" and stage.name == "critic":
            return (
                f"{stage.prompt}\n\nОТВЕТ АНАЛИТИКА:\n{answers['analyst']}\n\n"
                f"ОТВЕТ ИНЖЕНЕРА:\n{answers['engineer']}"
            )
        return stage.prompt

    def final_prompt(self, question: str, answers: dict[str, str]) -> str:
        if self.strategy == "step_by_step":
            return (
                f"{question}\n\nРешай пошагово. Разбей решение на явные шаги, "
                "покажи промежуточные выводы и только потом дай итог."
            )
        if self.strategy == "self_prompt":
            return answers["self_prompt"]
        if self.strategy == "experts":
            return (
                f"Задача: {question}\n\n"
                f"АНАЛИТИК:\n{answers['analyst']}\n\n"
                f"ИНЖЕНЕР:\n{answers['engineer']}\n\n"
                f"КРИТИК:\n{answers['critic']}\n\n"
                "Ты — ВЕДУЩИЙ группы экспертов. Сведи их работу в один итоговый "
                "ответ. При расхождении перепроверь спорное место сам."
            )
        return question


@dataclass(slots=True)
class InputPolicyContext:
    spec: AgentSpec
    question: str
    # Preparatory stages already receive the profile block with the payload;
    # this copy is for a plugin that shapes its stage prompts by it.
    user_profile: UserProfile | None = None


@dataclass(slots=True)
class OutputPolicyContext:
    spec: AgentSpec
    phase: Literal["before_llm", "after_llm"]
    request: LLMRequest
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # A strategy makes hidden preparatory calls whose output is an intermediate
    # artefact, not the answer. Only the answering call is the user's answer,
    # so only it may be reshaped by a response control.
    final: bool = True


class Policy(Protocol):
    """The common plugin contract used by both sides of the pipeline."""

    def apply(self, context: InputPolicyContext | OutputPolicyContext):
        ...


class InputPolicy(Policy, Protocol):
    def apply(self, context: InputPolicyContext) -> StrategyPlan:
        ...


class OutputPolicy(Policy, Protocol):
    def apply(self, context: OutputPolicyContext) -> None:
        ...


class StrategyInputPolicy:
    """Week-1 strategy variants, represented as an executable prompt plan."""

    def apply(self, context: InputPolicyContext) -> StrategyPlan:
        question = context.question
        strategy = context.spec.reasoning.strategy
        if strategy == "self_prompt":
            return StrategyPlan(
                strategy=strategy,
                stages=[
                    PromptStage(
                        "self_prompt",
                        "Ниже задача. НЕ решай её. Составь самодостаточный промпт, "
                        "который заставит языковую модель решить её максимально "
                        "надёжно и без ошибок. Включи условие задачи целиком. Выведи "
                        f"только текст промпта.\n\nЗадача: {question}",
                    )
                ],
            )
        if strategy == "experts":
            return StrategyPlan(
                strategy=strategy,
                stages=[
                    PromptStage(
                        "analyst",
                        f"Ты — АНАЛИТИК. Разбери задачу, выдели условия и предложи "
                        f"проверяемое решение.\n\nЗадача: {question}",
                    ),
                    PromptStage(
                        "engineer",
                        f"Ты — ИНЖЕНЕР. Реши задачу независимо, проверь граничные "
                        f"случаи и дай свой вывод.\n\nЗадача: {question}",
                    ),
                    PromptStage(
                        "critic",
                        f"Ты — КРИТИК. Найди типичные ошибки, которые стоит избежать "
                        f"при решении этой задачи, и предложи собственную проверку.\n\n"
                        f"Задача: {question}",
                    ),
                ],
            )
        return StrategyPlan(strategy=strategy)


class ResponseControlOutputPolicy:
    """Map the week-1 response-control variants onto one provider request."""

    def apply(self, context: OutputPolicyContext) -> None:
        if context.phase != "before_llm" or not context.final:
            # Intentionally no early return before this point: plugins receive
            # the complete requested-call list after every model response.
            # Shaping a preparatory call would corrupt the artefact the final
            # prompt is built from — a self-written prompt squeezed into the
            # answer's JSON schema is no longer a prompt.
            return
        if context.spec.output.mode == "api":
            context.request.max_output_tokens = context.spec.output.max_output_tokens
            context.request.text_format = json.loads(context.spec.output.format_json)
        elif context.spec.output.mode == "prompt":
            for item in reversed(context.request.messages):
                if item.get("role") == "user" and isinstance(item.get("content"), str):
                    item["content"] += (
                        "\n\nТребования к ответу:\n" + context.spec.output.rules
                    )
                    break


def default_input_policy(_: AgentSpec) -> InputPolicy:
    return StrategyInputPolicy()


def default_output_policy(_: AgentSpec) -> OutputPolicy:
    return ResponseControlOutputPolicy()
