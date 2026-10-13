"""The user profile: declared preferences on how to answer this person.

It describes the human, not the agent, so it lives next to the memory space
rather than inside `AgentSpec`: a profile in the spec would change the config
hash and split every preset into "preset × person". Versions are immutable and
addressed by the hash of their canonical JSON, like specs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_CONSTRAINTS = 20
MAX_CONSTRAINT_CHARS = 200

Constraint = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=MAX_CONSTRAINT_CHARS)
]

LANGUAGE_TEXT = {"ru": "русский", "en": "английский"}
TONE_TEXT = {
    "neutral": "нейтральный",
    "friendly": "дружелюбный",
    "strict": "строгий, без лишних слов",
}
VERBOSITY_TEXT = {
    "brief": "кратко, только суть",
    "normal": "обычный объём",
    "detailed": "подробно, с пояснениями",
}
STRUCTURE_TEXT = {
    "prose": "связный текст",
    "bullets": "маркированный список",
    "steps": "нумерованные шаги",
}
CODE_TEXT = {
    "none": "без кода",
    "minimal": "код только там, где без него не обойтись, короткими фрагментами",
    "full": "полные рабочие примеры кода",
}

# The priority line is part of the block rather than logic in code: the spec's
# output contract is machine-facing, the profile is a person's wish.
PROFILE_HEADER = (
    "Профиль пользователя — как отвечать этому человеку. Учитывай его в каждом"
    " ответе, даже если вопрос о стиле не говорит. При конфликте формат ответа"
    " из настроек агента важнее профиля, а профиль важнее памяти. Прямая просьба"
    " пользователя в текущем сообщении («ответь по-английски», «впредь короче»)"
    " важнее профиля: выполни её. Язык, на котором задан вопрос, такой просьбой"
    " не считается."
)


class UserProfile(BaseModel):
    """Flat on purpose: the form is generated from the JSON Schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1, max_length=100, title="Имя", description="Как назван профиль"
    )
    about: str = Field(
        default="",
        max_length=500,
        title="О человеке",
        description="Кто это, одной-двумя фразами",
    )
    language: Literal["ru", "en", "auto"] = Field(
        default="auto", title="Язык", description="auto — язык вопроса"
    )
    tone: Literal["neutral", "friendly", "strict"] = Field(
        default="neutral", title="Тон"
    )
    verbosity: Literal["brief", "normal", "detailed"] = Field(
        default="normal", title="Объём"
    )
    structure: Literal["prose", "bullets", "steps"] = Field(
        default="prose", title="Форма ответа"
    )
    max_words: int | None = Field(
        default=None,
        ge=10,
        le=5000,
        title="Потолок слов",
        description="Мягкий предел длины ответа; пусто — без предела",
    )
    code_style: Literal["none", "minimal", "full"] = Field(
        default="minimal", title="Код", description="Нужен ли код и сколько"
    )
    constraints: list[Constraint] = Field(
        default_factory=list,
        title="Ограничения",
        description="Запреты и обязательные правила, по одному на строку",
    )

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Имя профиля не может быть пустым")
        return value

    @field_validator("about")
    @classmethod
    def clean_about(cls, value: str) -> str:
        return value.strip()

    @field_validator("constraints")
    @classmethod
    def limit_constraints(cls, value: list[str]) -> list[str]:
        # Blank lines are form noise; everything else counts toward the limit.
        # An over-limit profile is refused, never silently cut: the person
        # declared each rule and would not notice one missing.
        value = [item for item in value if item]
        if len(value) > MAX_CONSTRAINTS:
            raise ValueError(f"Не больше {MAX_CONSTRAINTS} ограничений")
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def render(self) -> str:
        """Deterministic block text: one version, byte-identical output."""
        lines = [PROFILE_HEADER, f"Имя: {self.name}"]
        if self.about:
            lines.append(f"О человеке: {self.about}")
        if self.language != "auto":
            lines.append(
                f"Язык ответа: {LANGUAGE_TEXT[self.language]}, даже если вопрос"
                " задан на другом языке"
            )
        else:
            lines.append("Язык ответа: язык вопроса")
        lines.append(f"Тон: {TONE_TEXT[self.tone]}")
        lines.append(f"Объём: {VERBOSITY_TEXT[self.verbosity]}")
        if self.max_words is not None:
            lines.append(f"Не длиннее {self.max_words} слов")
        lines.append(f"Форма: {STRUCTURE_TEXT[self.structure]}")
        lines.append(f"Код: {CODE_TEXT[self.code_style]}")
        if self.constraints:
            lines.append("Ограничения:")
            lines.extend(f"- {item}" for item in self.constraints)
        return "\n".join(lines)
