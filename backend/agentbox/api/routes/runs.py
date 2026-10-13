"""Confirmed matrix experiments and their comparison reports."""

from __future__ import annotations

import csv
import io
import json
import secrets
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator

from agentbox.core.experiments import (
    EstimateError,
    Price,
    comparison_report,
    estimate_matrix,
)
from agentbox.core.ports import LLMError
from agentbox.core.invariants import InvariantSet
from agentbox.core.spec import AgentSpec, normalize_budget_usd
from agentbox.core.user_profile import UserProfile
from agentbox.store import repository as repo

router = APIRouter(prefix="/api/runs", tags=["runs"])

# An unconfirmed estimate is cheap but not free to keep. Holding the last N is
# enough for a person comparing a few matrices, and bounds the process.
MAX_PENDING_QUOTES = 50


class MatrixRequest(BaseModel):
    """The immutable payload that gets priced, then confirmed unchanged."""

    name: str | None = Field(default=None, max_length=100)
    prompt: str = Field(min_length=1, max_length=100_000)
    repeats: int = Field(default=3, ge=1, le=20)
    budget_usd: float | None = Field(default=None, gt=0)
    specs: list[AgentSpec] = Field(default_factory=list, max_length=30)
    preset_ids: list[str] = Field(default_factory=list, max_length=30)
    # Empty keeps the old matrix: every configuration runs without a profile.
    user_profile_ids: list[str] = Field(default_factory=list, max_length=10)
    # One complete rules snapshot applies to every row of this matrix.
    invariants: InvariantSet = Field(default_factory=InvariantSet)

    @field_validator("budget_usd", mode="before")
    @classmethod
    def zero_budget_means_unlimited(cls, value: object) -> object:
        return normalize_budget_usd(value)

    @model_validator(mode="after")
    def one_matrix_source(self) -> "MatrixRequest":
        if bool(self.specs) == bool(self.preset_ids):
            raise ValueError("Передайте либо specs, либо preset_ids")
        if len(set(self.preset_ids)) != len(self.preset_ids):
            raise ValueError("Пресет нельзя добавить в матрицу дважды")
        if len(set(self.user_profile_ids)) != len(self.user_profile_ids):
            raise ValueError("Профиль нельзя добавить в матрицу дважды")
        base = len(self.specs) or len(self.preset_ids)
        if base * max(1, len(self.user_profile_ids)) > 30:
            raise ValueError("В матрице не больше 30 конфигураций")
        return self


class ConfirmRun(BaseModel):
    quote_id: str = Field(min_length=1)


async def _prices(request: Request) -> dict[tuple[str, str], Price]:
    try:
        catalog = await request.app.state.providers.list_models()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        (item["provider"], item["id"]): Price(
            item["input_usd_per_million"], item["output_usd_per_million"]
        )
        for item in catalog
        if item["input_usd_per_million"] is not None
        and item["output_usd_per_million"] is not None
    }


@router.post("/estimate")
async def estimate(body: MatrixRequest, request: Request) -> dict:
    """Return a bound from the live supported-model catalog; never starts calls."""
    specs = body.specs
    config_names = [f"Конфигурация {index + 1}" for index in range(len(specs))]
    if body.preset_ids:
        presets = [repo.get_preset(request.app.state.db, item) for item in body.preset_ids]
        missing = [item for item, preset in zip(body.preset_ids, presets) if preset is None]
        if missing:
            raise HTTPException(status_code=404, detail=f"Пресет не найден: {missing[0]}")
        specs = [AgentSpec(**preset["spec"]) for preset in presets if preset is not None]
        config_names = [preset["name"] for preset in presets if preset is not None]
    profiles = _profile_versions(body.user_profile_ids, request)
    if profiles:
        # Cartesian product: every (preset, profile) pair is one configuration.
        pairs = [
            (spec, f"{name} · {profile.name}", profile)
            for spec, name in zip(specs, config_names)
            for profile in profiles
        ]
        specs = [spec for spec, _, _ in pairs]
        config_names = [name for _, name, _ in pairs]
        profiles = [profile for _, _, profile in pairs]
    try:
        quote = estimate_matrix(
            body.prompt,
            body.repeats,
            specs,
            await _prices(request),
            config_names,
            user_profiles=profiles or None,
            invariant_sets=[body.invariants for _ in specs],
        )
    except EstimateError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    quote_id = secrets.token_urlsafe(24)
    quotes: dict[str, dict] = request.app.state.run_quotes
    quotes[quote_id] = {
        "request": body,
        "quote": quote,
        "specs": specs,
        "config_names": config_names,
        # The version is fixed at quote time: what was priced is what runs.
        "user_profiles": profiles or None,
        "invariants": body.invariants,
    }
    while len(quotes) > MAX_PENDING_QUOTES:
        quotes.pop(next(iter(quotes)))
    return {"quote_id": quote_id, "quote": quote.as_dict()}


@router.post("")
async def start(body: ConfirmRun, request: Request) -> dict:
    """Consume one displayed quote before making any paid provider call."""
    quoted = request.app.state.run_quotes.pop(body.quote_id, None)
    if quoted is None:
        raise HTTPException(
            status_code=409,
            detail="Оценка устарела или уже подтверждена; рассчитайте стоимость заново.",
        )
    matrix: MatrixRequest = quoted["request"]
    quote = quoted["quote"]
    specs: list[AgentSpec] = quoted["specs"]
    bindings = {}
    try:
        for spec in specs:
            provider = request.app.state.providers.require(spec.model.provider)
            assert provider.client is not None
            bindings[provider.name] = (
                provider.client,
                provider.capabilities.transport_for,
            )
    except LLMError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    run_id = repo.create_run(
        request.app.state.db,
        name=matrix.name,
        prompt=matrix.prompt,
        repeats=matrix.repeats,
        budget_usd=matrix.budget_usd,
        estimated_cost_usd=quote.total_usd,
        specs=specs,
        config_names=quoted["config_names"],
        user_profile_hashes=(
            [profile.hash for profile in quoted["user_profiles"]]
            if quoted["user_profiles"]
            else None
        ),
        invariant_sets=[quoted["invariants"] for _ in specs],
    )
    request.app.state.supervisor.start(
        run_id=run_id,
        provider_for=lambda spec: bindings[spec.model.provider],
        prompt=matrix.prompt,
        specs=specs,
        repeats=matrix.repeats,
        budget_usd=matrix.budget_usd,
        config_names=quoted["config_names"],
        user_profiles=quoted["user_profiles"],
        invariant_sets=[quoted["invariants"] for _ in specs],
    )
    return await request.app.state.supervisor.dashboard(run_id)


def _profile_versions(profile_ids: list[str], request: Request) -> list[UserProfile]:
    conn = request.app.state.db
    versions = []
    for profile_id in profile_ids:
        pointer = repo.get_user_profile(conn, profile_id)
        version = (
            repo.get_user_profile_version(conn, pointer["version_hash"]) if pointer else None
        )
        if version is None:
            raise HTTPException(status_code=404, detail=f"Профиль не найден: {profile_id}")
        versions.append(version)
    return versions


@router.get("/{run_id}")
def get(run_id: str, request: Request) -> dict:
    report = _report(run_id, request)
    return {**report, "dashboard": request.app.state.supervisor.registry.snapshot(run_id)}


@router.get("/{run_id}/export")
def export(run_id: str, request: Request, format: Literal["json", "csv"] = "json") -> Response:
    """Download the immutable run data, not a screenshot of the UI table."""
    report = _report(run_id, request)
    filename = f"agentbox-run-{run_id}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if format == "json":
        return Response(
            content=json.dumps(report, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers=headers,
        )
    return Response(
        content=_csv(report),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


def _report(run_id: str, request: Request) -> dict:
    report = repo.get_run(request.app.state.db, run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return comparison_report(report)


def _csv(report: dict) -> str:
    """One cell per row: convenient for a spreadsheet without losing verdicts."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "run_id", "status", "config_index", "repeat_index", "cell_status",
            "answer", "score", "passed", "judge_error", "resolution", "actual_cost_usd",
        ],
    )
    writer.writeheader()
    for result in report["results"]:
        judge = result.get("judge") or {}
        writer.writerow(
            {
                "run_id": report["id"],
                "status": report["status"],
                "config_index": result["config_index"],
                "repeat_index": result["repeat_index"],
                "cell_status": result["status"],
                "answer": result.get("answer", ""),
                "score": judge.get("score", ""),
                "passed": judge.get("passed", ""),
                "judge_error": judge.get("error", ""),
                "resolution": result.get("resolution", ""),
                "actual_cost_usd": result.get("actual_cost_usd", 0),
            }
        )
    return output.getvalue()


@router.get("/{run_id}/dashboard")
async def dashboard(run_id: str, request: Request) -> dict:
    if repo.get_run(request.app.state.db, run_id) is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return await request.app.state.supervisor.dashboard(run_id)
