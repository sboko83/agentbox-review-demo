"""Named pointers to immutable AgentSpec versions."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from agentbox.core.spec import AgentSpec
from agentbox.store import repository as repo

router = APIRouter(prefix="/api/presets", tags=["presets"])


class PresetBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    spec: AgentSpec
    is_default: bool = False

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Имя пресета не может быть пустым")
        return value


class UpdatePreset(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    spec: AgentSpec

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Имя пресета не может быть пустым")
        return value


def _duplicate(error: sqlite3.IntegrityError) -> HTTPException:
    if "agent_presets.name" in str(error):
        return HTTPException(status_code=409, detail="Пресет с таким именем уже существует")
    return HTTPException(status_code=409, detail=str(error))


@router.get("")
def list_presets(request: Request) -> list[dict]:
    return repo.list_presets(request.app.state.db)


@router.post("", status_code=201)
def create_preset(body: PresetBody, request: Request) -> dict:
    try:
        return repo.create_preset(
            request.app.state.db, body.name, body.spec, body.is_default
        )
    except sqlite3.IntegrityError as error:
        raise _duplicate(error) from error


@router.get("/{preset_id}")
def get_preset(preset_id: str, request: Request) -> dict:
    preset = repo.get_preset(request.app.state.db, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Пресет не найден")
    return preset


@router.put("/{preset_id}")
def update_preset(preset_id: str, body: UpdatePreset, request: Request) -> dict:
    try:
        preset = repo.update_preset(
            request.app.state.db, preset_id, body.name, body.spec
        )
    except sqlite3.IntegrityError as error:
        raise _duplicate(error) from error
    if preset is None:
        raise HTTPException(status_code=404, detail="Пресет не найден")
    return preset


@router.put("/{preset_id}/default")
def set_default_preset(preset_id: str, request: Request) -> dict:
    preset = repo.set_default_preset(request.app.state.db, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Пресет не найден")
    return preset


@router.delete("/{preset_id}")
def delete_preset(preset_id: str, request: Request) -> dict:
    if not repo.delete_preset(request.app.state.db, preset_id):
        raise HTTPException(status_code=404, detail="Пресет не найден")
    return {"deleted": preset_id}
