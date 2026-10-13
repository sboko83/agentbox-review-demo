"""Named pointers to immutable user profile versions."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agentbox.core.user_profile import UserProfile
from agentbox.store import repository as repo

router = APIRouter(prefix="/api/user-profiles", tags=["user-profiles"])


class CreateUserProfile(UserProfile):
    is_default: bool = False

    def body(self) -> UserProfile:
        return UserProfile.model_validate(self.model_dump(exclude={"is_default"}))


class AboutPatch(BaseModel):
    about: str


def _duplicate(error: sqlite3.IntegrityError) -> HTTPException:
    if "user_profiles.name" in str(error):
        return HTTPException(status_code=409, detail="Профиль с таким именем уже существует")
    return HTTPException(status_code=409, detail=str(error))


def _missing() -> HTTPException:
    return HTTPException(status_code=404, detail="Профиль пользователя не найден")


@router.get("")
def list_user_profiles(request: Request) -> list[dict]:
    return repo.list_user_profiles(request.app.state.db)


@router.get("/schema")
def user_profile_schema() -> dict:
    return UserProfile.model_json_schema()


@router.post("", status_code=201)
def create_user_profile(body: CreateUserProfile, request: Request) -> dict:
    try:
        return repo.create_user_profile(
            request.app.state.db, body.body(), body.is_default
        )
    except sqlite3.IntegrityError as error:
        raise _duplicate(error) from error


@router.get("/{profile_id}")
def get_user_profile(profile_id: str, request: Request) -> dict:
    profile = repo.get_user_profile(request.app.state.db, profile_id)
    if profile is None:
        raise _missing()
    return profile


@router.put("/{profile_id}")
def update_user_profile(profile_id: str, body: UserProfile, request: Request) -> dict:
    try:
        profile = repo.update_user_profile(request.app.state.db, profile_id, body)
    except sqlite3.IntegrityError as error:
        raise _duplicate(error) from error
    if profile is None:
        raise _missing()
    return profile


@router.patch("/{profile_id}/about")
def patch_user_profile_about(profile_id: str, body: AboutPatch, request: Request) -> dict:
    """Quick edit of `about` only, without creating a new profile version."""
    conn = request.app.state.db
    row = conn.execute(
        "SELECT version_hash FROM user_profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    version = conn.execute(
        "SELECT body_json FROM user_profile_versions WHERE hash = ?",
        (row["version_hash"],),
    ).fetchone()
    data = json.loads(version["body_json"])
    data["about"] = body.about
    conn.execute(
        "UPDATE user_profile_versions SET body_json = ? WHERE hash = ?",
        (json.dumps(data, ensure_ascii=False), row["version_hash"]),
    )
    conn.commit()
    return repo.get_user_profile(conn, profile_id)


@router.put("/{profile_id}/default")
def set_default_user_profile(profile_id: str, request: Request) -> dict:
    profile = repo.set_default_user_profile(request.app.state.db, profile_id)
    if profile is None:
        raise _missing()
    return profile


@router.delete("/{profile_id}")
def delete_user_profile(profile_id: str, request: Request) -> dict:
    if not repo.delete_user_profile(request.app.state.db, profile_id):
        raise _missing()
    return {"deleted": profile_id}
