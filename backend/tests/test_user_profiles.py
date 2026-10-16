import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agentbox.api.app import create_app
from agentbox.core.user_profile import MAX_CONSTRAINTS, PROFILE_HEADER, UserProfile
from agentbox.store import repository as repo


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBOX_DB", str(tmp_path / "test.db"))
    with TestClient(create_app()) as test_client:
        yield test_client


def profile(**fields) -> dict:
    return {"name": "Сергей", **fields}


def test_hash_ignores_field_order_and_formatting():
    first = UserProfile(name="Сергей", language="ru", constraints=["без платных сервисов"])
    second = UserProfile.model_validate(
        {"constraints": ["  без платных сервисов ", ""], "language": "ru", "name": " Сергей "}
    )

    assert first.hash == second.hash
    assert first.render() == second.render()
    assert first.hash != UserProfile(name="Сергей", language="en").hash


def test_render_is_deterministic_and_carries_the_priority_line():
    body = UserProfile(
        name="Сергей",
        about="iOS-разработчик",
        language="en",
        verbosity="brief",
        structure="steps",
        max_words=120,
        code_style="none",
        constraints=["без платных сервисов"],
    ).render()

    assert body.startswith(PROFILE_HEADER)
    assert "формат ответа из настроек агента важнее профиля" in body
    assert "Прямая просьба пользователя в текущем сообщении" in body
    assert "важнее профиля: выполни её" in body
    assert "Язык, на котором задан вопрос, такой просьбой не считается" in body
    for line in (
        "О человеке: iOS-разработчик",
        "Язык ответа: английский, даже если вопрос задан на другом языке",
        "Не длиннее 120 слов",
        "Форма: нумерованные шаги",
        "Код: без кода",
        "- без платных сервисов",
    ):
        assert line in body


@pytest.mark.parametrize(
    "fields",
    [
        {"about": "x" * 501},
        {"constraints": ["x" * 201]},
        {"constraints": [f"правило {index}" for index in range(MAX_CONSTRAINTS + 1)]},
        {"language": "de"},
        {"max_words": 5},
        {"name": "   "},
        {"favourite_color": "blue"},
    ],
)
def test_over_limit_profile_is_refused_not_cut(client, fields):
    with pytest.raises(ValidationError):
        UserProfile(**profile(**fields))
    response = client.post("/api/user-profiles", json=profile(**fields))

    assert response.status_code == 422
    assert client.get("/api/user-profiles").json() == []


def test_same_content_is_one_version(client):
    first = client.post("/api/user-profiles", json=profile(tone="strict")).json()
    second = client.post(
        "/api/user-profiles", json={"tone": "strict", "name": "Сергей-2"}
    ).json()
    renamed_back = client.put(
        f"/api/user-profiles/{second['id']}", json=profile(tone="strict")
    )
    conn = client.app.state.db

    # A duplicate name is refused, and the refusal writes nothing.
    assert renamed_back.status_code == 409
    unchanged = client.put(f"/api/user-profiles/{first['id']}", json=profile(tone="strict"))
    assert unchanged.json()["version_hash"] == first["version_hash"]
    assert conn.execute("SELECT COUNT(*) FROM user_profile_versions").fetchone()[0] == 2

    edited = client.put(
        f"/api/user-profiles/{first['id']}", json=profile(tone="friendly")
    ).json()
    assert edited["version_hash"] != first["version_hash"]
    assert edited["profile"]["tone"] == "friendly"
    # The old version stays: ledger rows written under it still resolve.
    assert repo.get_user_profile_version(conn, first["version_hash"]).tone == "strict"


def test_schema_endpoint_serves_the_form(client):
    schema = client.get("/api/user-profiles/schema").json()

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name"]
    assert set(schema["properties"]) == {
        "name", "about", "language", "tone", "verbosity",
        "structure", "max_words", "code_style", "constraints",
    }


def test_default_profile_is_assigned_to_new_sessions(client):
    assert client.post("/api/sessions", json={}).json()["user_profile_id"] is None

    first = client.post("/api/user-profiles", json=profile()).json()
    second = client.post(
        "/api/user-profiles", json={"name": "Гость", "is_default": True}
    ).json()
    assert client.post("/api/sessions", json={}).json()["user_profile_id"] == second["id"]

    made_default = client.put(f"/api/user-profiles/{first['id']}/default").json()
    assert made_default["is_default"] is True
    listed = client.get("/api/user-profiles").json()
    assert [item["id"] for item in listed if item["is_default"]] == [first["id"]]
    assert client.post("/api/sessions", json={}).json()["user_profile_id"] == first["id"]

    # An explicit choice wins, and an explicit null opts out of the default.
    chosen = client.post("/api/sessions", json={"user_profile_id": second["id"]}).json()
    assert chosen["user_profile_id"] == second["id"]
    bare = client.post("/api/sessions", json={"user_profile_id": None}).json()
    assert bare["user_profile_id"] is None
    missing = client.post("/api/sessions", json={"user_profile_id": "nope"})
    assert missing.status_code == 404


def test_session_follows_the_pointer_and_can_switch(client):
    first = client.post("/api/user-profiles", json=profile(language="ru")).json()
    second = client.post("/api/user-profiles", json={"name": "Guest"}).json()
    session_id = client.post(
        "/api/sessions", json={"user_profile_id": first["id"]}
    ).json()["id"]

    client.put(f"/api/user-profiles/{first['id']}", json=profile(language="en"))
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["user_profile_id"] == first["id"]
    assert detail["user_profile"]["profile"]["language"] == "en"
    listed = {item["id"]: item for item in client.get("/api/sessions").json()}
    assert listed[session_id]["user_profile_name"] == "Сергей"

    switched = client.patch(
        f"/api/sessions/{session_id}", json={"user_profile_id": second["id"]}
    ).json()
    assert switched["user_profile"]["name"] == "Guest"
    assert client.patch(
        f"/api/sessions/{session_id}", json={"user_profile_id": "nope"}
    ).status_code == 404
    cleared = client.patch(f"/api/sessions/{session_id}", json={"user_profile_id": None})
    assert cleared.json()["user_profile"] is None

    client.patch(f"/api/sessions/{session_id}", json={"user_profile_id": second["id"]})
    assert client.delete(f"/api/user-profiles/{second['id']}").status_code == 200
    assert client.get(f"/api/user-profiles/{second['id']}").status_code == 404
    assert client.get(f"/api/sessions/{session_id}").json()["user_profile_id"] is None


def test_branch_inherits_the_pointer(client):
    person = client.post("/api/user-profiles", json=profile()).json()
    session_id = client.post(
        "/api/sessions", json={"user_profile_id": person["id"]}
    ).json()["id"]
    conn = client.app.state.db
    repo.append_message(conn, session_id, "user", "вопрос")
    repo.append_message(conn, session_id, "assistant", "ответ")

    branch = client.post(f"/api/sessions/{session_id}/fork", json={"seq": 2}).json()
    client.put(f"/api/user-profiles/{person['id']}", json=profile(verbosity="brief"))

    detail = client.get(f"/api/sessions/{branch['id']}").json()
    assert detail["user_profile_id"] == person["id"]
    assert detail["user_profile"]["profile"]["verbosity"] == "brief"


def test_run_session_profile_is_immutable(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    conn = client.app.state.db
    conn.execute(
        "INSERT INTO runs (id, prompt, status, created_at) VALUES ('r1', 'x', 'done', 'now')"
    )
    conn.execute("UPDATE sessions SET run_id = 'r1' WHERE id = ?", (session_id,))
    conn.commit()

    response = client.patch(f"/api/sessions/{session_id}", json={"user_profile_id": None})

    assert response.status_code == 409


def test_patch_about_saves_a_new_version(client):
    created = client.post("/api/user-profiles", json=profile(about="старое")).json()
    edited = client.patch(
        f"/api/user-profiles/{created['id']}/about", json={"about": "новое"}
    )
    assert edited.status_code == 200
    assert edited.json()["profile"]["about"] == "новое"
    assert edited.json()["version_hash"] != created["version_hash"]
    # The old version is untouched: ledger rows written under it still resolve.
    conn = client.app.state.db
    assert repo.get_user_profile_version(conn, created["version_hash"]).about == "старое"
    assert edited.json() == client.get(f"/api/user-profiles/{created['id']}").json()
    missing = client.patch("/api/user-profiles/nope/about", json={"about": "x"})
    assert missing.status_code == 404
    too_long = client.patch(
        f"/api/user-profiles/{created['id']}/about", json={"about": "x" * 501}
    )
    assert too_long.status_code == 422


def test_patch_about_does_not_overwrite_a_concurrent_edit(client, monkeypatch):
    created = client.post("/api/user-profiles", json=profile(about="старое")).json()
    renamed = client.put(
        f"/api/user-profiles/{created['id']}", json=profile(name="Другой", about="старое")
    ).json()
    # The PATCH read its snapshot before the PUT above moved the pointer.
    monkeypatch.setattr(repo, "get_user_profile", lambda conn, profile_id: created)
    response = client.patch(
        f"/api/user-profiles/{created['id']}/about", json={"about": "новое"}
    )
    assert response.status_code == 409
    monkeypatch.undo()
    conn = client.app.state.db
    current = repo.get_user_profile(conn, created["id"])
    assert current["version_hash"] == renamed["version_hash"]
    # The version written for the refused edit was rolled back with the pointer.
    versions = conn.execute("SELECT COUNT(*) FROM user_profile_versions").fetchone()[0]
    assert versions == 2


def test_patch_about_of_a_profile_deleted_meanwhile_is_404(client, monkeypatch):
    created = client.post("/api/user-profiles", json=profile(about="старое")).json()
    client.delete(f"/api/user-profiles/{created['id']}")
    # The PATCH read its snapshot before the DELETE above.
    monkeypatch.setattr(repo, "get_user_profile", lambda conn, profile_id: created)
    response = client.patch(
        f"/api/user-profiles/{created['id']}/about", json={"about": "новое"}
    )
    assert response.status_code == 404
