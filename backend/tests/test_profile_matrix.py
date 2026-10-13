"""Matrix runs over (preset, profile) pairs and the profile version they cite."""

import json
import time

from agentbox.core.user_profile import UserProfile
from agentbox.llm.stub import StubLLM

from test_sessions_api import client  # noqa: F401 - shared fixture

VERDICT = json.dumps({"score": 9, "rationale": "ок"})


class RoleStub(StubLLM):
    """Answers the judge with a verdict and the agent with text, in any order."""

    async def stream(self, request):
        self.script = [(VERDICT if request.text_format else "ответ", [])]
        async for chunk in super().stream(request):
            yield chunk


def _system_text(request) -> str:
    return "\n".join(
        str(item.get("content")) for item in request.messages if item["role"] == "system"
    )


def _preset(client, name: str) -> dict:
    return client.post(
        "/api/presets", json={"name": name, "spec": {"model": {"role": name}}}
    ).json()


def _profile(client, name: str, **fields) -> dict:
    return client.post("/api/user-profiles", json={"name": name, **fields}).json()


def _finished(client, run_id: str) -> dict:
    for _ in range(100):
        report = client.get(f"/api/runs/{run_id}").json()
        if report["status"] != "running":
            return report
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def _run(client, body: dict) -> dict:
    quote = client.post("/api/runs/estimate", json=body)
    assert quote.status_code == 200, quote.text
    started = client.post("/api/runs", json={"quote_id": quote.json()["quote_id"]})
    assert started.status_code == 200, started.text
    return _finished(client, started.json()["run_id"])


def test_matrix_is_the_cartesian_product_of_presets_and_profiles(client):
    stub = RoleStub()
    client.app.state.providers["zen"].client = stub
    presets = [_preset(client, "A"), _preset(client, "B")]
    profiles = [_profile(client, "dev", language="en"), _profile(client, "pm", tone="strict")]

    quote = client.post(
        "/api/runs/estimate",
        json={
            "prompt": "вопрос",
            "repeats": 1,
            "preset_ids": [item["id"] for item in presets],
            "user_profile_ids": [item["id"] for item in profiles],
        },
    ).json()
    assert [item["name"] for item in quote["quote"]["by_config"]] == [
        "A · dev", "A · pm", "B · dev", "B · pm",
    ]
    report = _run(
        client,
        {
            "prompt": "вопрос",
            "repeats": 1,
            "preset_ids": [item["id"] for item in presets],
            "user_profile_ids": [item["id"] for item in profiles],
        },
    )

    assert report["status"] == "completed"
    assert [
        (group["name"], group["user_profile"]["hash"], group["passed"])
        for group in report["comparison"]
    ] == [
        ("A · dev", profiles[0]["version_hash"], 1),
        ("A · pm", profiles[1]["version_hash"], 1),
        ("B · dev", profiles[0]["version_hash"], 1),
        ("B · pm", profiles[1]["version_hash"], 1),
    ]
    rendered = {UserProfile(**item["profile"]).render() for item in profiles}
    answers = [request for request in stub.requests if not request.text_format]
    judges = [request for request in stub.requests if request.text_format]
    assert len(answers) == len(judges) == 4
    # Each answer carries exactly one of the two profiles; the judge never does.
    assert sorted(
        next(text for text in rendered if text in _system_text(request))
        for request in answers
    ) == sorted([*rendered, *rendered])
    assert not any(text in _system_text(request) for request in judges for text in rendered)


def test_run_keeps_the_profile_version_after_the_profile_is_edited(client):
    client.app.state.providers["zen"].client = RoleStub()
    preset = _preset(client, "A")
    profile = _profile(client, "dev", language="en")
    before = _run(
        client,
        {
            "prompt": "вопрос",
            "repeats": 1,
            "preset_ids": [preset["id"]],
            "user_profile_ids": [profile["id"]],
        },
    )

    edited = client.put(
        f"/api/user-profiles/{profile['id']}", json={"name": "dev", "language": "ru"}
    ).json()
    assert edited["version_hash"] != profile["version_hash"]
    after = client.get(f"/api/runs/{before['id']}").json()

    assert after["configs"] == before["configs"]
    assert after["configs"][0]["user_profile"] == {
        "hash": profile["version_hash"],
        "profile": profile["profile"],
    }
    assert after["results"] == before["results"]
    hashes = client.app.state.db.execute(
        "SELECT model, user_profile_hash FROM ledger WHERE run_id = ? ORDER BY id",
        (before["id"],),
    ).fetchall()
    # The answer row cites the version it sent; the judge row cites none.
    assert [row["user_profile_hash"] for row in hashes] == [profile["version_hash"], None]


def test_matrix_rejects_unknown_or_repeated_profiles(client):
    preset = _preset(client, "A")
    profile = _profile(client, "dev")
    body = {"prompt": "вопрос", "preset_ids": [preset["id"]]}
    assert client.post(
        "/api/runs/estimate", json={**body, "user_profile_ids": ["missing"]}
    ).status_code == 404
    assert client.post(
        "/api/runs/estimate",
        json={**body, "user_profile_ids": [profile["id"], profile["id"]]},
    ).status_code == 422
