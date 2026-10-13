"""Runtime settings: paths, server binding and shared resource limits.

Provider-specific credentials and URLs are owned by `agentbox.llm.provider`.
Nothing here reaches the network; loading is deliberately dependency-free so
the settings can be read by scripts and tests alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # agentbox/
REPO_ROOT = PROJECT_ROOT.parent
SECRETS_FILE = REPO_ROOT / "secrets" / ".env"


def read_env_file(path: Path = SECRETS_FILE) -> dict[str, str]:
    """Parse a `KEY=value` file. Missing file yields an empty mapping."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


@dataclass(frozen=True)
class Settings:
    db_path: Path
    host: str
    port: int
    provider_concurrency: int
    executor_concurrency: int
    docker_image: str
    docker_cpus: float
    docker_memory_mb: int
    docker_pids_limit: int


def load_settings() -> Settings:
    """Environment wins over `secrets/.env` so a run can be overridden ad hoc."""
    db_path = Path(
        os.environ.get("AGENTBOX_DB", PROJECT_ROOT / "data" / "agentbox.db")
    )
    return Settings(
        db_path=db_path,
        host=os.environ.get("AGENTBOX_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENTBOX_PORT", "8080")),
        provider_concurrency=int(os.environ.get("AGENTBOX_PROVIDER_CONCURRENCY", "4")),
        executor_concurrency=int(os.environ.get("AGENTBOX_EXECUTOR_CONCURRENCY", "2")),
        docker_image=os.environ.get("AGENTBOX_DOCKER_IMAGE", "agentbox-tools:latest"),
        docker_cpus=float(os.environ.get("AGENTBOX_DOCKER_CPUS", "1.0")),
        docker_memory_mb=int(os.environ.get("AGENTBOX_DOCKER_MEMORY_MB", "512")),
        docker_pids_limit=int(os.environ.get("AGENTBOX_DOCKER_PIDS_LIMIT", "128")),
    )
