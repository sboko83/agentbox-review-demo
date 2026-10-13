"""FastAPI application: wiring and the health endpoint.

The core knows nothing about HTTP; this module is the adapter that owns the
process lifecycle and hands dependencies to everything below it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agentbox import __version__
from agentbox.api.routes import presets, runs, sessions, user_profiles
from agentbox.config import Settings, load_settings
from agentbox.core.pool import ExecutorPool, ProviderGate, RunSupervisor
from agentbox.llm.provider import build_provider_registry
from agentbox.store.db import init_db, table_names

# The dev frontend runs on its own port; Vite proxies /api, but a direct call
# from the browser has to be allowed too.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    tables: list[str]
    providers: dict[str, bool]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    conn = init_db(settings.db_path)
    app.state.settings = settings
    app.state.db = conn
    # Pending tool confirmations: a streaming turn parks on one of these
    # futures while the browser decides. Keyed by session and call id.
    app.state.approvals = {}
    # An estimate is intentionally short-lived and single-use: changing a
    # matrix after seeing its price requires a new estimate and confirmation.
    app.state.run_quotes = {}
    # Without provider credentials the platform still starts: health says so
    # and model calls answer 503 instead of the process refusing to boot.
    app.state.providers = build_provider_registry()
    app.state.provider_gate = ProviderGate(settings.provider_concurrency)
    app.state.executor_pool = ExecutorPool(settings.executor_concurrency)
    app.state.supervisor = RunSupervisor(
        conn, app.state.provider_gate, app.state.executor_pool
    )
    try:
        yield
    finally:
        await app.state.supervisor.aclose()
        await app.state.providers.aclose()
        conn.close()


def create_app() -> FastAPI:
    app = FastAPI(title="agentbox", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        settings: Settings = app.state.settings
        tables = table_names(app.state.db)
        providers = app.state.providers.status
        status = "ok" if any(providers.values()) and tables else "degraded"
        return Health(
            status=status,
            version=__version__,
            database=str(settings.db_path),
            tables=tables,
            providers=providers,
        )

    app.include_router(sessions.schema_router)
    app.include_router(presets.router)
    app.include_router(user_profiles.router)
    app.include_router(sessions.router)
    app.include_router(runs.router)
    return app


app = create_app()
