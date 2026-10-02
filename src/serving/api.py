"""Small FastAPI wrapper around a single exported ranking model."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.batcher import BatcherOverloadedError, DynamicBatcher
from src.serving.ranker import Ranker


class FeatureToken(BaseModel):
    feature_id: str = Field(min_length=1, max_length=256)
    value: float = 1.0


class Candidate(BaseModel):
    ad_id: str = Field(min_length=1, max_length=128)
    features: dict[str, list[FeatureToken]] = Field(default_factory=dict)


class RankRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_features: dict[str, list[FeatureToken]] = Field(default_factory=dict)
    context_features: dict[str, list[FeatureToken]] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(min_length=1, max_length=128)
    score_mode: Literal["ctr", "ctcvr"] = "ctcvr"
    probability_mode: Literal["raw", "platt"] = "raw"


def create_app(
    artifact_directory: str | Path,
    *,
    device: str = "cpu",
    dynamic_batching: bool = True,
    max_batch_requests: int = 8,
    max_batch_candidates: int = 256,
    max_batch_wait_ms: float = 2.0,
    max_queue_size: int = 1024,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranker = Ranker(artifact_directory, device=device)
        app.state.batcher = None
        if dynamic_batching:
            app.state.batcher = DynamicBatcher(
                app.state.ranker,
                max_requests=max_batch_requests,
                max_candidates=max_batch_candidates,
                max_wait_ms=max_batch_wait_ms,
                max_queue_size=max_queue_size,
            )
            await app.state.batcher.start()
        try:
            yield
        finally:
            if app.state.batcher is not None:
                await app.state.batcher.close()
                del app.state.batcher
            del app.state.ranker

    app = FastAPI(title="FIER offline ranking demo", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        batcher = request.app.state.batcher
        return {
            "status": "ok",
            "model_version": request.app.state.ranker.model_version,
            "dynamic_batching": batcher is not None,
            "batcher": None if batcher is None else batcher.stats(),
        }

    @app.post("/rank")
    async def rank(payload: RankRequest, request: Request) -> dict:
        rank_arguments = {
            "request_id": payload.request_id or str(uuid4()),
            "user_features": {
                field: [token.model_dump() for token in tokens]
                for field, tokens in payload.user_features.items()
            },
            "context_features": {
                field: [token.model_dump() for token in tokens]
                for field, tokens in payload.context_features.items()
            },
            "candidates": [candidate.model_dump() for candidate in payload.candidates],
            "score_mode": payload.score_mode,
            "probability_mode": payload.probability_mode,
        }
        try:
            batcher = request.app.state.batcher
            if batcher is not None:
                return await batcher.submit(rank_arguments)
            return await asyncio.to_thread(
                request.app.state.ranker.rank, **rank_arguments
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except BatcherOverloadedError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app
