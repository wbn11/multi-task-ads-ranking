"""Async dynamic micro-batching for the ranking service."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.serving.ranker import PreparedRankRequest, Ranker


class BatcherOverloadedError(RuntimeError):
    """Raised when the bounded request queue cannot accept more work."""


@dataclass(slots=True)
class _QueueItem:
    payload: dict[str, Any]
    candidate_count: int
    enqueued_at: float
    future: asyncio.Future[dict[str, Any]]


class DynamicBatcher:
    """Merge concurrent requests into one model forward within a short window."""

    def __init__(
        self,
        ranker: Ranker,
        *,
        max_requests: int = 8,
        max_candidates: int = 256,
        max_wait_ms: float = 2.0,
        max_queue_size: int = 1024,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if max_wait_ms < 0.0:
            raise ValueError("max_wait_ms cannot be negative")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.ranker = ranker
        self.max_requests = max_requests
        self.max_candidates = max_candidates
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._counters: dict[str, float | int] = {
            "submitted_requests": 0,
            "rejected_requests": 0,
            "processed_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "processed_candidates": 0,
            "batches": 0,
            "max_observed_requests_per_batch": 0,
            "max_observed_candidates_per_batch": 0,
            "max_observed_queue_depth": 0,
            "total_queue_wait_ms": 0.0,
            "total_batch_execution_ms": 0.0,
        }

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("dynamic batcher has already been started")
        self._task = asyncio.create_task(self._run(), name="rank-dynamic-batcher")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed or self._task is None:
            raise RuntimeError("dynamic batcher is not running")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        candidate_count = len(candidates)
        if candidate_count > self.max_candidates:
            raise ValueError(
                "one request has more candidates than max_batch_candidates"
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        item = _QueueItem(
            payload=payload,
            candidate_count=candidate_count,
            enqueued_at=time.perf_counter(),
            future=future,
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as error:
            self._counters["rejected_requests"] += 1
            raise BatcherOverloadedError("dynamic batching queue is full") from error
        self._counters["submitted_requests"] += 1
        self._counters["max_observed_queue_depth"] = max(
            int(self._counters["max_observed_queue_depth"]), self._queue.qsize()
        )
        return await future

    def stats(self) -> dict[str, Any]:
        counters = dict(self._counters)
        batches = int(counters["batches"])
        processed = int(counters["processed_requests"])
        counters.update(
            {
                "enabled": True,
                "max_requests": self.max_requests,
                "max_candidates": self.max_candidates,
                "max_wait_ms": self.max_wait_seconds * 1000.0,
                "max_queue_size": self.max_queue_size,
                "current_queue_depth": self._queue.qsize(),
                "average_requests_per_batch": (
                    processed / batches if batches else 0.0
                ),
                "average_candidates_per_batch": (
                    int(counters["processed_candidates"]) / batches
                    if batches
                    else 0.0
                ),
                "average_queue_wait_ms": (
                    float(counters["total_queue_wait_ms"]) / processed
                    if processed
                    else 0.0
                ),
                "average_batch_execution_ms": (
                    float(counters["total_batch_execution_ms"]) / batches
                    if batches
                    else 0.0
                ),
            }
        )
        return counters

    def _execute_batch(
        self, items: list[_QueueItem]
    ) -> list[dict[str, Any] | Exception]:
        prepared: list[PreparedRankRequest] = []
        prepared_indices: list[int] = []
        outcomes: list[dict[str, Any] | Exception | None] = [None] * len(items)
        for index, item in enumerate(items):
            try:
                prepared.append(self.ranker.prepare_request(**item.payload))
                prepared_indices.append(index)
            except Exception as error:  # Returned to only the invalid request.
                outcomes[index] = error
        if prepared:
            try:
                scored = self.ranker.rank_prepared_many(prepared)
            except Exception as error:
                for index in prepared_indices:
                    outcomes[index] = error
            else:
                for index, result in zip(prepared_indices, scored, strict=True):
                    outcomes[index] = result
        return [
            outcome
            if outcome is not None
            else RuntimeError("dynamic batching produced no outcome")
            for outcome in outcomes
        ]

    async def _run(self) -> None:
        pending: _QueueItem | None = None
        stop_after_batch = False
        while True:
            first = pending
            pending = None
            if first is None:
                queued = await self._queue.get()
                if queued is None:
                    break
                first = queued

            batch = [first]
            candidate_count = first.candidate_count
            deadline = asyncio.get_running_loop().time() + self.max_wait_seconds
            while len(batch) < self.max_requests:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0.0:
                    break
                try:
                    queued = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                except TimeoutError:
                    break
                if queued is None:
                    stop_after_batch = True
                    break
                if candidate_count + queued.candidate_count > self.max_candidates:
                    pending = queued
                    break
                batch.append(queued)
                candidate_count += queued.candidate_count

            execution_started = time.perf_counter()
            self._counters["total_queue_wait_ms"] += sum(
                (execution_started - item.enqueued_at) * 1000.0 for item in batch
            )
            outcomes = await asyncio.to_thread(self._execute_batch, batch)
            execution_ms = (time.perf_counter() - execution_started) * 1000.0
            successful = sum(isinstance(outcome, dict) for outcome in outcomes)
            self._counters["batches"] += 1
            self._counters["processed_requests"] += len(batch)
            self._counters["successful_requests"] += successful
            self._counters["failed_requests"] += len(batch) - successful
            self._counters["processed_candidates"] += candidate_count
            self._counters["total_batch_execution_ms"] += execution_ms
            self._counters["max_observed_requests_per_batch"] = max(
                int(self._counters["max_observed_requests_per_batch"]), len(batch)
            )
            self._counters["max_observed_candidates_per_batch"] = max(
                int(self._counters["max_observed_candidates_per_batch"]),
                candidate_count,
            )
            for item, outcome in zip(batch, outcomes, strict=True):
                if item.future.cancelled():
                    continue
                if isinstance(outcome, Exception):
                    item.future.set_exception(outcome)
                else:
                    item.future.set_result(outcome)
            if stop_after_batch:
                break
