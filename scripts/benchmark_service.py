"""Measure client-observed /rank throughput and latency on a running service."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_BATCH_COUNTERS = (
    "submitted_requests",
    "rejected_requests",
    "processed_requests",
    "successful_requests",
    "failed_requests",
    "processed_candidates",
    "batches",
    "total_queue_wait_ms",
    "total_batch_execution_ms",
)


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("percentile needs non-empty values and fraction in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def make_payload(template: dict[str, Any], batch_size: int) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    originals = template.get("candidates")
    if not isinstance(originals, list) or not originals:
        raise ValueError("payload template must contain candidates")
    payload = dict(template)
    payload["candidates"] = []
    for index in range(batch_size):
        candidate = dict(originals[index % len(originals)])
        candidate["ad_id"] = f"{candidate['ad_id']}_bench_{index}"
        payload["candidates"].append(candidate)
    return payload


def request_feature_token_count(payload: dict[str, Any]) -> int:
    """Match Ranker.prepare_request's per-request feature-token accounting."""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("payload must contain a non-empty candidates list")

    def field_token_count(fields: Any, *, name: str) -> int:
        if not isinstance(fields, dict):
            raise ValueError(f"{name} must be a field-to-token-list mapping")
        total = 0
        for field_id, tokens in fields.items():
            if not isinstance(tokens, list):
                raise ValueError(f"{name}[{field_id!r}] must be a token list")
            total += len(tokens)
        return total

    shared = field_token_count(
        payload.get("user_features", {}), name="user_features"
    ) + field_token_count(
        payload.get("context_features", {}), name="context_features"
    )
    candidate_total = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} must be an object")
        candidate_total += field_token_count(
            candidate.get("features", {}), name=f"candidate {index} features"
        )
    return shared * len(candidates) + candidate_total


def filter_templates_by_token_limit(
    templates: list[dict[str, Any]],
    *,
    batch_sizes: list[int],
    max_request_feature_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one fixed template set that is valid for every benchmark case."""

    if not batch_sizes or max_request_feature_tokens <= 0:
        raise ValueError("batch sizes and token limit must be positive")
    validation_batch_size = max(batch_sizes)
    retained: list[dict[str, Any]] = []
    retained_indices: list[int] = []
    dropped: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        expanded = make_payload(template, validation_batch_size)
        estimated_tokens = request_feature_token_count(expanded)
        if estimated_tokens <= max_request_feature_tokens:
            retained.append(template)
            retained_indices.append(index)
        else:
            dropped.append(
                {
                    "index": index,
                    "request_id": template.get("request_id"),
                    "estimated_tokens": estimated_tokens,
                }
            )
    return retained, {
        "max_request_feature_tokens": max_request_feature_tokens,
        "validation_batch_size": validation_batch_size,
        "original_templates": len(templates),
        "retained_templates": len(retained),
        "retained_indices": retained_indices,
        "dropped_templates": dropped,
    }


def _post(url: str, body: bytes, *, timeout: float, expected_count: int) -> float:
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {error.code} from /rank: {detail}") from error
    latency_ms = (time.perf_counter() - started) * 1000.0
    if len(result.get("results", [])) != expected_count:
        raise RuntimeError("response contains the wrong number of ranked ads")
    return latency_ms


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            payload = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {error.code} from health endpoint: {detail}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("health endpoint returned a non-object response")
    return payload


def batcher_counter_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Summarize server-side work performed during one measured case."""

    before_batcher = before.get("batcher")
    after_batcher = after.get("batcher")
    if not isinstance(before_batcher, dict) or not isinstance(after_batcher, dict):
        return {"enabled": False}
    delta: dict[str, Any] = {"enabled": True}
    for key in _BATCH_COUNTERS:
        delta[key] = float(after_batcher.get(key, 0.0)) - float(
            before_batcher.get(key, 0.0)
        )
        if key not in {"total_queue_wait_ms", "total_batch_execution_ms"}:
            delta[key] = int(delta[key])
    batches = int(delta["batches"])
    processed = int(delta["processed_requests"])
    delta["average_requests_per_batch"] = (
        processed / batches if batches else 0.0
    )
    delta["average_candidates_per_batch"] = (
        int(delta["processed_candidates"]) / batches if batches else 0.0
    )
    delta["average_queue_wait_ms"] = (
        float(delta["total_queue_wait_ms"]) / processed if processed else 0.0
    )
    delta["average_batch_execution_ms"] = (
        float(delta["total_batch_execution_ms"]) / batches if batches else 0.0
    )
    delta["configuration"] = {
        key: after_batcher.get(key)
        for key in (
            "max_requests",
            "max_candidates",
            "max_wait_ms",
            "max_queue_size",
        )
    }
    return delta


def benchmark_case(
    *,
    url: str,
    payloads: list[dict[str, Any]],
    requests: int,
    warmup: int,
    concurrency: int,
    timeout: float,
    health_url: str | None = None,
) -> dict[str, Any]:
    if requests <= 0 or warmup < 0 or concurrency <= 0 or timeout <= 0:
        raise ValueError("invalid benchmark count, concurrency or timeout")
    if not payloads:
        raise ValueError("at least one payload is required")
    batch_size = len(payloads[0]["candidates"])
    if any(len(payload["candidates"]) != batch_size for payload in payloads):
        raise ValueError("all payloads in one case need the same candidate count")
    bodies = [
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for payload in payloads
    ]
    for index in range(warmup):
        template_index = index % len(bodies)
        try:
            _post(
                url,
                bodies[template_index],
                timeout=timeout,
                expected_count=batch_size,
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "warmup request failed: "
                f"batch_size={batch_size} concurrency={concurrency} "
                f"template_index={template_index}: {error}"
            ) from error

    service_before = (
        _get_json(health_url, timeout=timeout) if health_url is not None else None
    )
    latencies: list[float] = []
    failures: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _post,
                url,
                bodies[index % len(bodies)],
                timeout=timeout,
                expected_count=batch_size,
            )
            for index in range(requests)
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except (HTTPError, URLError, OSError, ValueError, RuntimeError) as error:
                if len(failures) < 5:
                    failures.append(str(error))
    elapsed = time.perf_counter() - started
    service_after = (
        _get_json(health_url, timeout=timeout) if health_url is not None else None
    )
    successful = len(latencies)
    result = {
        "valid": successful == requests,
        "payload_templates": len(payloads),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "measured_requests": requests,
        "successful_requests": successful,
        "failed_requests": requests - successful,
        "failure_examples": failures,
        "wall_seconds": elapsed,
        "qps": successful / elapsed,
        "candidate_scores_per_second": successful * batch_size / elapsed,
        "p50_latency_ms": (
            percentile_nearest_rank(latencies, 0.50) if latencies else None
        ),
        "p95_latency_ms": (
            percentile_nearest_rank(latencies, 0.95) if latencies else None
        ),
        "p99_latency_ms": (
            percentile_nearest_rank(latencies, 0.99) if latencies else None
        ),
    }
    if service_before is not None and service_after is not None:
        result["service_batching"] = batcher_counter_delta(
            service_before, service_after
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/rank")
    parser.add_argument(
        "--health-url",
        help="optional service health endpoint used to capture batching counters",
    )
    parser.add_argument("--payload-file", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-request-feature-tokens",
        type=int,
        default=16_384,
        help=(
            "pre-filter one fixed template set that stays within the service's "
            "per-request feature-token limit for the largest requested batch"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 1 <= size <= 128 for size in args.batch_sizes):
        parser.error("all batch sizes must be in [1, 128]")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_request_feature_tokens <= 0:
        parser.error("--max-request-feature-tokens must be positive")
    templates = json.loads(args.payload_file.read_text(encoding="utf-8"))
    if isinstance(templates, dict):
        templates = [templates]
    if not isinstance(templates, list) or not templates or not all(
        isinstance(template, dict) for template in templates
    ):
        parser.error("payload file must contain a JSON object or non-empty array")
    try:
        templates, template_filter = filter_templates_by_token_limit(
            templates,
            batch_sizes=args.batch_sizes,
            max_request_feature_tokens=args.max_request_feature_tokens,
        )
    except ValueError as error:
        parser.error(str(error))
    if not templates:
        parser.error(
            "every payload template exceeds the request feature-token limit at "
            f"batch_size={max(args.batch_sizes)}"
        )
    dropped = template_filter["dropped_templates"]
    print(
        "payload preflight: "
        f"kept={len(templates)}/{template_filter['original_templates']} "
        f"batch_size={template_filter['validation_batch_size']} "
        f"limit={args.max_request_feature_tokens}",
        flush=True,
    )
    if dropped:
        print(
            "dropped oversized templates: "
            + ", ".join(
                f"index={item['index']} tokens={item['estimated_tokens']}"
                for item in dropped
            ),
            flush=True,
        )
    cases = []
    summaries = []
    for batch_size in args.batch_sizes:
        payloads = [make_payload(template, batch_size) for template in templates]
        for concurrency in args.concurrencies:
            rounds = []
            for repeat in range(1, args.repeats + 1):
                result = benchmark_case(
                    url=args.url,
                    payloads=payloads,
                    requests=args.requests,
                    warmup=args.warmup,
                    concurrency=concurrency,
                    timeout=args.timeout,
                    health_url=args.health_url,
                )
                result["repeat"] = repeat
                cases.append(result)
                rounds.append(result)
                batching = result.get("service_batching") or {}
                batch_message = ""
                if batching.get("enabled"):
                    batch_message = (
                        " server_batches="
                        f"{batching['batches']}"
                        " avg_requests_per_batch="
                        f"{batching['average_requests_per_batch']:.2f}"
                    )
                print(
                    f"batch={batch_size} concurrency={concurrency} "
                    f"repeat={repeat}/{args.repeats} "
                    f"QPS={result['qps']:.2f} "
                    f"P95={result['p95_latency_ms']} ms "
                    f"success={result['successful_requests']}/{args.requests}"
                    f"{batch_message}",
                    flush=True,
                )
            all_valid = all(result["valid"] for result in rounds)
            summary = {
                "valid": all_valid,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "repeats": args.repeats,
                "median_qps": (
                    statistics.median(result["qps"] for result in rounds)
                    if all_valid else None
                ),
                "median_candidate_scores_per_second": (
                    statistics.median(
                        result["candidate_scores_per_second"] for result in rounds
                    ) if all_valid else None
                ),
                "median_p95_latency_ms": (
                    statistics.median(result["p95_latency_ms"] for result in rounds)
                    if all_valid else None
                ),
                "min_p95_latency_ms": (
                    min(result["p95_latency_ms"] for result in rounds)
                    if all_valid else None
                ),
                "max_p95_latency_ms": (
                    max(result["p95_latency_ms"] for result in rounds)
                    if all_valid else None
                ),
            }
            summaries.append(summary)
            print(
                f"summary batch={batch_size} concurrency={concurrency} "
                f"median_QPS={summary['median_qps']} "
                f"median_P95={summary['median_p95_latency_ms']} ms",
                flush=True,
            )
    report = {
        "valid": all(case["valid"] for case in cases),
        "url": args.url,
        "health_url": args.health_url,
        "latency_scope": "client_end_to_end_http_including_connection_setup",
        "percentile_method": "nearest_rank",
        "payload_file": str(args.payload_file),
        "payload_templates": len(templates),
        "template_filter": template_filter,
        "requests_per_repeat": args.requests,
        "repeats": args.repeats,
        "cases": cases,
        "summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"valid": report["valid"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
