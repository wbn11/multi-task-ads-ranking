"""Small-artifact tests; no full Ali-CCP dataset or GPU required."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import torch
import yaml

from scripts.benchmark_service import (
    _post,
    batcher_counter_delta,
    benchmark_case,
    filter_templates_by_token_limit,
    main as benchmark_main,
    make_payload,
    percentile_nearest_rank,
    request_feature_token_count,
)
from scripts.build_rank_payload import _decode_selected_samples, build_rank_payloads
from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.models.factory import build_multitask_model
from src.serving.batcher import DynamicBatcher
from src.serving.export import export_ranker
from src.serving.ranker import Ranker


class ServingTest(unittest.TestCase):
    @staticmethod
    def _make_run(root: Path) -> tuple[Path, Path]:
        run = root / "run"
        processed = root / "processed"
        run.mkdir()
        processed.mkdir()
        encoder = FeatureEncoder.fit(
            {field: Counter({"known": 5}) for field in ALL_FIELD_IDS},
            min_frequency=1,
        )
        encoder.save(processed / "vocab.json")
        model_config = {
            "name": "dcn_ple_esmm",
            "embedding_dim": 2,
            "embedding_pooling": "weighted_mean",
            "num_cross_layers": 1,
            "cross_layer_norm": True,
            "num_shared_experts": 1,
            "num_task_experts": 1,
            "expert_hidden_dims": [8],
            "tower_hidden_dims": [4],
            "dropout": 0.0,
            "gate_dropout": 0.0,
        }
        model = build_multitask_model(encoder, model_config)
        torch.save({"epoch": 2, "model_state_dict": model.state_dict()}, run / "best.pt")
        (run / "config.yaml").write_text(
            yaml.safe_dump({"model_config": {"model": model_config}}),
            encoding="utf-8",
        )
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "negative_sampling": {"negative_keep_probability": 1.0},
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                }
            ),
            encoding="utf-8",
        )
        return run, processed

    def test_export_and_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, processed = self._make_run(root)
            platt_state = {
                "type": "platt",
                "slope": 0.8,
                "intercept": -0.2,
                "epsilon": 1e-7,
                "max_iterations": 100,
                "tolerance": 1e-8,
                "l2_regularization": 1e-6,
                "iterations": 3,
                "converged": True,
                "training_samples": 100,
                "training_objective": 0.1,
            }
            (run / "multitask_calibrators.json").write_text(
                json.dumps(
                    {
                        "fit_split": "validation",
                        "input_stage": "raw_heads",
                        "negative_keep_probability": 1.0,
                        "test_used_for_fit": False,
                        "methods": {
                            "platt": {"ctr": platt_state, "cvr": platt_state}
                        },
                    }
                ),
                encoding="utf-8",
            )
            artifact = root / "artifact"
            report = export_ranker(
                run_directory=run,
                processed_directory=processed,
                output_directory=artifact,
            )
            self.assertTrue(report["valid"])
            self.assertTrue(report["platt_available"])
            self.assertLessEqual(report["trace_max_absolute_error"], 1e-6)
            ranker = Ranker(artifact)
            result = ranker.rank(
                request_id="r1",
                user_features={"101": [{"feature_id": "known", "value": 1.0}]},
                context_features={"301": [{"feature_id": "known"}]},
                candidates=[
                    {
                        "ad_id": "ad-a",
                        "features": {"205": [{"feature_id": "known"}]},
                    },
                    {
                        "ad_id": "ad-b",
                        "features": {
                            "205": [
                                {"feature_id": "unknown"},
                                {"feature_id": "known", "value": 2.0},
                            ]
                        },
                    },
                    {"ad_id": "ad-c", "features": {}},
                ],
            )
            self.assertEqual(len(result["results"]), 3)
            self.assertEqual([row["rank"] for row in result["results"]], [1, 2, 3])
            self.assertEqual({row["ad_id"] for row in result["results"]}, {"ad-a", "ad-b", "ad-c"})
            for row in result["results"]:
                self.assertAlmostEqual(row["pctcvr"], row["pctr"] * row["pcvr"], places=6)
                self.assertEqual(row["score"], row["pctcvr"])
            calibrated = ranker.rank(
                request_id="r2",
                user_features={},
                context_features={},
                candidates=[{"ad_id": "ad-a", "features": {}}],
                probability_mode="platt",
            )
            self.assertEqual(calibrated["probability_mode"], "platt")
            self.assertAlmostEqual(
                calibrated["results"][0]["pctcvr"],
                calibrated["results"][0]["pctr"] * calibrated["results"][0]["pcvr"],
                places=6,
            )
            with self.assertRaisesRegex(ValueError, "invalid field"):
                ranker.rank(
                    request_id="bad",
                    user_features={"205": [{"feature_id": "known"}]},
                    context_features={},
                    candidates=[{"ad_id": "x", "features": {}}],
                )
            batched = ranker.rank_many(
                [
                    {
                        "request_id": "many-1",
                        "user_features": {},
                        "context_features": {},
                        "candidates": [{"ad_id": "a", "features": {}}],
                    },
                    {
                        "request_id": "many-2",
                        "user_features": {},
                        "context_features": {},
                        "candidates": [
                            {"ad_id": "b", "features": {}},
                            {"ad_id": "c", "features": {}},
                        ],
                    },
                ]
            )
            self.assertEqual(
                [result["request_id"] for result in batched],
                ["many-1", "many-2"],
            )
            self.assertEqual([len(result["results"]) for result in batched], [1, 2])
            single = ranker.rank(
                request_id="many-1",
                user_features={},
                context_features={},
                candidates=[{"ad_id": "a", "features": {}}],
            )
            for key in ("pctr", "pcvr", "pctcvr", "score"):
                self.assertAlmostEqual(
                    batched[0]["results"][0][key],
                    single["results"][0][key],
                    places=7,
                )

    def test_missing_vocab_is_rejected_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, processed = self._make_run(root)
            (processed / "vocab.json").unlink()
            artifact = root / "artifact"
            with self.assertRaises(FileNotFoundError):
                export_ranker(
                    run_directory=run,
                    processed_directory=processed,
                    output_directory=artifact,
                )
            self.assertFalse(artifact.exists())

    def test_benchmark_helpers(self) -> None:
        self.assertEqual(percentile_nearest_rank([1, 2, 3, 4], 0.95), 4)
        template = {"candidates": [{"ad_id": "item", "features": {}}]}
        payload = make_payload(template, 3)
        payload["request_id"] = "first-user"
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertEqual(len({row["ad_id"] for row in payload["candidates"]}), 3)
        second = {**payload, "request_id": "another-user"}
        with patch("scripts.benchmark_service._post", return_value=2.0) as post:
            result = benchmark_case(
                url="http://127.0.0.1:8000/rank",
                payloads=[payload, second],
                requests=5,
                warmup=1,
                concurrency=2,
                timeout=1.0,
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload_templates"], 2)
        self.assertEqual(result["p95_latency_ms"], 2.0)
        self.assertGreater(result["qps"], 0)
        request_ids = {
            json.loads(call.args[1])["request_id"]
            for call in post.call_args_list
        }
        self.assertIn("another-user", request_ids)

    def test_benchmark_filters_templates_for_largest_batch(self) -> None:
        def tokens(count: int) -> list[dict[str, object]]:
            return [
                {"feature_id": f"token-{index}", "value": 1.0}
                for index in range(count)
            ]

        safe = {
            "request_id": "safe",
            "user_features": {"101": tokens(1)},
            "context_features": {},
            "candidates": [
                {"ad_id": "safe-item", "features": {"205": tokens(1)}}
            ],
        }
        oversized = {
            "request_id": "oversized",
            "user_features": {"101": tokens(3)},
            "context_features": {},
            "candidates": [
                {"ad_id": "large-item", "features": {"205": tokens(2)}}
            ],
        }
        self.assertEqual(request_feature_token_count(make_payload(safe, 4)), 8)
        self.assertEqual(
            request_feature_token_count(make_payload(oversized, 4)), 20
        )

        retained, report = filter_templates_by_token_limit(
            [safe, oversized],
            batch_sizes=[1, 4],
            max_request_feature_tokens=10,
        )

        self.assertEqual(retained, [safe])
        self.assertEqual(report["validation_batch_size"], 4)
        self.assertEqual(report["retained_indices"], [0])
        self.assertEqual(
            report["dropped_templates"],
            [
                {
                    "index": 1,
                    "request_id": "oversized",
                    "estimated_tokens": 20,
                }
            ],
        )

    def test_benchmark_displays_http_422_response(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8000/rank",
            422,
            "Unprocessable Entity",
            {},
            BytesIO(b'{"detail":"request contains too many feature tokens"}'),
        )
        with patch("scripts.benchmark_service.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                RuntimeError, "HTTP 422.*request contains too many feature tokens"
            ):
                _post(
                    "http://127.0.0.1:8000/rank",
                    b"{}",
                    timeout=1.0,
                    expected_count=1,
                )

    def test_benchmark_warmup_displays_failing_template(self) -> None:
        payload = {"candidates": [{"ad_id": "item", "features": {}}]}
        with patch(
            "scripts.benchmark_service._post",
            side_effect=[1.0, RuntimeError("HTTP 422: rejected")],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "batch_size=1 concurrency=2 template_index=1.*HTTP 422",
            ):
                benchmark_case(
                    url="http://127.0.0.1:8000/rank",
                    payloads=[payload, payload],
                    requests=1,
                    warmup=2,
                    concurrency=2,
                    timeout=1.0,
                )

    def test_real_sample_payload_reverses_known_and_unk_indices(self) -> None:
        encoder = FeatureEncoder.fit(
            {field: Counter({"known": 5}) for field in ALL_FIELD_IDS},
            min_frequency=1,
        )
        sample = {
            "features": {
                "101": {"ids": [2], "values": [1.0]},
                "301": {"ids": [2], "values": [1.0]},
                "205": {"ids": [2], "values": [1.0]},
                "508": {"ids": [1], "values": [2.5]},
            }
        }
        payload = _decode_selected_samples(encoder, [sample])
        self.assertEqual(payload["user_features"]["101"][0]["feature_id"], "known")
        unseen = payload["candidates"][0]["features"]["508"][0]
        self.assertEqual(encoder.encode_token("508", unseen["feature_id"]), 1)
        self.assertEqual(unseen["value"], 2.5)

    def test_multi_user_payload_selection_keeps_contexts_separate(self) -> None:
        encoder = FeatureEncoder.fit(
            {field: Counter({"known": 5}) for field in ALL_FIELD_IDS},
            min_frequency=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            processed = Path(temporary)
            encoder.save(processed / "vocab.json")

            def sample(user: str, context_id: int) -> dict:
                return {
                    "user_id": user,
                    "features": {
                        "101": {"ids": [2], "values": [1.0]},
                        "301": {"ids": [context_id], "values": [1.0]},
                        "205": {"ids": [2], "values": [1.0]},
                    },
                }

            rows = [
                sample("u1", 2),
                sample("u2", 2),
                sample("u1", 1),  # Same user but a different scene: skip.
                sample("u1", 2),
                sample("u2", 2),
            ]

            class FakeDataset:
                def __init__(self, _path, *, split):
                    self.split = split

                def __iter__(self):
                    return iter(rows)

                def close(self):
                    pass

            with patch("scripts.build_rank_payload.AdsDataset", FakeDataset):
                payloads, stats = build_rank_payloads(
                    processed, users=2, max_candidates=2, max_scanned=10
                )
            self.assertEqual(stats["selected_users"], 2)
            self.assertEqual(stats["candidates_per_user"], [2, 2])
            self.assertEqual(len(payloads), 2)
            self.assertNotEqual(payloads[0]["request_id"], payloads[1]["request_id"])

    def test_benchmark_cli_summarizes_three_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload_file = root / "requests.json"
            output_file = root / "benchmark.json"
            payload_file.write_text(
                json.dumps(
                    [
                        {"request_id": "u1", "candidates": [{"ad_id": "a"}]},
                        {"request_id": "u2", "candidates": [{"ad_id": "b"}]},
                    ]
                ),
                encoding="utf-8",
            )
            rounds = [
                {
                    "valid": True,
                    "qps": qps,
                    "candidate_scores_per_second": qps,
                    "p95_latency_ms": p95,
                    "successful_requests": 5,
                }
                for qps, p95 in [(10.0, 20.0), (12.0, 30.0), (11.0, 25.0)]
            ]
            argv = [
                "benchmark_service.py",
                "--payload-file", str(payload_file),
                "--batch-sizes", "1",
                "--concurrencies", "1",
                "--requests", "5",
                "--warmup", "0",
                "--repeats", "3",
                "--output", str(output_file),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.benchmark_service.benchmark_case", side_effect=rounds
            ) as benchmark:
                benchmark_main()
            self.assertEqual(benchmark.call_count, 3)
            self.assertEqual(len(benchmark.call_args.kwargs["payloads"]), 2)
            report = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            self.assertEqual(report["payload_templates"], 2)
            self.assertEqual(report["summary"][0]["median_qps"], 11.0)
            self.assertEqual(report["summary"][0]["median_p95_latency_ms"], 25.0)

    def test_batcher_counter_delta(self) -> None:
        before = {
            "batcher": {
                "submitted_requests": 10,
                "processed_requests": 10,
                "successful_requests": 10,
                "failed_requests": 0,
                "rejected_requests": 0,
                "processed_candidates": 20,
                "batches": 5,
                "total_queue_wait_ms": 10.0,
                "total_batch_execution_ms": 25.0,
            }
        }
        after = {
            "batcher": {
                "submitted_requests": 18,
                "processed_requests": 18,
                "successful_requests": 18,
                "failed_requests": 0,
                "rejected_requests": 0,
                "processed_candidates": 84,
                "batches": 7,
                "total_queue_wait_ms": 26.0,
                "total_batch_execution_ms": 35.0,
                "max_requests": 8,
                "max_candidates": 256,
                "max_wait_ms": 2.0,
                "max_queue_size": 1024,
            }
        }
        delta = batcher_counter_delta(before, after)
        self.assertTrue(delta["enabled"])
        self.assertEqual(delta["processed_requests"], 8)
        self.assertEqual(delta["processed_candidates"], 64)
        self.assertEqual(delta["batches"], 2)
        self.assertEqual(delta["average_requests_per_batch"], 4.0)
        self.assertEqual(delta["average_candidates_per_batch"], 32.0)


class _FakeRanker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def prepare_request(self, **payload):
        if payload["request_id"] == "invalid":
            raise ValueError("invalid request")
        return payload

    def rank_prepared_many(self, requests):
        self.batch_sizes.append(len(requests))
        return [
            {"request_id": request["request_id"], "results": []}
            for request in requests
        ]


class DynamicBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_are_merged_and_split(self) -> None:
        ranker = _FakeRanker()
        batcher = DynamicBatcher(
            ranker,
            max_requests=4,
            max_candidates=8,
            max_wait_ms=50.0,
            max_queue_size=16,
        )
        await batcher.start()
        try:
            tasks = [
                asyncio.create_task(
                    batcher.submit(
                        {
                            "request_id": f"r{index}",
                            "candidates": [{"ad_id": f"a{index}"}],
                        }
                    )
                )
                for index in range(4)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            await batcher.close()

        self.assertEqual(ranker.batch_sizes, [4])
        self.assertEqual(
            [result["request_id"] for result in results],
            ["r0", "r1", "r2", "r3"],
        )
        stats = batcher.stats()
        self.assertEqual(stats["batches"], 1)
        self.assertEqual(stats["processed_requests"], 4)
        self.assertEqual(stats["average_requests_per_batch"], 4.0)

    async def test_invalid_request_does_not_fail_its_batch_peer(self) -> None:
        ranker = _FakeRanker()
        batcher = DynamicBatcher(
            ranker,
            max_requests=2,
            max_candidates=8,
            max_wait_ms=50.0,
        )
        await batcher.start()
        try:
            outcomes = await asyncio.gather(
                batcher.submit(
                    {"request_id": "valid", "candidates": [{"ad_id": "a"}]}
                ),
                batcher.submit(
                    {"request_id": "invalid", "candidates": [{"ad_id": "b"}]}
                ),
                return_exceptions=True,
            )
        finally:
            await batcher.close()

        self.assertIsInstance(outcomes[0], dict)
        self.assertIsInstance(outcomes[1], ValueError)
        self.assertEqual(ranker.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
