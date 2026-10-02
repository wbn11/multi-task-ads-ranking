"""Create one or more /rank requests from processed Ali-CCP Test impressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AdsDataset
from src.data.feature_encoder import FeatureEncoder, PAD_INDEX, UNK_INDEX
from src.data.feature_schema import COMMON_FIELD_IDS, SAMPLE_FIELD_IDS


def _decode_selected_samples(
    encoder: FeatureEncoder,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reverse only token indices used by these samples, not the entire vocab."""

    needed: dict[str, set[int]] = {}
    for sample in samples:
        for field_id, field in sample["features"].items():
            needed.setdefault(field_id, set()).update(int(token) for token in field["ids"])

    inverse: dict[str, dict[int, str]] = {}
    for field_id, indices in needed.items():
        found = {
            index: token
            for token, index in encoder.field_vocabularies[field_id].token_to_index.items()
            if index in indices
        }
        if UNK_INDEX in indices:
            placeholder = "__aliccp_unseen_test_token__"
            while encoder.encode_token(field_id, placeholder) != UNK_INDEX:
                placeholder += "_"
            found[UNK_INDEX] = placeholder
        if PAD_INDEX in indices or set(found) != indices:
            raise ValueError(f"cannot reverse encoded field {field_id}")
        inverse[field_id] = found

    def decode(fields: dict[str, Any], allowed: set[str]) -> dict[str, list[dict[str, Any]]]:
        decoded: dict[str, list[dict[str, Any]]] = {}
        for field_id, field in fields.items():
            if field_id not in allowed:
                continue
            ids = field["ids"]
            values = field["values"]
            if len(ids) != len(values):
                raise ValueError(f"ids/values length mismatch for field {field_id}")
            decoded[field_id] = [
                {"feature_id": inverse[field_id][int(index)], "value": float(value)}
                for index, value in zip(ids, values, strict=True)
            ]
        return decoded

    first = samples[0]
    return {
        "request_id": "aliccp-test-offline-sample",
        "user_features": decode(first["features"], set(COMMON_FIELD_IDS)),
        "context_features": decode(first["features"], {"301"}),
        "candidates": [
            {
                "ad_id": f"test-impression-{index + 1}",
                "features": decode(
                    sample["features"], set(SAMPLE_FIELD_IDS) - {"301"}
                ),
            }
            for index, sample in enumerate(samples)
        ],
        "score_mode": "ctcvr",
        "probability_mode": "raw",
    }


def build_rank_payload(
    processed_directory: str | Path,
    *,
    max_candidates: int = 8,
    max_scanned: int = 1_000_000,
) -> tuple[dict[str, Any], dict[str, int]]:
    payloads, stats = build_rank_payloads(
        processed_directory,
        users=1,
        max_candidates=max_candidates,
        max_scanned=max_scanned,
    )
    return payloads[0], {
        "rows_scanned": stats["rows_scanned"],
        "selected_candidates": stats["candidates_per_user"][0],
    }


def build_rank_payloads(
    processed_directory: str | Path,
    *,
    users: int,
    max_candidates: int = 8,
    max_scanned: int = 1_000_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= users <= 128:
        raise ValueError("users must be in [1, 128]")
    if not 1 <= max_candidates <= 128:
        raise ValueError("max_candidates must be in [1, 128]")
    if max_scanned < max_candidates:
        raise ValueError("max_scanned must be at least max_candidates")
    processed_dir = Path(processed_directory)
    encoder = FeatureEncoder.load(processed_dir / "vocab.json")
    dataset = AdsDataset(processed_dir, split="test")
    groups: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    rows_scanned = 0
    try:
        for sample in dataset:
            rows_scanned += 1
            user_id = sample.get("user_id")
            if user_id is not None:
                user_id = str(user_id)
            if user_id:
                context = sample["features"].get("301")
                if user_id not in groups and len(groups) < users:
                    groups[user_id] = (context, [])
                group = groups.get(user_id)
                if group is not None and context == group[0]:
                    if len(group[1]) < max_candidates:
                        group[1].append(sample)
                if len(groups) == users and all(
                    len(selected) >= max_candidates for _, selected in groups.values()
                ):
                    break
            if rows_scanned >= max_scanned:
                break
    finally:
        dataset.close()
    if not groups:
        raise RuntimeError("no Test impression with a user ID was found")
    payloads = [
        _decode_selected_samples(encoder, selected)
        for _, selected in groups.values()
    ]
    for index, payload in enumerate(payloads, start=1):
        payload["request_id"] = f"aliccp-test-offline-user-{index}"
    return payloads, {
        "rows_scanned": rows_scanned,
        "selected_users": len(payloads),
        "candidates_per_user": [len(payload["candidates"]) for payload in payloads],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-directory", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-scanned", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads, stats = build_rank_payloads(
        args.processed_directory,
        users=args.users,
        max_candidates=args.max_candidates,
        max_scanned=args.max_scanned,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payloads[0] if args.users == 1 else payloads,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"valid": True, "output": str(args.output), **stats}))


if __name__ == "__main__":
    main()
