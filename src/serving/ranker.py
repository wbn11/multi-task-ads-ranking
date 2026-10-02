"""Feature encoding and one-request batched inference for the /rank demo."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.calibration.platt import PlattCalibrator
from src.data.dataset import collate_ads_batch
from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import COMMON_FIELD_IDS, SAMPLE_FIELD_IDS
from src.data.parser import SparseFeature
from src.serving.export import EXPORT_FORMAT_VERSION


MAX_CANDIDATES = 128
MAX_TOKENS_PER_FIELD = 1024
MAX_TOKENS_PER_REQUEST = 16384


@dataclass(slots=True)
class PreparedRankRequest:
    """One validated request whose candidates are ready for collation."""

    request_id: str
    ad_ids: list[str]
    samples: list[dict[str, Any]]
    score_mode: str
    probability_mode: str

    @property
    def candidate_count(self) -> int:
        return len(self.samples)


class Ranker:
    """Load one immutable artifact and score one or more ranking requests."""

    def __init__(self, artifact_directory: str | Path, *, device: str = "cpu") -> None:
        artifact_dir = Path(artifact_directory)
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != EXPORT_FORMAT_VERSION:
            raise ValueError("unsupported ranking artifact format")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.device = torch.device(device)
        self.model_version = str(manifest["model_version"])
        self.encoder = FeatureEncoder.load(artifact_dir / "vocab.json")
        self.model = torch.jit.load(
            str(artifact_dir / "ranker.ts"), map_location=self.device
        ).eval()
        self.platt: dict[str, PlattCalibrator] | None = None
        platt_path = artifact_dir / "platt.json"
        if platt_path.is_file():
            payload = json.loads(platt_path.read_text(encoding="utf-8"))
            if payload.get("method") != "platt" or payload.get("fit_split") != "validation":
                raise ValueError("invalid Platt calibration artifact")
            self.platt = {
                head: PlattCalibrator.from_state_dict(payload["heads"][head])
                for head in ("ctr", "cvr")
            }

    @staticmethod
    def _tokens(
        fields: Mapping[str, Any],
        *,
        allowed: set[str],
        name: str,
    ) -> list[SparseFeature]:
        if not isinstance(fields, Mapping):
            raise ValueError(f"{name} must be a field-to-token-list mapping")
        result: list[SparseFeature] = []
        for field_id, tokens in fields.items():
            if field_id not in allowed:
                raise ValueError(f"{name} contains invalid field {field_id!r}")
            if not isinstance(tokens, list) or len(tokens) > MAX_TOKENS_PER_FIELD:
                raise ValueError(
                    f"{name}[{field_id}] must be a list of at most "
                    f"{MAX_TOKENS_PER_FIELD} tokens"
                )
            for token in tokens:
                if not isinstance(token, Mapping):
                    raise ValueError(f"{name}[{field_id}] contains a non-object token")
                feature_id = token.get("feature_id")
                if not isinstance(feature_id, str) or not feature_id:
                    raise ValueError("each token needs a non-empty raw feature_id")
                try:
                    value = float(token.get("value", 1.0))
                except (TypeError, ValueError) as error:
                    raise ValueError("token value must be numeric") from error
                if not math.isfinite(value):
                    raise ValueError("token value must be finite")
                result.append(SparseFeature(field_id, feature_id, str(value)))
        return result

    def prepare_request(
        self,
        *,
        request_id: str,
        user_features: Mapping[str, Any],
        context_features: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        score_mode: str = "ctcvr",
        probability_mode: str = "raw",
    ) -> PreparedRankRequest:
        """Validate and encode one request without running the model."""

        if score_mode not in {"ctr", "ctcvr"}:
            raise ValueError("score_mode must be ctr or ctcvr")
        if probability_mode not in {"raw", "platt"}:
            raise ValueError("probability_mode must be raw or platt")
        if probability_mode == "platt" and self.platt is None:
            raise ValueError("the export has no validation-fitted Platt calibrator")
        if not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise ValueError(f"candidates must contain 1 to {MAX_CANDIDATES} ads")

        shared_tokens = self._tokens(
            user_features, allowed=set(COMMON_FIELD_IDS), name="user_features"
        ) + self._tokens(
            context_features, allowed={"301"}, name="context_features"
        )
        candidate_field_ids = set(SAMPLE_FIELD_IDS) - {"301"}
        samples: list[dict[str, Any]] = []
        ad_ids: list[str] = []
        total_tokens = len(shared_tokens) * len(candidates)
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("each candidate must be an object")
            ad_id = candidate.get("ad_id")
            if not isinstance(ad_id, str) or not ad_id:
                raise ValueError("each candidate needs a non-empty string ad_id")
            candidate_tokens = self._tokens(
                candidate.get("features", {}),
                allowed=candidate_field_ids,
                name=f"candidate {ad_id!r} features",
            )
            total_tokens += len(candidate_tokens)
            if total_tokens > MAX_TOKENS_PER_REQUEST:
                raise ValueError("request contains too many feature tokens")
            encoded = self.encoder.encode_features([*shared_tokens, *candidate_tokens])
            samples.append(
                {
                    "sample_id": ad_id,
                    "common_feature_id": request_id,
                    "user_id": None,
                    "features": encoded,
                    "click": 0,
                    "conversion": 0,
                    "ctcvr": 0,
                }
            )
            ad_ids.append(ad_id)
        if len(set(ad_ids)) != len(ad_ids):
            raise ValueError("candidate ad_id values must be unique within a request")

        return PreparedRankRequest(
            request_id=request_id,
            ad_ids=ad_ids,
            samples=samples,
            score_mode=score_mode,
            probability_mode=probability_mode,
        )

    def rank_prepared_many(
        self, requests: Sequence[PreparedRankRequest]
    ) -> list[dict[str, Any]]:
        """Collate many requests, execute one forward, then split the outputs."""

        if not requests:
            raise ValueError("at least one prepared request is required")
        samples = [sample for request in requests for sample in request.samples]
        batch = collate_ads_batch(samples)
        features = {
            field: {name: tensor.to(self.device) for name, tensor in tensors.items()}
            for field, tensors in batch["features"].items()
        }
        with torch.inference_mode():
            ctr_tensor, cvr_tensor, ctcvr_tensor = self.model(features)
        ctr = ctr_tensor.detach().cpu().numpy()
        cvr = cvr_tensor.detach().cpu().numpy()
        ctcvr = ctcvr_tensor.detach().cpu().numpy()
        if not (
            torch.isfinite(torch.as_tensor(ctr)).all()
            and torch.isfinite(torch.as_tensor(cvr)).all()
            and torch.isfinite(torch.as_tensor(ctcvr)).all()
        ):
            raise RuntimeError("model returned non-finite probabilities")

        results: list[dict[str, Any]] = []
        offset = 0
        for request in requests:
            end = offset + request.candidate_count
            request_ctr = ctr[offset:end]
            request_cvr = cvr[offset:end]
            request_ctcvr = ctcvr[offset:end]
            if request.probability_mode == "platt":
                assert self.platt is not None
                request_ctr = self.platt["ctr"].transform(request_ctr)
                request_cvr = self.platt["cvr"].transform(request_cvr)
                request_ctcvr = request_ctr * request_cvr
            if not (
                torch.isfinite(torch.as_tensor(request_ctr)).all()
                and torch.isfinite(torch.as_tensor(request_cvr)).all()
                and torch.isfinite(torch.as_tensor(request_ctcvr)).all()
            ):
                raise RuntimeError("calibration returned non-finite probabilities")

            records = []
            for index, ad_id in enumerate(request.ad_ids):
                pctr = float(request_ctr[index])
                pcvr = float(request_cvr[index])
                pctcvr = float(request_ctcvr[index])
                records.append(
                    {
                        "ad_id": ad_id,
                        "pctr": pctr,
                        "pcvr": pcvr,
                        "pctcvr": pctcvr,
                        "score": (
                            pctr if request.score_mode == "ctr" else pctcvr
                        ),
                    }
                )
            records.sort(key=lambda row: row["score"], reverse=True)
            for rank, row in enumerate(records, start=1):
                row["rank"] = rank
            results.append(
                {
                    "request_id": request.request_id,
                    "model_version": self.model_version,
                    "score_mode": request.score_mode,
                    "probability_mode": request.probability_mode,
                    "results": records,
                }
            )
            offset = end
        if offset != len(samples):
            raise RuntimeError("batched ranking output split is inconsistent")
        return results

    def rank_many(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate multiple request mappings and score them in one forward."""

        prepared = [self.prepare_request(**dict(request)) for request in requests]
        return self.rank_prepared_many(prepared)

    def rank(
        self,
        *,
        request_id: str,
        user_features: Mapping[str, Any],
        context_features: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        score_mode: str = "ctcvr",
        probability_mode: str = "raw",
    ) -> dict[str, Any]:
        """Compatibility wrapper for one-request inference."""

        prepared = self.prepare_request(
            request_id=request_id,
            user_features=user_features,
            context_features=context_features,
            candidates=candidates,
            score_mode=score_mode,
            probability_mode=probability_mode,
        )
        return self.rank_prepared_many([prepared])[0]
