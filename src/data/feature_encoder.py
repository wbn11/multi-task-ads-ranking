"""Per-field vocabularies for converting raw Ali-CCP IDs to model indices."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .feature_schema import ALL_FIELD_IDS, FIELD_SPEC_BY_ID
from .parser import SparseFeature


PAD_INDEX = 0
UNK_INDEX = 1
FIRST_KNOWN_INDEX = 2
VOCAB_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class FieldVocabulary:
    field_id: str
    token_to_index: dict[str, int]
    raw_cardinality: int
    retained_cardinality: int
    dropped_occurrences: int

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_index) + FIRST_KNOWN_INDEX

    def encode(self, raw_feature_id: str) -> int:
        return self.token_to_index.get(raw_feature_id, UNK_INDEX)


class FeatureEncoder:
    """Encode each field in an independent, train-fitted vocabulary."""

    def __init__(
        self,
        field_vocabularies: Mapping[str, FieldVocabulary],
        *,
        min_frequency: int,
    ) -> None:
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        missing_fields = set(ALL_FIELD_IDS) - set(field_vocabularies)
        extra_fields = set(field_vocabularies) - set(ALL_FIELD_IDS)
        if missing_fields or extra_fields:
            raise ValueError(
                f"field vocabulary mismatch: missing={sorted(missing_fields)}, "
                f"extra={sorted(extra_fields)}"
            )
        self.field_vocabularies = dict(field_vocabularies)
        self.min_frequency = min_frequency

    @classmethod
    def fit(
        cls,
        frequencies: Mapping[str, Counter[str]],
        *,
        min_frequency: int,
    ) -> "FeatureEncoder":
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        vocabularies: dict[str, FieldVocabulary] = {}
        for field_id in ALL_FIELD_IDS:
            counter = frequencies.get(field_id, Counter())
            retained = [
                (raw_feature_id, count)
                for raw_feature_id, count in counter.items()
                if count >= min_frequency
            ]
            retained.sort(key=lambda item: (-item[1], item[0]))
            token_to_index = {
                raw_feature_id: index
                for index, (raw_feature_id, _) in enumerate(
                    retained, start=FIRST_KNOWN_INDEX
                )
            }
            dropped_occurrences = sum(
                count for count in counter.values() if count < min_frequency
            )
            vocabularies[field_id] = FieldVocabulary(
                field_id=field_id,
                token_to_index=token_to_index,
                raw_cardinality=len(counter),
                retained_cardinality=len(token_to_index),
                dropped_occurrences=dropped_occurrences,
            )
        return cls(vocabularies, min_frequency=min_frequency)

    def encode_token(self, field_id: str, raw_feature_id: str) -> int:
        try:
            vocabulary = self.field_vocabularies[field_id]
        except KeyError as error:
            raise KeyError(f"unknown Ali-CCP field_id: {field_id}") from error
        return vocabulary.encode(raw_feature_id)

    def encode_features(
        self, features: Iterable[SparseFeature]
    ) -> dict[str, dict[str, list[int] | list[float]]]:
        encoded: dict[str, dict[str, list[int] | list[float]]] = {}
        for feature in features:
            if feature.field_id not in FIELD_SPEC_BY_ID:
                raise ValueError(f"unexpected Ali-CCP field_id: {feature.field_id}")
            try:
                value = float(feature.value)
            except ValueError as error:
                raise ValueError(
                    f"field {feature.field_id} feature {feature.feature_id} has "
                    f"a non-numeric value: {feature.value!r}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"field {feature.field_id} feature {feature.feature_id} has "
                    f"a non-finite value: {feature.value!r}"
                )

            field = encoded.setdefault(feature.field_id, {"ids": [], "values": []})
            ids = field["ids"]
            values = field["values"]
            assert isinstance(ids, list)
            assert isinstance(values, list)
            ids.append(self.encode_token(feature.field_id, feature.feature_id))
            values.append(value)
        return encoded

    def field_summary(self) -> dict[str, dict[str, int | str]]:
        return {
            field_id: {
                "name": FIELD_SPEC_BY_ID[field_id].name,
                "domain": FIELD_SPEC_BY_ID[field_id].domain,
                "raw_cardinality": vocabulary.raw_cardinality,
                "retained_cardinality": vocabulary.retained_cardinality,
                "vocab_size_with_special_tokens": vocabulary.vocab_size,
                "dropped_occurrences": vocabulary.dropped_occurrences,
            }
            for field_id, vocabulary in self.field_vocabularies.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": VOCAB_FORMAT_VERSION,
            "min_frequency": self.min_frequency,
            "special_indices": {"pad": PAD_INDEX, "unk": UNK_INDEX},
            "fields": {
                field_id: {
                    "spec": FIELD_SPEC_BY_ID[field_id].as_dict(),
                    "token_to_index": vocabulary.token_to_index,
                    "raw_cardinality": vocabulary.raw_cardinality,
                    "retained_cardinality": vocabulary.retained_cardinality,
                    "dropped_occurrences": vocabulary.dropped_occurrences,
                }
                for field_id, vocabulary in self.field_vocabularies.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureEncoder":
        if payload.get("format_version") != VOCAB_FORMAT_VERSION:
            raise ValueError("unsupported feature vocabulary format version")
        min_frequency = int(payload["min_frequency"])
        fields = payload["fields"]
        if not isinstance(fields, Mapping):
            raise ValueError("vocabulary fields must be a mapping")

        vocabularies: dict[str, FieldVocabulary] = {}
        for field_id in ALL_FIELD_IDS:
            field_payload = fields.get(field_id)
            if not isinstance(field_payload, Mapping):
                raise ValueError(f"missing vocabulary for field {field_id}")
            raw_mapping = field_payload.get("token_to_index")
            if not isinstance(raw_mapping, Mapping):
                raise ValueError(f"invalid token mapping for field {field_id}")
            token_to_index = {
                str(token): int(index) for token, index in raw_mapping.items()
            }
            vocabularies[field_id] = FieldVocabulary(
                field_id=field_id,
                token_to_index=token_to_index,
                raw_cardinality=int(field_payload["raw_cardinality"]),
                retained_cardinality=int(field_payload["retained_cardinality"]),
                dropped_occurrences=int(field_payload["dropped_occurrences"]),
            )
        return cls(vocabularies, min_frequency=min_frequency)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("vocabulary root must be a mapping")
        return cls.from_dict(payload)
