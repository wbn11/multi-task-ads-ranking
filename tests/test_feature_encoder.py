from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data.feature_encoder import FeatureEncoder, PAD_INDEX, UNK_INDEX


class FeatureEncoderTest(unittest.TestCase):
    def test_per_field_vocabulary_and_unknown_mapping(self) -> None:
        frequencies = {
            "101": Counter({"user_a": 3, "user_rare": 1}),
            "205": Counter({"item_a": 2}),
        }
        encoder = FeatureEncoder.fit(frequencies, min_frequency=2)

        self.assertEqual(PAD_INDEX, 0)
        self.assertEqual(UNK_INDEX, 1)
        self.assertEqual(encoder.encode_token("101", "user_a"), 2)
        self.assertEqual(encoder.encode_token("101", "user_rare"), UNK_INDEX)
        self.assertEqual(encoder.encode_token("101", "new_user"), UNK_INDEX)
        self.assertEqual(encoder.encode_token("205", "item_a"), 2)
        self.assertEqual(encoder.field_vocabularies["101"].vocab_size, 3)

    def test_vocabulary_round_trip(self) -> None:
        encoder = FeatureEncoder.fit(
            {"101": Counter({"user_a": 2})},
            min_frequency=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vocab.json"
            encoder.save(path)
            restored = FeatureEncoder.load(path)

        self.assertEqual(restored.min_frequency, 2)
        self.assertEqual(restored.encode_token("101", "user_a"), 2)
        self.assertEqual(restored.encode_token("101", "new_user"), UNK_INDEX)


if __name__ == "__main__":
    unittest.main()
