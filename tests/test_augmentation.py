"""Tests for the data augmentation module."""

import json
import os
import random
import tempfile
import unittest
from collections import defaultdict

from src.data.augmentation import (
    STAGE_DEFAULTS,
    add_grammar_noise,
    add_spelling_noise,
    add_synthetic_noise,
    augment_dataset,
    balanced_sampling,
    generate_noisy_samples,
    inject_identity_samples,
)


def _make_sample(lang="en", corpus="test", sid="s001", source="I is happy.", target="I am happy."):
    """Create a minimal valid sample for testing."""
    return {
        "id": sid,
        "lang": lang,
        "corpus": corpus,
        "split": "train",
        "ref": "ref1",
        "messages": [
            {"role": "user", "content": f"Correct the following {lang} text:\n\n{source}"},
            {"role": "assistant", "content": target},
        ],
        "prompt": f"Correct the following {lang} text:\n\n{source}",
        "source": source,
        "target": target,
    }


def _make_multilang_samples(n_per_lang=10):
    """Create samples across multiple languages."""
    samples = []
    langs = {"en": 30, "cs": 20, "de": 15, "ru": 10, "is": 5}
    for lang, count in langs.items():
        for i in range(count):
            samples.append(_make_sample(
                lang=lang, sid=f"{lang}_{i:03d}",
                source=f"Text with error {i} in {lang}.",
                target=f"Text without error {i} in {lang}.",
            ))
    return samples


class TestSpellingNoise(unittest.TestCase):

    def test_produces_different_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        noisy = add_spelling_noise(text, "en", noise_rate=0.2, seed=42)
        self.assertNotEqual(text, noisy)

    def test_length_roughly_preserved(self):
        text = "The quick brown fox jumps over the lazy dog."
        noisy = add_spelling_noise(text, "en", noise_rate=0.05, seed=42)
        self.assertGreater(len(noisy), len(text) * 0.5)
        self.assertLess(len(noisy), len(text) * 1.5)

    def test_deterministic_with_seed(self):
        text = "Hello world, this is a test sentence."
        a = add_spelling_noise(text, "en", noise_rate=0.1, seed=123)
        b = add_spelling_noise(text, "en", noise_rate=0.1, seed=123)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        text = "Hello world, this is a test sentence for noise."
        a = add_spelling_noise(text, "en", noise_rate=0.2, seed=1)
        b = add_spelling_noise(text, "en", noise_rate=0.2, seed=2)
        self.assertNotEqual(a, b)

    def test_short_text(self):
        # Should not crash on very short text
        result = add_spelling_noise("Hi", "en", noise_rate=0.5, seed=42)
        self.assertIsInstance(result, str)

    def test_backward_compat_alias(self):
        # add_synthetic_noise should be identical to add_spelling_noise
        self.assertIs(add_synthetic_noise, add_spelling_noise)


class TestGrammarNoise(unittest.TestCase):

    def test_produces_different_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        noisy = add_grammar_noise(text, "en", noise_rate=0.2, seed=42)
        self.assertNotEqual(text, noisy)

    def test_word_deletion(self):
        text = "a b c d e f g h i j"
        noisy = add_grammar_noise(text, "en", noise_rate=0.3, seed=42)
        # Should have fewer or more words (due to deletion/duplication mix)
        self.assertNotEqual(text, noisy)

    def test_deterministic_with_seed(self):
        text = "This is a longer test sentence with many words."
        a = add_grammar_noise(text, "en", noise_rate=0.1, seed=99)
        b = add_grammar_noise(text, "en", noise_rate=0.1, seed=99)
        self.assertEqual(a, b)

    def test_single_word(self):
        text = "Hello"
        result = add_grammar_noise(text, "en", noise_rate=0.5, seed=42)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_multilingual(self):
        # Should work on non-English text
        text = "Я живу в большом городе и люблю свою работу."
        noisy = add_grammar_noise(text, "ru", noise_rate=0.15, seed=42)
        self.assertNotEqual(text, noisy)

    def test_word_order_changed(self):
        text = "one two three four five six seven eight nine ten"
        noisy = add_grammar_noise(text, "en", noise_rate=0.3, seed=10)
        # At least some words should still be present
        original_words = set(text.split())
        noisy_words = set(noisy.split())
        # Most words should survive
        self.assertGreater(len(original_words & noisy_words), len(original_words) * 0.5)


class TestIdentityInjection(unittest.TestCase):

    def test_basic(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(100)]
        id_samples = inject_identity_samples(samples, ratio=0.1, seed=42)
        self.assertEqual(len(id_samples), 10)

    def test_source_equals_target(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        id_samples = inject_identity_samples(samples, ratio=0.1, seed=42)
        for s in id_samples:
            self.assertEqual(s["source"], s["target"])

    def test_id_suffix(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        id_samples = inject_identity_samples(samples, ratio=0.1, seed=42)
        for s in id_samples:
            self.assertTrue(s["id"].endswith("_id"))

    def test_ratio_respected(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(200)]
        for ratio in [0.05, 0.07, 0.1, 0.15]:
            id_samples = inject_identity_samples(samples, ratio=ratio, seed=42)
            expected = int(200 * ratio)
            self.assertEqual(len(id_samples), expected)


class TestGenerateNoisySamples(unittest.TestCase):

    def test_spelling_type(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        noisy = generate_noisy_samples(samples, ratio=0.2, noise_type="spelling", seed=42)
        self.assertEqual(len(noisy), 10)
        for s in noisy:
            self.assertTrue(s["id"].endswith("_aug"))

    def test_grammar_type(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        noisy = generate_noisy_samples(samples, ratio=0.2, noise_type="grammar", seed=42)
        self.assertEqual(len(noisy), 10)

    def test_mixed_type(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        noisy = generate_noisy_samples(samples, ratio=0.2, noise_type="mixed", seed=42)
        self.assertEqual(len(noisy), 10)

    def test_zero_ratio_returns_empty(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(50)]
        noisy = generate_noisy_samples(samples, ratio=0.0, seed=42)
        self.assertEqual(len(noisy), 0)

    def test_source_differs_from_target(self):
        samples = [_make_sample(
            sid=f"s{i}",
            source="This has many words in the sentence to mutate.",
            target="This has many words in the sentence to mutate.",
        ) for i in range(20)]
        noisy = generate_noisy_samples(samples, ratio=0.5, noise_rate=0.15, noise_type="grammar", seed=42)
        # At least some should have source != target
        diffs = sum(1 for s in noisy if s["source"] != s["target"])
        self.assertGreater(diffs, 0)

    def test_invalid_noise_type(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(10)]
        with self.assertRaises(ValueError):
            generate_noisy_samples(samples, ratio=0.5, noise_type="invalid", seed=42)


class TestBalancedSampling(unittest.TestCase):

    def test_sqrt_reduces_imbalance(self):
        samples = _make_multilang_samples()
        balanced = balanced_sampling(samples, strategy="sqrt", seed=42)

        by_lang = defaultdict(int)
        for s in balanced:
            by_lang[s["lang"]] += 1

        counts = list(by_lang.values())
        ratio = max(counts) / max(min(counts), 1)
        # sqrt should bring ratio closer to 1 vs original (30/5 = 6)
        self.assertLess(ratio, 5)

    def test_uniform(self):
        samples = _make_multilang_samples()
        balanced = balanced_sampling(samples, strategy="uniform", seed=42)
        by_lang = defaultdict(int)
        for s in balanced:
            by_lang[s["lang"]] += 1
        counts = list(by_lang.values())
        # All should be similar
        self.assertLess(max(counts) - min(counts), max(counts) * 0.5)

    def test_preserves_total_approximately(self):
        samples = _make_multilang_samples()
        original_count = len(samples)
        balanced = balanced_sampling(samples, strategy="sqrt", seed=42)
        # Total should be roughly the same
        self.assertGreater(len(balanced), original_count * 0.5)
        self.assertLess(len(balanced), original_count * 1.5)

    def test_target_per_lang(self):
        samples = _make_multilang_samples()
        balanced = balanced_sampling(samples, target_per_lang=10, seed=42)
        by_lang = defaultdict(int)
        for s in balanced:
            by_lang[s["lang"]] += 1
        for lang, count in by_lang.items():
            self.assertEqual(count, 10)


class TestStageDefaults(unittest.TestCase):

    def test_sft_defaults_exist(self):
        self.assertIn("sft", STAGE_DEFAULTS)
        self.assertGreater(STAGE_DEFAULTS["sft"]["noise_ratio"], 0)

    def test_grpo_defaults_no_noise(self):
        self.assertIn("grpo", STAGE_DEFAULTS)
        self.assertEqual(STAGE_DEFAULTS["grpo"]["noise_ratio"], 0.0)
        self.assertEqual(STAGE_DEFAULTS["grpo"]["noise_type"], "none")

    def test_grpo_identity_lower_than_sft(self):
        self.assertLess(
            STAGE_DEFAULTS["grpo"]["identity_ratio"],
            STAGE_DEFAULTS["sft"]["identity_ratio"],
        )


class TestAugmentDataset(unittest.TestCase):

    def _write_samples(self, samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _read_samples(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def test_sft_stage(self):
        samples = _make_multilang_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.jsonl")
            out = os.path.join(tmpdir, "output.jsonl")
            self._write_samples(samples, inp)
            result = augment_dataset(inp, out, stage="sft", seed=42)
            self.assertGreater(len(result), 0)
            # SFT should include augmented samples (_aug suffix)
            aug_ids = [s for s in result if s["id"].endswith("_aug")]
            self.assertGreater(len(aug_ids), 0)

    def test_grpo_stage_no_noise(self):
        samples = _make_multilang_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.jsonl")
            out = os.path.join(tmpdir, "output.jsonl")
            self._write_samples(samples, inp)
            result = augment_dataset(inp, out, stage="grpo", seed=42)
            self.assertGreater(len(result), 0)
            # GRPO should NOT include noise-augmented samples
            aug_ids = [s for s in result if s["id"].endswith("_aug")]
            self.assertEqual(len(aug_ids), 0)

    def test_grpo_has_identity(self):
        samples = _make_multilang_samples()
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.jsonl")
            out = os.path.join(tmpdir, "output.jsonl")
            self._write_samples(samples, inp)
            result = augment_dataset(inp, out, stage="grpo", seed=42)
            id_samples = [s for s in result if s["id"].endswith("_id")]
            self.assertGreater(len(id_samples), 0)

    def test_override_defaults(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(100)]
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.jsonl")
            out = os.path.join(tmpdir, "output.jsonl")
            self._write_samples(samples, inp)
            # Force noise on even in grpo stage
            result = augment_dataset(
                inp, out, stage="grpo",
                noise_ratio=0.1, noise_type="grammar", noise_rate=0.05,
                seed=42,
            )
            aug_ids = [s for s in result if s["id"].endswith("_aug")]
            self.assertGreater(len(aug_ids), 0)

    def test_output_file_written(self):
        samples = [_make_sample(sid=f"s{i}") for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.jsonl")
            out = os.path.join(tmpdir, "output.jsonl")
            self._write_samples(samples, inp)
            augment_dataset(inp, out, stage="sft", seed=42)
            self.assertTrue(os.path.exists(out))
            output_samples = self._read_samples(out)
            self.assertGreater(len(output_samples), 0)


if __name__ == "__main__":
    unittest.main()
