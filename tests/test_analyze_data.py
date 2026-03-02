"""Tests for the data analysis / diagnostic module."""

import json
import os
import tempfile
import unittest

from src.data.analyze_data import (
    compare_stats,
    compute_lang_stats,
    format_comparison,
    format_single_stats,
    load_jsonl,
)


def _s(sid="s1", lang="en", source="I is happy.", target="I am happy."):
    """Create a minimal sample."""
    return {"id": sid, "lang": lang, "source": source, "target": target}


class TestComputeLangStats(unittest.TestCase):

    def test_basic_counts(self):
        samples = [
            _s("s1", "en", "bad text", "good text"),
            _s("s2", "en", "another bad", "another good"),
            _s("s3", "cs", "špatný", "správný"),
        ]
        stats = compute_lang_stats(samples)
        self.assertEqual(stats["en"]["n_samples"], 2)
        self.assertEqual(stats["cs"]["n_samples"], 1)
        self.assertEqual(stats["__overall__"]["n_samples"], 3)

    def test_identity_detection(self):
        samples = [
            _s("s1", "en", "same text", "same text"),  # identity
            _s("s2", "en", "bad", "good"),              # not identity
            _s("s3_id", "en", "also same", "also same"), # identity + injected
        ]
        stats = compute_lang_stats(samples)
        self.assertEqual(stats["en"]["n_identity"], 2)
        self.assertEqual(stats["en"]["n_identity_injected"], 1)
        self.assertAlmostEqual(stats["en"]["identity_ratio"], 2 / 3)

    def test_augmented_detection(self):
        samples = [
            _s("s1", "en", "a", "b"),
            _s("s2_aug", "en", "c", "d"),
            _s("s3_aug", "en", "e", "f"),
        ]
        stats = compute_lang_stats(samples)
        self.assertEqual(stats["en"]["n_augmented"], 2)

    def test_edit_distance_metrics(self):
        # "abc" vs "axc" → edit distance = 1, max_len = 3, sim = 2/3
        samples = [_s("s1", "en", "abc", "axc")]
        stats = compute_lang_stats(samples)
        self.assertAlmostEqual(stats["en"]["avg_edit_distance"], 1.0)
        self.assertAlmostEqual(stats["en"]["avg_edit_similarity"], 2.0 / 3.0)
        self.assertAlmostEqual(stats["en"]["error_rate"], 1.0 / 3.0)

    def test_correction_density(self):
        # "abc" vs "axc" → edit_dist=1, source_len=3, density = 1/3
        samples = [_s("s1", "en", "abc", "axc")]
        stats = compute_lang_stats(samples)
        self.assertAlmostEqual(stats["en"]["avg_correction_density"], 1.0 / 3.0)

    def test_identical_texts_zero_error(self):
        samples = [_s("s1", "en", "perfect text", "perfect text")]
        stats = compute_lang_stats(samples)
        self.assertAlmostEqual(stats["en"]["error_rate"], 0.0)
        self.assertAlmostEqual(stats["en"]["avg_edit_similarity"], 1.0)

    def test_length_metrics(self):
        samples = [_s("s1", "en", "hello world", "hello beautiful world")]
        stats = compute_lang_stats(samples)
        self.assertAlmostEqual(stats["en"]["avg_source_len_char"], 11.0)
        self.assertAlmostEqual(stats["en"]["avg_target_len_char"], 21.0)
        self.assertAlmostEqual(stats["en"]["avg_source_len_word"], 2.0)
        self.assertAlmostEqual(stats["en"]["avg_target_len_word"], 3.0)

    def test_multiple_languages(self):
        samples = [
            _s("s1", "en", "bad", "good"),
            _s("s2", "en", "worse", "better"),
            _s("s3", "cs", "špatně", "dobře"),
            _s("s4", "de", "falsch", "richtig"),
        ]
        stats = compute_lang_stats(samples)
        self.assertIn("en", stats)
        self.assertIn("cs", stats)
        self.assertIn("de", stats)
        self.assertIn("__overall__", stats)
        self.assertEqual(stats["en"]["n_samples"], 2)
        self.assertEqual(stats["cs"]["n_samples"], 1)
        self.assertEqual(stats["de"]["n_samples"], 1)
        self.assertEqual(stats["__overall__"]["n_samples"], 4)

    def test_empty_input(self):
        stats = compute_lang_stats([])
        self.assertEqual(len(stats), 0)

    def test_empty_source_or_target(self):
        samples = [_s("s1", "en", "", "some target")]
        stats = compute_lang_stats(samples)
        self.assertEqual(stats["en"]["n_samples"], 1)
        self.assertAlmostEqual(stats["en"]["avg_edit_similarity"], 0.0)


class TestCompareStats(unittest.TestCase):

    def test_basic_comparison(self):
        before = compute_lang_stats([
            _s("s1", "en", "bad", "good"),
            _s("s2", "en", "worse", "better"),
        ])
        after = compute_lang_stats([
            _s("s1", "en", "bad", "good"),
            _s("s2", "en", "worse", "better"),
            _s("s3_id", "en", "good", "good"),  # identity injected
            _s("s4_aug", "en", "badd", "good"),  # augmented
        ])
        comp = compare_stats(before, after)
        self.assertIn("en", comp)
        self.assertEqual(comp["en"]["delta"]["n_samples"], 2)
        self.assertGreater(comp["en"]["after"]["n_augmented"], comp["en"]["before"]["n_augmented"])

    def test_new_language_in_after(self):
        before = compute_lang_stats([_s("s1", "en", "a", "b")])
        after = compute_lang_stats([
            _s("s1", "en", "a", "b"),
            _s("s2", "cs", "x", "y"),
        ])
        comp = compare_stats(before, after)
        self.assertIn("cs", comp)
        self.assertEqual(comp["cs"]["delta"]["n_samples"], 1)

    def test_delta_sign(self):
        before = compute_lang_stats([_s("s1", "en", "bad text", "good text")])
        # After has identity samples → identity_ratio goes up, error_rate goes down
        after = compute_lang_stats([
            _s("s1", "en", "bad text", "good text"),
            _s("s2_id", "en", "good text", "good text"),
        ])
        comp = compare_stats(before, after)
        self.assertGreater(comp["en"]["delta"]["identity_ratio"], 0)


class TestFormatSingleStats(unittest.TestCase):

    def test_produces_string(self):
        stats = compute_lang_stats([
            _s("s1", "en", "bad", "good"),
            _s("s2", "cs", "špatně", "dobře"),
        ])
        output = format_single_stats(stats)
        self.assertIsInstance(output, str)
        self.assertIn("en", output)
        self.assertIn("cs", output)
        self.assertIn("ALL", output)

    def test_header_present(self):
        stats = compute_lang_stats([_s("s1", "en", "a", "b")])
        output = format_single_stats(stats)
        self.assertIn("Lang", output)
        self.assertIn("ErrRate", output)


class TestFormatComparison(unittest.TestCase):

    def test_produces_string(self):
        before = compute_lang_stats([_s("s1", "en", "a", "b")])
        after = compute_lang_stats([_s("s1", "en", "a", "b"), _s("s2_id", "en", "b", "b")])
        comp = compare_stats(before, after)
        output = format_comparison(comp)
        self.assertIsInstance(output, str)
        self.assertIn("en", output)


class TestLoadJsonl(unittest.TestCase):

    def test_basic(self):
        samples = [_s("s1", "en", "a", "b"), _s("s2", "cs", "x", "y")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["id"], "s1")
        finally:
            os.unlink(path)

    def test_empty_lines_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_s()) + "\n")
            f.write("\n")
            f.write(json.dumps(_s("s2")) + "\n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            self.assertEqual(len(loaded), 2)
        finally:
            os.unlink(path)


class TestIntegration(unittest.TestCase):
    """End-to-end test: create data, augment, compare."""

    def test_augment_and_compare(self):
        from src.data.augmentation import augment_dataset

        samples = []
        for lang, count in [("en", 20), ("cs", 15), ("de", 10)]:
            for i in range(count):
                samples.append({
                    "id": f"{lang}_{i:03d}",
                    "lang": lang,
                    "corpus": "test",
                    "split": "train",
                    "ref": "ref1",
                    "messages": [
                        {"role": "user", "content": f"Correct:\n\nText with error {i} in {lang}."},
                        {"role": "assistant", "content": f"Text without error {i} in {lang}."},
                    ],
                    "prompt": f"Correct:\n\nText with error {i} in {lang}.",
                    "source": f"Text with error {i} in {lang}.",
                    "target": f"Text without error {i} in {lang}.",
                })

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw.jsonl")
            aug_path = os.path.join(tmpdir, "aug.jsonl")

            with open(raw_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            augment_dataset(raw_path, aug_path, stage="sft", seed=42)

            raw_samples = load_jsonl(raw_path)
            aug_samples = load_jsonl(aug_path)

            before_stats = compute_lang_stats(raw_samples)
            after_stats = compute_lang_stats(aug_samples)
            comparison = compare_stats(before_stats, after_stats)

            # After augmentation should have more samples
            self.assertGreater(
                after_stats["__overall__"]["n_samples"],
                before_stats["__overall__"]["n_samples"] * 0.5,
            )

            # Identity ratio should increase
            self.assertGreater(
                after_stats["__overall__"]["identity_ratio"],
                before_stats["__overall__"]["identity_ratio"],
            )

            # Format should work without errors
            output = format_comparison(comparison)
            self.assertIn("en", output)
            self.assertIn("cs", output)
            self.assertIn("de", output)


if __name__ == "__main__":
    unittest.main()
