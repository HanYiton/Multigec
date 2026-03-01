"""Tests for the grammar correction reward function."""

import json
import math
import os
import tempfile
import unittest

from src.rewards.grammar_reward import (
    FORMAT_FAIL_SCORE,
    char_edit_distance,
    chrf_score,
    compute_score,
    compute_weights,
    edit_distance_similarity,
    format_gate,
    normalize_text,
    write_progress,
    _read_progress,
    _progress_cache,
    _sigmoid,
    PROGRESS_FILE,
)


class TestNormalizeText(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(normalize_text("hello  world"), "hello world")

    def test_unicode_normalization(self):
        # NFC normalization: e + combining acute → é
        decomposed = "e\u0301"
        composed = "\u00e9"
        self.assertEqual(normalize_text(decomposed), normalize_text(composed))

    def test_strip(self):
        self.assertEqual(normalize_text("  hello  "), "hello")

    def test_tabs_and_newlines(self):
        self.assertEqual(normalize_text("hello\t\nworld"), "hello world")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")


class TestCharEditDistance(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(char_edit_distance("abc", "abc"), 0)

    def test_one_insert(self):
        self.assertEqual(char_edit_distance("abc", "abcd"), 1)

    def test_one_delete(self):
        self.assertEqual(char_edit_distance("abcd", "abc"), 1)

    def test_one_replace(self):
        self.assertEqual(char_edit_distance("abc", "axc"), 1)

    def test_empty(self):
        self.assertEqual(char_edit_distance("", "abc"), 3)
        self.assertEqual(char_edit_distance("abc", ""), 3)

    def test_both_empty(self):
        self.assertEqual(char_edit_distance("", ""), 0)

    def test_symmetric(self):
        self.assertEqual(
            char_edit_distance("kitten", "sitting"),
            char_edit_distance("sitting", "kitten"),
        )

    def test_known_value(self):
        # kitten -> sitting: k→s, e→i, +g = 3
        self.assertEqual(char_edit_distance("kitten", "sitting"), 3)


class TestEditDistanceSimilarity(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(edit_distance_similarity("abc", "abc"), 1.0)

    def test_completely_different(self):
        self.assertAlmostEqual(edit_distance_similarity("aaa", "bbb"), 0.0)

    def test_empty_both(self):
        self.assertAlmostEqual(edit_distance_similarity("", ""), 1.0)

    def test_range(self):
        score = edit_distance_similarity("hello", "hallo")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_partial(self):
        # "abc" vs "axc" → dist=1, max_len=3, sim=2/3
        self.assertAlmostEqual(edit_distance_similarity("abc", "axc"), 2.0 / 3.0)


class TestChrfScore(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(chrf_score("hello world", "hello world"), 1.0)

    def test_empty_both(self):
        self.assertAlmostEqual(chrf_score("", ""), 1.0)

    def test_empty_pred(self):
        self.assertAlmostEqual(chrf_score("", "hello"), 0.0)

    def test_empty_ref(self):
        self.assertAlmostEqual(chrf_score("hello", ""), 0.0)

    def test_partial_overlap(self):
        score = chrf_score("hello world", "hello earth")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_overlap(self):
        score = chrf_score("aaa", "bbb")
        self.assertAlmostEqual(score, 0.0)

    def test_higher_for_more_similar(self):
        s1 = chrf_score("I am happy", "I am happy today")
        s2 = chrf_score("I am happy", "xyz completely different")
        self.assertGreater(s1, s2)

    def test_beta_effect(self):
        # Higher beta weighs recall more
        pred = "hello"
        ref = "hello world"
        score_b1 = chrf_score(pred, ref, beta=1.0)
        score_b2 = chrf_score(pred, ref, beta=2.0)
        # With higher recall weight and pred being a subset, beta=2 should differ
        self.assertNotAlmostEqual(score_b1, score_b2)

    def test_morphological_robustness(self):
        # chrF should give partial credit for morphological variants
        score = chrf_score("бежал", "бежала")  # Russian: ran (m) vs ran (f)
        self.assertGreater(score, 0.5)


class TestFormatGate(unittest.TestCase):

    def test_empty_fails(self):
        self.assertFalse(format_gate("", "some source"))

    def test_whitespace_only_fails(self):
        self.assertFalse(format_gate("   \t\n  ", "some source"))

    def test_normal_passes(self):
        self.assertTrue(format_gate("corrected text", "original text"))

    def test_too_short_fails(self):
        # ratio = 1/100 = 0.01 < 0.3
        self.assertFalse(format_gate("x", "x" * 100))

    def test_too_long_fails(self):
        # ratio = 400/100 = 4.0 > 3.0
        self.assertFalse(format_gate("x" * 400, "x" * 100))

    def test_reasonable_length_passes(self):
        self.assertTrue(format_gate("x" * 80, "x" * 100))

    def test_no_source_skips_length_check(self):
        self.assertTrue(format_gate("hello", ""))

    def test_instruction_leakage_fails(self):
        leaked = "Please correct all grammatical errors in the following Czech text: ..."
        self.assertFalse(format_gate(leaked, "some source"))

    def test_instruction_leakage_case_insensitive(self):
        leaked = "CORRECT THE GRAMMATICAL ERRORS IN THE FOLLOWING text stuff"
        self.assertFalse(format_gate(leaked, "source"))

    def test_normal_text_with_correct_word_passes(self):
        # "correct" in normal context should NOT trigger the gate
        self.assertTrue(format_gate("The answer is correct.", "The answer is correkt."))


class TestSigmoid(unittest.TestCase):

    def test_zero(self):
        self.assertAlmostEqual(_sigmoid(0.0), 0.5)

    def test_large_positive(self):
        self.assertAlmostEqual(_sigmoid(100.0), 1.0, places=5)

    def test_large_negative(self):
        self.assertAlmostEqual(_sigmoid(-100.0), 0.0, places=5)

    def test_symmetry(self):
        self.assertAlmostEqual(_sigmoid(2.0) + _sigmoid(-2.0), 1.0)


class TestComputeWeights(unittest.TestCase):

    def test_sum_to_one(self):
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            w_p, w_c = compute_weights(p)
            self.assertAlmostEqual(w_p + w_c, 1.0, places=10)

    def test_early_training(self):
        w_p, w_c = compute_weights(0.0)
        # Early: preserve ≈ 0.5, correct ≈ 0.5
        self.assertGreater(w_p, 0.45)
        self.assertAlmostEqual(w_p, w_c, places=1)

    def test_late_training(self):
        w_p, w_c = compute_weights(1.0)
        # Late: preserve ≈ 0.25, correct ≈ 0.75
        self.assertLess(w_p, 0.30)
        self.assertGreater(w_c, 0.70)

    def test_monotonic(self):
        # w_correct should increase monotonically with progress
        prev_wc = 0.0
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            _, w_c = compute_weights(p)
            self.assertGreaterEqual(w_c, prev_wc)
            prev_wc = w_c


class TestComputeScore(unittest.TestCase):

    def _score(self, pred, ref, source=None, data_source="multigec_en"):
        extra = {"source": source} if source else None
        return compute_score(data_source, pred, ref, extra)

    def test_empty_output(self):
        score = self._score("", "corrected text", source="original text")
        self.assertEqual(score, FORMAT_FAIL_SCORE)

    def test_exact_match(self):
        score = self._score("I am happy.", "I am happy.", source="I is happy.")
        self.assertEqual(score, 1.0)

    def test_good_correction(self):
        source = "I is happy."
        pred = "I am happy."
        ref = "I am happy."
        score = self._score(pred, ref, source=source)
        self.assertEqual(score, 1.0)

    def test_partial_correction(self):
        source = "I is happy and she are sad."
        pred = "I am happy and she are sad."  # fixed one error
        ref = "I am happy and she is sad."   # both fixed
        score = self._score(pred, ref, source=source)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 1.0)

    def test_copy_gets_lower_than_correction(self):
        source = "I is happy."
        ref = "I am happy."
        # Copy unchanged
        score_copy = self._score(source, ref, source=source)
        # Correct it
        score_correct = self._score(ref, ref, source=source)
        self.assertGreater(score_correct, score_copy)

    def test_garbage_output(self):
        source = "Hello world."
        pred = "x"  # too short → format gate fails
        ref = "Hello world."
        score = self._score(pred, ref, source=source)
        self.assertEqual(score, FORMAT_FAIL_SCORE)

    def test_complete_rewrite(self):
        source = "I is happy."
        pred = "The sun shines brightly on the meadow."
        ref = "I am happy."
        score = self._score(pred, ref, source=source)
        # Should get low R_correct (different from ref) AND low R_preserve (different from source)
        self.assertLess(score, 0.5)

    def test_no_source_fallback(self):
        # Without source, R_preserve defaults to 0.5
        score = self._score("I am happy.", "I am happy and glad.")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_instruction_leakage(self):
        source = "I is happy."
        pred = "Please correct all grammatical errors in the following text: I am happy."
        ref = "I am happy."
        score = self._score(pred, ref, source=source)
        self.assertEqual(score, FORMAT_FAIL_SCORE)

    def test_multilingual_czech(self):
        source = "Já jsem šťastná."
        pred = "Já jsem šťastná."  # identity (already correct)
        ref = "Já jsem šťastná."
        score = self._score(pred, ref, source=source, data_source="multigec_cs")
        self.assertEqual(score, 1.0)

    def test_multilingual_russian(self):
        source = "Я есть счастливый."
        pred = "Я счастлив."
        ref = "Я счастлив."
        score = self._score(pred, ref, source=source, data_source="multigec_ru")
        self.assertEqual(score, 1.0)

    def test_score_range(self):
        # Scores should be in [-1, 1]
        test_cases = [
            ("hello", "hello", "hello"),
            ("abc", "xyz", "abc"),
            ("partial match here", "partial fix here", "partial match here"),
        ]
        for pred, ref, source in test_cases:
            score = self._score(pred, ref, source=source)
            self.assertGreaterEqual(score, -1.0, f"pred={pred}, ref={ref}")
            self.assertLessEqual(score, 1.0, f"pred={pred}, ref={ref}")


class TestWriteProgress(unittest.TestCase):

    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            write_progress(50, 100, path=tmp_path)
            with open(tmp_path, "r") as f:
                data = json.load(f)
            self.assertEqual(data["step"], 50)
            self.assertEqual(data["total_steps"], 100)
        finally:
            os.unlink(tmp_path)

    def test_atomic_write(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            # Write initial
            write_progress(10, 100, path=tmp_path)
            # Overwrite
            write_progress(20, 100, path=tmp_path)
            with open(tmp_path, "r") as f:
                data = json.load(f)
            self.assertEqual(data["step"], 20)
        finally:
            os.unlink(tmp_path)


class TestProgressCache(unittest.TestCase):

    def test_default_progress_zero(self):
        # Without a progress file, should return 0.0
        _progress_cache["value"] = 0.0
        _progress_cache["call_count"] = 0
        # First call doesn't trigger a read (call_count=1, not multiple of 100)
        p = _read_progress()
        self.assertEqual(p, 0.0)


class TestRewardIntegration(unittest.TestCase):
    """End-to-end tests simulating realistic GEC scenarios."""

    def test_identity_correct_text(self):
        """When source is already correct, copying it should score well."""
        text = "The cat sat on the mat."
        score = compute_score(
            "multigec_en", text, text,
            {"source": text, "lang": "en", "corpus": "test"},
        )
        self.assertEqual(score, 1.0)

    def test_quality_ordering(self):
        """Better corrections should get higher scores."""
        source = "She dont like apples and she dont like oranges."
        ref = "She doesn't like apples and she doesn't like oranges."

        # Perfect correction
        s_perfect = compute_score("multigec_en", ref, ref, {"source": source})

        # Partial correction (one fix)
        partial = "She doesn't like apples and she dont like oranges."
        s_partial = compute_score("multigec_en", partial, ref, {"source": source})

        # Copy (no fix)
        s_copy = compute_score("multigec_en", source, ref, {"source": source})

        self.assertGreater(s_perfect, s_partial)
        self.assertGreater(s_partial, s_copy)

    def test_overcorrection_penalized(self):
        """Rewriting the text entirely should score lower than minimal correction."""
        source = "I is happy."
        ref = "I am happy."
        rewrite = "Happiness fills my entire being today."

        s_correct = compute_score("multigec_en", ref, ref, {"source": source})
        s_rewrite = compute_score("multigec_en", rewrite, ref, {"source": source})

        self.assertGreater(s_correct, s_rewrite)


if __name__ == "__main__":
    unittest.main()
