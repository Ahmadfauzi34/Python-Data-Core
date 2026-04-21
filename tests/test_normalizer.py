"""Unit test untuk text_normalizer (parser query Indonesia)."""
from __future__ import annotations
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "fhrr_project"))

from interface.text_normalizer import (  # noqa: E402
    normalize, strip_clitic, tokenize, verb_variants,
    build_variant_index, find_mentions, find_question_prefix,
)


class StripCliticTests(unittest.TestCase):
    def test_strip_kah(self):
        self.assertEqual(strip_clitic("siapakah"), "siapa")
        self.assertEqual(strip_clitic("apakah"), "apa")
        self.assertEqual(strip_clitic("kapankah"), "kapan")

    def test_strip_lah_pun(self):
        self.assertEqual(strip_clitic("itulah"), "itu")
        self.assertEqual(strip_clitic("walaupun"), "walau")

    def test_no_strip_when_too_short(self):
        # "kah" sendiri terlalu pendek, jangan jadi "" / aneh
        self.assertEqual(strip_clitic("kah"), "kah")
        self.assertEqual(strip_clitic("ah"), "ah")

    def test_idempotent(self):
        self.assertEqual(strip_clitic(strip_clitic("siapakah")), "siapa")


class NormalizeTests(unittest.TestCase):
    def test_lowercase_and_punct(self):
        self.assertEqual(normalize("Siapa makan mangga?"), "siapa makan mangga")

    def test_multispace_collapsed(self):
        self.assertEqual(normalize("siapa   yang  makan"), "siapa yang makan")

    def test_clitic_per_word(self):
        self.assertEqual(normalize("Siapakah yang memakan?"), "siapa yang memakan")


class VerbVariantTests(unittest.TestCase):
    def test_basic_prefixes(self):
        v = verb_variants("makan")
        self.assertIn("makan", v)
        self.assertIn("memakan", v)
        self.assertIn("dimakan", v)
        self.assertIn("termakan", v)

    def test_nasal_mutation(self):
        # tulis -> menulis (t luluh)
        self.assertIn("menulis", verb_variants("tulis"))
        # pukul -> memukul
        self.assertIn("memukul", verb_variants("pukul"))
        # kirim -> mengirim
        self.assertIn("mengirim", verb_variants("kirim"))


class VariantIndexTests(unittest.TestCase):
    def test_resolves_variant_to_stem(self):
        idx = build_variant_index(["makan", "minum", "lari"])
        self.assertEqual(idx["memakan"], "makan")
        self.assertEqual(idx["dimakan"], "makan")
        self.assertEqual(idx["berlari"], "lari")

    def test_shorter_stem_wins_on_collision(self):
        # Stem "makan" dan "akan" tidak bertabrakan tapi kalau ada
        # dua stem yang produce varian sama, yang pendek menang.
        idx = build_variant_index(["main", "bermain"])  # "bermain" muncul dari "main"
        self.assertEqual(idx["bermain"], "main")


class FindMentionsTests(unittest.TestCase):
    def test_word_boundary_no_substring(self):
        idx = build_variant_index(["ani", "mangga"])
        # Token "ani" TIDAK boleh cocok di kata "mani"
        self.assertEqual(find_mentions("mani makan mangga", idx), ["mangga"])

    def test_finds_verb_variant(self):
        idx = build_variant_index(["ani", "makan", "mangga"])
        self.assertEqual(find_mentions("ani memakan mangga", idx), ["ani", "makan", "mangga"])

    def test_dedup_preserves_order(self):
        idx = build_variant_index(["ani", "makan"])
        self.assertEqual(find_mentions("ani makan, ani makan lagi", idx), ["ani", "makan"])


class QuestionPrefixTests(unittest.TestCase):
    PREFIXES = ["siapa", "siapakah", "di mana", "kapan", "apa"]

    def test_at_start(self):
        self.assertEqual(find_question_prefix("Siapa makan?", self.PREFIXES), "siapa")

    def test_longest_match_wins(self):
        # "siapakah" lebih panjang dari "siapa", harus menang setelah strip clitic...
        # Catatan: setelah normalize "siapakah" jadi "siapa", jadi yang menang "siapa".
        # Test yang valid: "di mana" (multi-word) menang dari "apa"
        self.assertEqual(find_question_prefix("di mana ani?", self.PREFIXES), "di mana")

    def test_clitic_stripped(self):
        # "Kapankah" -> "kapan"
        self.assertEqual(find_question_prefix("Kapankah dia datang?", self.PREFIXES), "kapan")

    def test_keyword_anywhere_fallback(self):
        # Tidak di awal, tapi muncul sebagai kata
        self.assertEqual(find_question_prefix("Anak siapa yang makan?", self.PREFIXES), "siapa")

    def test_no_match(self):
        self.assertIsNone(find_question_prefix("xyz qwerty", self.PREFIXES))


if __name__ == "__main__":
    unittest.main()
