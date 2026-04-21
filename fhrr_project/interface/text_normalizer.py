"""
Normalisasi teks untuk parser query Indonesia.

Tugas utama:
- lowercase + strip tanda baca + collapse whitespace
- strip clitic/suffix tanya: -kah, -lah, -pun
- match token dengan word-boundary (bukan substring) -> hindari "ani" cocok di "mani"
- generate varian verba (me-, di-, ter-, ber-, + nasal mutation) dari stem di vocab
  -> "memakan", "dimakan" otomatis nge-resolve ke "makan" tanpa perlu hardcode

Tujuannya kecil & deterministik. Bukan stemmer Sastrawi penuh — kalau butuh
recall lebih tinggi, ganti `verb_variants()` jadi adapter ke library eksternal.
"""
from __future__ import annotations
import re
from typing import Iterable

# Suffix klitik tanya/penegas yang aman dilepas (jarang berubah makna inti).
_CLITIC_SUFFIXES = ("kah", "lah", "pun")

# Karakter yang dianggap pemisah kata (selain spasi).
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def strip_clitic(word: str) -> str:
    """Buang -kah/-lah/-pun di akhir kata. Idempoten."""
    for suf in _CLITIC_SUFFIXES:
        if len(word) > len(suf) + 1 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def normalize(text: str) -> str:
    """Lowercase, hapus tanda baca, lepas klitik per kata, collapse spasi."""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return " ".join(strip_clitic(w) for w in t.split())


def tokenize(text: str) -> list[str]:
    """Pisah jadi kata setelah normalize."""
    return normalize(text).split()


def verb_variants(stem: str) -> set[str]:
    """Bangkitkan varian verba Indonesia umum dari stem.

    Cakup prefix produktif: me-, di-, ter-, ber-, plus alomorfi nasal
    untuk me- (mem-, men-, meng-, meny-, menge-) termasuk peluluhan
    konsonan awal p/t/k/s.
    """
    if not stem:
        return set()
    out: set[str] = {stem}
    for p in ("me", "mem", "men", "meng", "menge", "meny", "di", "ter", "ber"):
        out.add(p + stem)

    first = stem[0]
    rest = stem[1:]
    # Peluluhan: pukul -> memukul, tulis -> menulis, kirim -> mengirim, sapu -> menyapu
    nasal_map = {"p": ("mem", rest), "t": ("men", rest), "k": ("meng", rest), "s": ("meny", rest)}
    if first in nasal_map:
        prefix, body = nasal_map[first]
        out.add(prefix + body)
        out.add("di" + stem)  # bentuk pasif tetap pakai stem utuh
    return out


def build_variant_index(stems: Iterable[str]) -> dict[str, str]:
    """Bangun mapping `varian -> stem kanonik` dari list stem (token vocab).

    Bila dua stem menghasilkan varian sama, stem yang lebih pendek menang
    (umumnya stem pendek = bentuk dasar yang lebih relevan).
    """
    index: dict[str, str] = {}
    for s in stems:
        for v in verb_variants(s):
            cur = index.get(v)
            if cur is None or len(s) < len(cur):
                index[v] = s
    return index


def find_mentions(text: str, variant_index: dict[str, str]) -> list[str]:
    """Cari stem yang disebut di teks (urutan kemunculan, tanpa duplikat).

    Match per-kata (word boundary), bukan substring.
    """
    seen: set[str] = set()
    out: list[str] = []
    for word in tokenize(text):
        stem = variant_index.get(word)
        if stem and stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def find_question_prefix(text: str, prefixes: Iterable[str]) -> str | None:
    """Cari prefix tanya. Coba di awal kalimat dulu (longest match),
    kalau tidak ada baru cari sebagai kata di mana saja."""
    norm = normalize(text)
    sorted_prefixes = sorted(prefixes, key=len, reverse=True)
    for p in sorted_prefixes:
        if norm.startswith(p):
            return p
    # Fallback: kata di mana saja (untuk "anak siapa yang makan?")
    words = norm.split()
    word_set = set(words)
    for p in sorted_prefixes:
        if " " in p:
            if p in norm:
                return p
        elif p in word_set:
            return p
    return None
