"""
Konstanta role kanonik & key triple KG.

Tujuan:
- Hilangkan magic string ("agen", "pasien", "subject", ...) yang tersebar.
- Salah ketik (mis. `bindings["agne"]`) ketangkap LSP, bukan jadi None silent.

Nilai string sengaja sama dengan sebelumnya supaya YAML dataset tidak perlu diubah.
"""
from typing import Final


class Role:
    """Role argumen pada satu observation/binding."""
    AGEN: Final[str] = "agen"
    PASIEN: Final[str] = "pasien"
    PREDIKAT: Final[str] = "predikat"
    LOKASI: Final[str] = "lokasi"
    WAKTU: Final[str] = "waktu"
    INSTRUMEN: Final[str] = "instrumen"
    ATRIBUT: Final[str] = "atribut"
    ARAH: Final[str] = "arah"
    SUMBER: Final[str] = "sumber"
    MANNER: Final[str] = "manner"
    # Alias generik (dipakai oleh dataset berbahasa Inggris)
    SUBJECT: Final[str] = "subject"
    PREDICATE: Final[str] = "predicate"
    OBJECT: Final[str] = "object"


ALL_ROLES: Final[frozenset[str]] = frozenset({
    Role.AGEN, Role.PASIEN, Role.PREDIKAT, Role.LOKASI, Role.WAKTU,
    Role.INSTRUMEN, Role.ATRIBUT, Role.ARAH, Role.SUMBER, Role.MANNER,
    Role.SUBJECT, Role.PREDICATE, Role.OBJECT,
})

# Pasangan alias: kalau salah satu kosong, fallback ke pasangannya.
ROLE_ALIASES: Final[dict[str, str]] = {
    Role.AGEN: Role.SUBJECT,
    Role.PREDIKAT: Role.PREDICATE,
    Role.PASIEN: Role.OBJECT,
    Role.SUBJECT: Role.AGEN,
    Role.PREDICATE: Role.PREDIKAT,
    Role.OBJECT: Role.PASIEN,
}


def get_binding(bindings: dict, role: str) -> str | None:
    """Ambil binding[role], fallback ke alias bila kosong."""
    val = bindings.get(role)
    if val is not None:
        return val
    alias = ROLE_ALIASES.get(role)
    return bindings.get(alias) if alias else None


class TripleKey:
    """Key untuk dict triple KG (KGTriple yang sudah di-encode)."""
    SUBJECT: Final[str] = "subject"
    PREDICATE: Final[str] = "predicate"
    OBJECT: Final[str] = "object"
    METADATA: Final[str] = "metadata"


# Mapping kata-tanya Indonesia/Inggris -> role target.
# Dipakai parser query untuk menentukan slot apa yang ditanyakan.
QUESTION_TO_ROLE: Final[dict[str, str]] = {
    "siapa": Role.AGEN,
    "siapakah": Role.AGEN,
    "who": Role.AGEN,
    "apa": Role.PASIEN,
    "apakah": Role.PASIEN,
    "what": Role.PASIEN,
    "di mana": Role.LOKASI,
    "dimana": Role.LOKASI,
    "dimanakah": Role.LOKASI,
    "where": Role.LOKASI,
    "kapan": Role.WAKTU,
    "when": Role.WAKTU,
    "mengapa": Role.SUMBER,
    "kenapa": Role.SUMBER,
    "why": Role.SUMBER,
    "bagaimana": Role.ATRIBUT,
    "how": Role.ATRIBUT,
    "dengan apa": Role.INSTRUMEN,
}
