"""
Schema dataset FHRR — dataclasses + validator.

Tujuan:
- Tangkap salah ketik field & dangling reference di boot, bukan saat query.
- Dokumentasi struktur dataset yang machine-checkable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


from fhrr_project.core.roles import ALL_ROLES as KNOWN_ROLES  # noqa: E402,F401


@dataclass
class DatasetIssue:
    severity: str  # 'error' | 'warning'
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.where}: {self.message}"


def validate_dataset(ds: dict) -> list[DatasetIssue]:
    """Validasi dataset. Return list issue (kosong = clean).

    Cek:
      - Struktur top-level (vocab.categories, observations, ...).
      - Setiap observation punya 'id' & 'bindings' dict.
      - Tidak ada ID duplikat antar observations / qa_pairs / dst.
      - qa_pairs.source merujuk ke observation yang ada.
      - comprehension_tasks.context merujuk observation yang ada.
      - comprehension_tasks.inference_rule merujuk reasoning_pattern yang ada.
      - teaching_episodes.verify_by merujuk qa atau logical_pair yang ada.
      - Token di bindings yang tidak ada di vocab → warning (open-vocab handle nanti).
      - Role di bindings yang tidak dikenal → warning.
    """
    issues: list[DatasetIssue] = []

    # ---- struktur dasar ----
    if not isinstance(ds, dict):
        issues.append(DatasetIssue("error", "<root>", "dataset harus berupa dict"))
        return issues

    vocab = ds.get("vocab", {})
    categories = vocab.get("categories", {})
    if not categories:
        issues.append(DatasetIssue("warning", "vocab.categories", "kosong — engine akan minim token"))

    all_tokens: set[str] = set()
    for cat, tokens in categories.items():
        if not isinstance(tokens, list):
            issues.append(DatasetIssue("error", f"vocab.categories.{cat}", "harus berupa list"))
            continue
        all_tokens.update(tokens)

    # ---- observations ----
    observations = ds.get("observations", [])
    obs_ids: set[str] = set()
    for i, obs in enumerate(observations):
        loc = f"observations[{i}]"
        if "id" not in obs:
            issues.append(DatasetIssue("error", loc, "wajib punya 'id'"))
            continue
        if obs["id"] in obs_ids:
            issues.append(DatasetIssue("error", loc, f"id duplikat: {obs['id']}"))
        obs_ids.add(obs["id"])

        bindings = obs.get("bindings")
        if not isinstance(bindings, dict):
            issues.append(DatasetIssue("error", f"{loc}.bindings", "wajib dict"))
            continue
        for role, token in bindings.items():
            if role not in KNOWN_ROLES:
                issues.append(DatasetIssue("warning", f"{loc}.bindings.{role}", f"role tidak dikenal (token={token!r})"))
            if isinstance(token, str) and token not in all_tokens:
                issues.append(DatasetIssue("warning", f"{loc}.bindings.{role}", f"token {token!r} tidak ada di vocab"))

    # ---- qa_pairs ----
    qa_ids: set[str] = set()
    for i, qa in enumerate(ds.get("qa_pairs", [])):
        loc = f"qa_pairs[{i}]"
        if "id" not in qa:
            issues.append(DatasetIssue("error", loc, "wajib punya 'id'"))
            continue
        if qa["id"] in qa_ids:
            issues.append(DatasetIssue("error", loc, f"id duplikat: {qa['id']}"))
        qa_ids.add(qa["id"])

        src = qa.get("source")
        if src and src not in obs_ids:
            issues.append(DatasetIssue("error", f"{loc}.source", f"merujuk observation tidak ada: {src!r}"))

        ans_role = qa.get("answer_role")
        if ans_role and ans_role not in KNOWN_ROLES:
            issues.append(DatasetIssue("warning", f"{loc}.answer_role", f"role tidak dikenal: {ans_role!r}"))

    # ---- reasoning_patterns ----
    rule_ids: set[str] = set()
    rule_names: set[str] = set()
    for i, r in enumerate(ds.get("reasoning_patterns", [])):
        loc = f"reasoning_patterns[{i}]"
        if "id" in r:
            if r["id"] in rule_ids:
                issues.append(DatasetIssue("error", loc, f"id duplikat: {r['id']}"))
            rule_ids.add(r["id"])
        if "name" in r:
            rule_names.add(r["name"])

    # ---- comprehension_tasks ----
    for i, c in enumerate(ds.get("comprehension_tasks", [])):
        loc = f"comprehension_tasks[{i}]"
        ctx = c.get("context")
        ctx_list = ctx if isinstance(ctx, list) else [ctx] if ctx else []
        for cid in ctx_list:
            if cid not in obs_ids:
                issues.append(DatasetIssue("error", f"{loc}.context", f"merujuk observation tidak ada: {cid!r}"))
        rule = c.get("inference_rule")
        if rule and rule not in rule_ids and rule not in rule_names:
            issues.append(DatasetIssue("error", f"{loc}.inference_rule", f"merujuk rule tidak ada: {rule!r}"))

    # ---- logical_pairs ----
    lp_ids: set[str] = set()
    for i, lp in enumerate(ds.get("logical_pairs", [])):
        loc = f"logical_pairs[{i}]"
        if "id" in lp:
            if lp["id"] in lp_ids:
                issues.append(DatasetIssue("error", loc, f"id duplikat: {lp['id']}"))
            lp_ids.add(lp["id"])

    # ---- teaching_episodes ----
    for i, te in enumerate(ds.get("teaching_episodes", [])):
        loc = f"teaching_episodes[{i}]"
        v = te.get("verify_by")
        if v and v not in qa_ids and v not in lp_ids:
            issues.append(DatasetIssue("warning", f"{loc}.verify_by", f"merujuk id tidak ada: {v!r}"))

    return issues


def assert_valid(ds: dict, *, strict: bool = True) -> list[DatasetIssue]:
    """Validasi & raise ValueError kalau ada error.

    strict=True: error apapun -> raise. Warning hanya di-print.
    strict=False: print semua, jangan raise.
    """
    issues = validate_dataset(ds)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if strict and errors:
        raise ValueError(f"Dataset tidak valid: {len(errors)} error, {len(warnings)} warning")
    return issues
