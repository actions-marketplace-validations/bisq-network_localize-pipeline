"""Glossary matching must survive Java Properties newline serialization."""

import json

import pytest

from localize.formats.metadata import JSON_FORMAT
from localize.formats.properties import reassemble_file
from localize.translate_localization_files import (
    _prefer_glossary_compliant_draft,
    run_per_key_validation_with_summary,
    run_post_translation_validation,
)
from localize.formats.validation import find_glossary_mismatches


def _properties_mismatches(source, target, glossary):
    return find_glossary_mismatches(
        source, target, glossary, decode_properties_whitespace=True
    )


@pytest.mark.parametrize("separator", ["\n", "\r", "\t", r"\n", r"\r", r"\t"])
@pytest.mark.parametrize("source_term,target_term", [("trade", "거래"), ("content", "Efni")])
def test_glossary_matches_terms_after_logical_whitespace(separator, source_term, target_term):
    glossary = {source_term: target_term}
    assert _properties_mismatches(
        f"Information.{separator}{source_term}", f"안내.{separator}{target_term}", glossary
    ) == []
    assert _properties_mismatches(
        f"Information.{separator}{source_term}", "안내. 누락", glossary
    ) == [(source_term, target_term)]


@pytest.mark.parametrize("escape", ["n", "r", "t"])
@pytest.mark.parametrize("slash_count", [1, 2, 3, 4])
def test_glossary_only_decodes_unescaped_whitespace(escape, slash_count):
    separator = "\\" * slash_count + escape
    # Odd runs end in a whitespace escape; even runs preserve a literal n/r/t.
    expected = [] if slash_count % 2 else [("trade", "거래")]
    assert _properties_mismatches(
        "a trade", separator + "거래", {"trade": "거래"}
    ) == expected
    expected_source = [("trade", "거래")] if slash_count % 2 else []
    assert _properties_mismatches(
        separator + "trade", "누락", {"trade": "거래"}
    ) == expected_source


def test_escaped_whitespace_keeps_term_counts_and_word_boundaries():
    assert _properties_mismatches(
        r"trade\ntrade", r"거래\n거래", {"trade": "거래"}
    ) == []
    assert _properties_mismatches(
        r"trade\ntrade", r"거래\n누락", {"trade": "거래"}
    ) == [("trade", "거래")]
    for target in [r"안내\n거래소", r"안내\n거래나", r"안내\nX거래", "누락"]:
        assert _properties_mismatches(
            "a trade", target, {"trade": "거래"}
        ) == [("trade", "거래")]
    assert _properties_mismatches(
        r"notrade\ntrades", "누락", {"trade": "거래"}
    ) == []


@pytest.mark.parametrize("escape,actual", [(r"\n", "\n"), (r"\r", "\r"), (r"\t", "\t")])
def test_escaped_whitespace_keeps_exact_multiword_phrase_matching(escape, actual):
    glossary = {"private chat": "비공개 채팅"}
    assert _properties_mismatches(
        "private" + escape + "chat", "누락", glossary
    ) == _properties_mismatches("private" + actual + "chat", "누락", glossary) == []
    assert _properties_mismatches(
        "private chat", "비공개" + escape + "채팅", glossary
    ) == [("private chat", "비공개 채팅")]


@pytest.mark.parametrize("source_separator", ["\n", r"\n"])
@pytest.mark.parametrize("source_body,candidate,glossary", [
    (
        "If a trade is missing, contact Bisq.",
        "저장된 데이터.\n거래 누락 시 Bisq에 문의하세요.",
        {"trade": "거래"},
    ),
    ("Content is unverified.", "Upplýsingar.\nEfni er óstaðfest.", {"content": "Efni"}),
])
def test_valid_translation_survives_properties_roundtrip(
    source_separator, source_body, candidate, glossary
):
    source = {"key": "Stored data." + source_separator + source_body}
    valid, summary = run_per_key_validation_with_summary(
        {"key": candidate}, source, "synthetic_ko.properties",
        translation_glossary=glossary, selected_keys={"key"},
        decode_properties_whitespace=True,
    )
    assert summary["failed_keys"] == []
    content = reassemble_file([{"type": "entry", "key": "key", "value": valid["key"]}])
    assert "\\n" + next(iter(glossary.values())) in content
    assert run_post_translation_validation(
        content, source, "synthetic_ko.properties", changed_keys_for_run={"key"},
        translation_glossary=glossary,
    )


def test_post_validation_still_rejects_missing_glossary_term_after_serialization():
    source = {"key": r"Stored data.\nIf a trade is missing, contact Bisq."}
    content = reassemble_file([
        {"type": "entry", "key": "key", "value": "저장된 데이터.\n매매 누락 시 Bisq에 문의하세요."}
    ])
    assert not run_post_translation_validation(
        content, source, "synthetic_ko.properties", changed_keys_for_run={"key"},
        translation_glossary={"trade": "거래"},
    )


@pytest.mark.parametrize("source,target", [
    ("a trade", r"C:\nHandel"),
    (r"C:\trade", "Ausführung"),
])
def test_decoded_json_literal_paths_keep_glossary_word_boundaries(source, target):
    glossary = {"trade": "Handel"}
    assert find_glossary_mismatches(source, target, glossary) == [("trade", "Handel")]
    _, summary = run_per_key_validation_with_summary(
        {"key": target}, {"key": source}, "synthetic_de.json",
        translation_glossary=glossary, selected_keys={"key"},
    )
    assert summary["glossary_mismatch_keys"] == ["key"]
    assert not run_post_translation_validation(
        json.dumps({"key": target}), {"/key": source}, "synthetic_de.json",
        localization_format=JSON_FORMAT, changed_keys_for_run={"/key"},
        translation_glossary=glossary,
    )


def test_properties_draft_preference_uses_same_whitespace_representation():
    glossary = {"trade": "거래"}
    draft = "안내.\n거래"
    reviewed = "안내. 매매"
    assert _prefer_glossary_compliant_draft(
        "a trade", draft, reviewed, glossary, decode_properties_whitespace=True
    ) == draft
    assert _prefer_glossary_compliant_draft(
        "a trade", draft, r"안내.\n거래", glossary, decode_properties_whitespace=True
    ) == r"안내.\n거래"
