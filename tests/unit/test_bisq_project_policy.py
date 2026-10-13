"""Replay public Bisq defects against optional, source-bound profile policies."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import pytest

from localize.semantic_quality import TranslationChange, evaluate_semantic_rules
from localize.translation_quality_gate import (
    SourceIdenticalStats,
    analyze_semantic_qa_changes,
    build_quality_gate_report,
    load_quality_gate_config,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/project_policy/bisq_review_regressions.json"
RECORDS = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
RULE_IDS = {record["rule_id"] for record in RECORDS}


def profile_config(profile):
    return load_quality_gate_config(str(ROOT / "profiles" / profile / "config.yaml"))


def change_for(record, target=None):
    return TranslationChange(
        file=f"messages_{record['locale']}.properties",
        locale_code=record["locale"],
        key=record["key"],
        source_value=record["source"]["value"],
        old_value=None,
        new_value=record["original"]["value"] if target is None else target,
    )


def policy_findings(change, profile):
    rules = profile_config(profile)[3]
    return [finding.rule_id for finding in evaluate_semantic_rules([change], rules)
            if finding.rule_id in RULE_IDS]


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_known_reviewed_defect_is_rejected(record):
    assert policy_findings(change_for(record), record["profile"]) == [record["rule_id"]]


CONTROL_CASES = [
    (record, target["value"])
    for record in RECORDS
    for target in record["corrections"] + record["valid_alternatives"]
]


@pytest.mark.parametrize("record,target", CONTROL_CASES,
                         ids=[f"{record['id']}-{i}" for i, (record, _) in enumerate(CONTROL_CASES)])
def test_corrections_and_valid_alternatives_pass(record, target):
    assert policy_findings(change_for(record, target), record["profile"]) == []


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
@pytest.mark.parametrize("boundary", ["changed-source", "unrelated-key", "other-locale", "other-profile", "missing-source"])
def test_exact_policy_authority_does_not_spread(record, boundary):
    change = change_for(record)
    profile = record["profile"]
    if boundary == "changed-source":
        change = replace(change, source_value=change.source_value + " New behavior.")
    elif boundary == "unrelated-key":
        change = replace(change, key="unrelated." + change.key)
    elif boundary == "other-locale":
        change = replace(change, locale_code="de")
    elif boundary == "other-profile":
        profile = "bisq" if profile == "bisq-mobile" else "bisq-mobile"
    else:
        change = replace(change, source_value=None)
    assert policy_findings(change, profile) == []


def test_swedish_negated_unlock_is_outside_reviewed_source_contract():
    change = TranslationChange("messages_sv.properties", "sv",
                               "user.bondedRoles.registration.how.info",
                               "Do not unlock the bond.", None,
                               "Lås inte upp obligationen.")
    assert policy_findings(change, "bisq") == []


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_original_and_correction_reach_real_profile_quality_gate(tmp_path, record):
    resources = tmp_path / "resources"
    resources.mkdir()
    source_file = resources / "messages.properties"
    source_file.write_text(f"{record['key']}={record['source']['value']}\n", encoding="utf-8")
    target_file = resources / f"messages_{record['locale']}.properties"
    config, _, brands, rules = profile_config(record["profile"])
    controls = record["corrections"] or record["valid_alternatives"]
    for target, expected_block in [(record["original"]["value"], True),
                                   (controls[0]["value"], False)]:
        target_file.write_text(f"{record['key']}={target}\n", encoding="utf-8")
        diff = (f"diff --git a/resources/{target_file.name} b/resources/{target_file.name}\n"
                f"+++ b/resources/{target_file.name}\n@@ -0,0 +1 @@\n+{record['key']}={target}\n")
        stats = analyze_semantic_qa_changes(
            diff_text=diff, repo_root=str(tmp_path), input_folder=str(resources),
            locale_codes=[record["locale"]], semantic_rules=rules, brand_glossary=brands,
            retained_source_word_allowlist=config.retained_source_word_allowlist,
            ignore_key_patterns=config.ignore_key_patterns,
        )
        report = build_quality_gate_report(
            SourceIdenticalStats(), stats, {"files": {}, "pipeline_warnings": []},
            [f"resources/{target_file.name}"], str(resources), config,
        )
        assert report["blocking"] is expected_block, report
        assert {item["rule_id"] for item in stats.examples if item["rule_id"] in RULE_IDS} == (
            {record["rule_id"]} if expected_block else set()
        )


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_fixture_evidence_is_pinned_and_rules_are_exact(record):
    for evidence in [record["source"], record["original"], *record["corrections"]]:
        assert re.fullmatch(r"[a-f0-9]{40}", evidence["commit"])
        assert evidence["url"] == (
            f"https://github.com/{evidence['repository']}/blob/"
            f"{evidence['commit']}/{evidence['path']}"
        )
        assert hashlib.sha256(evidence["value"].encode()).hexdigest() == evidence["value_sha256"]
    rules = profile_config(record["profile"])[3]
    rule = next(rule for rule in rules if rule.id == record["rule_id"])
    assert rule.keys == (record["key"],)
    assert rule.locales == (record["locale"],)
    assert rule.source_regex.pattern.startswith(r"\A")
    assert rule.source_regex.pattern.endswith(r"\Z")
    assert rule.forbidden_target_regex.pattern.startswith(r"\A")
    assert rule.forbidden_target_regex.pattern.endswith(r"\Z")
    assert rule.required_target_regex is None
    assert rule.source_regex.search(record["source"]["value"])
    assert rule.forbidden_target_regex.search(record["original"]["value"])


@pytest.mark.parametrize("profile", ["bisq", "bisq-mobile"])
@pytest.mark.parametrize("enable_fixture_locales", [False, True])
def test_cli_respects_selected_profile_locale_scope(tmp_path, profile, enable_fixture_locales):
    """A rule does not implicitly enable translation for an unselected locale."""
    import subprocess
    import yaml
    from localize.translation_quality_gate import main

    records = [record for record in RECORDS if record["profile"] == profile]
    config_path = ROOT / "profiles" / profile / "config.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if enable_fixture_locales:
        present = {locale["code"] for locale in raw_config["supported_locales"]}
        for locale in sorted({record["locale"] for record in records} - present):
            raw_config["supported_locales"].append({"code": locale, "name": locale})
        config_path = tmp_path / "explicit-test-locales.yaml"
        config_path.write_text(yaml.safe_dump(raw_config, allow_unicode=True), encoding="utf-8")
    selected = {locale["code"] for locale in raw_config["supported_locales"]}
    resources = tmp_path / "resources"
    resources.mkdir()
    sources = {record["key"]: record["source"]["value"] for record in records}
    (resources / "messages.properties").write_text(
        "\n".join(f"{key}={value}" for key, value in sources.items()) + "\n", encoding="utf-8")
    targets = {}
    for record in records:
        targets.setdefault(record["locale"], {})[record["key"]] = record["original"]["value"]
    for locale, values in targets.items():
        (resources / f"messages_{locale}.properties").write_text(
            "\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "resources"], check=True, capture_output=True)
    report = tmp_path / "report.json"
    result = main([
        "--repo-root", str(tmp_path), "--input-folder", str(resources),
        "--config", str(config_path), "--validation-summary", str(tmp_path / "missing.json"),
        "--output-json", str(report), "--output-markdown", str(tmp_path / "report.md"),
        "--changed-files", *(f"resources/messages_{locale}.properties" for locale in targets),
    ])
    data = json.loads(report.read_text(encoding="utf-8"))
    expected = {record["rule_id"] for record in records if record["locale"] in selected}
    assert result == 1
    assert data["semantic_qa"]["errors_count"] == len(expected)
    assert {finding["rule_id"] for finding in data["semantic_qa"]["examples"]
            if finding["severity"] == "error"} == expected
