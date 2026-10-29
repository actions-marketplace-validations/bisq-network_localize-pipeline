"""Project policy is private data, never pipeline code or publication authority."""
import json
from pathlib import Path

import pytest

from localize.guardian.project_policy import (
    PolicyProposalError, build_policy_context, validate_policy_manifest,
    verify_policy_config,
)
from localize.guardian.real_data_impact import RealDataCorpus


def corpus(tmp_path: Path, *, accepted=True, target='Privet chat'):
    root = tmp_path / 'corpus'
    (root / 'repo/l10n').mkdir(parents=True)
    (root / 'repo/l10n/messages.properties').write_text('chat=Private chat\nother=Private chat on Privet\n')
    (root / 'repo/l10n/messages_pcm.properties').write_text(f'chat={target}\nother=Private chat for Privet\n')
    config = 'supported_locales:\n  - code: pcm\n'
    if accepted:
        config += 'accepted_source_identical_translations:\n  pcm:\n    chat: Private chat\n'
    (root / 'config.yaml').write_text(config)
    (root / 'changed-files.json').write_text(json.dumps(['l10n/messages_pcm.properties']))
    return RealDataCorpus(root=root, file_count=1)


def manifest(**changes):
    entry = dict(locale='pcm', key='chat', source_value='Private chat',
                 forbidden_value='Privet chat', evidence_feedback_ids=['review_comment:42'])
    entry.update(changes)
    return json.dumps({'entries': [entry]})


def test_literal_proposal_proves_observation_and_explicit_accepted_control(tmp_path):
    data = corpus(tmp_path)
    context = build_policy_context(data, bundle_digest='a' * 64)
    result = validate_policy_manifest(manifest(), context=context,
                                      evidence_ids=('review_comment:42',))
    assert result['status'] == 'needs_operator_review'
    assert result['fixtures'][0]['observed_forbidden'] is True
    assert result['fixtures'][0]['accepted_control_passes'] is True
    assert result['rules'][0]['keys'] == ['chat']
    assert result['rules'][0]['source_regex'] == r'\APrivate\ chat\Z'
    assert result['rules'][0]['forbidden_target_regex'] == r'\APrivet\ chat\Z'
    assert (data.root / 'config.yaml').read_bytes() == context.config_bytes


@pytest.mark.parametrize('accepted,target', [(False, 'Privet chat'), (True, 'Private chat')])
def test_current_or_model_text_cannot_supply_missing_independent_proof(tmp_path, accepted, target):
    context = build_policy_context(corpus(tmp_path, accepted=accepted, target=target), bundle_digest='a'*64)
    result = validate_policy_manifest(manifest(), context=context, evidence_ids=('review_comment:42',))
    assert result['status'] == 'evidence_incomplete'


@pytest.mark.parametrize('changes', [
    {'locale': '*'}, {'key': '*'}, {'source_value': 'Different'},
    {'evidence_feedback_ids': ['review_comment:99']},
    {'forbidden_value': 'Private chat'}, {'required_target_regex': '.*'},
])
def test_unsafe_out_of_scope_or_accepted_denials_are_rejected(tmp_path, changes):
    context = build_policy_context(corpus(tmp_path), bundle_digest='a'*64)
    with pytest.raises(PolicyProposalError):
        validate_policy_manifest(manifest(**changes), context=context,
                                 evidence_ids=('review_comment:42',))


@pytest.mark.parametrize('payload', ['{"entries":[],"entries":[]}', '{"entries":[],"config":{}}',
                                     '{"entries":[]}', '{"entries":[NaN]}'])
def test_invalid_manifest_fails_closed(tmp_path, payload):
    context = build_policy_context(corpus(tmp_path), bundle_digest='a'*64)
    with pytest.raises(PolicyProposalError):
        validate_policy_manifest(payload, context=context, evidence_ids=('review_comment:42',))


def test_changed_config_rejects_stale_proposal(tmp_path):
    data = corpus(tmp_path)
    context = build_policy_context(data, bundle_digest='a'*64)
    verify_policy_config(context)
    (data.root / 'config.yaml').write_text('supported_locales: []\n')
    with pytest.raises(PolicyProposalError, match='changed'):
        verify_policy_config(context)


def test_model_metacharacters_are_literals_not_regex(tmp_path):
    context = build_policy_context(corpus(tmp_path), bundle_digest='a'*64)
    result = validate_policy_manifest(manifest(forbidden_value='(a+)+$'), context=context,
                                      evidence_ids=('review_comment:42',))
    assert result['rules'][0]['forbidden_target_regex'] == r'\A\(a\+\)\+\$\Z'
    assert result['status'] == 'evidence_incomplete'


def test_author_can_decline_without_creating_a_policy_rule(tmp_path):
    context = build_policy_context(corpus(tmp_path), bundle_digest='a'*64)
    result = validate_policy_manifest(json.dumps({'entries': [], 'decline_reason': 'No supported literal defect'}),
                                      context=context, evidence_ids=('review_comment:42',))
    assert result['status'] == 'declined'
    assert result['rules'] == []


@pytest.mark.parametrize('key', ['chat*', 'chat?', 'items[0]/label'])
def test_actual_observed_glob_metacharacter_keys_cannot_broaden_policy(tmp_path, key):
    data = corpus(tmp_path)
    for file in (data.root / 'repo/l10n').iterdir():
        file.write_text(file.read_text().replace('chat=', key + '='))
    context = build_policy_context(data, bundle_digest='a'*64)
    assert any(entry.key == key for entry in context.entries)
    with pytest.raises(PolicyProposalError, match='exact'):
        validate_policy_manifest(manifest(key=key), context=context, evidence_ids=('review_comment:42',))


def test_large_multilocale_corpus_selects_feedback_before_observation_limits(tmp_path):
    data = corpus(tmp_path)
    config = data.root / 'config.yaml'
    config.write_text(config.read_text().replace('  - code: pcm', '  - code: pcm\n  - code: de'))
    (data.root / 'repo/l10n/messages_de.properties').write_text(''.join(f'other{i}=Target {i}\n' for i in range(2500)))
    context = build_policy_context(data, bundle_digest='a'*64,
                                   feedback_scopes=(('pcm', 'l10n/messages_pcm.properties'),))
    assert len(context.entries) == 2
    assert all(entry.locale_code == 'pcm' for entry in context.entries)


def test_large_single_file_uses_exact_feedback_anchors_without_truncation(tmp_path):
    data = corpus(tmp_path)
    source = data.root / 'repo/l10n/messages.properties'
    target = data.root / 'repo/l10n/messages_pcm.properties'
    source.write_text(source.read_text() + ''.join(f'large{i}=English source {i}\n' for i in range(2500)))
    target.write_text(target.read_text() + ''.join(f'large{i}=Pidgin target {i}\n' for i in range(2500)))
    context = build_policy_context(data, bundle_digest='a'*64,
                                   feedback_scopes=(('pcm', 'l10n/messages_pcm.properties'),),
                                   evidence_text='The complete label "Privet chat" is wrong.')
    assert len(context.entries) == 1
    assert context.entries[0].key == 'chat'
    with pytest.raises(PolicyProposalError, match='anchors'):
        build_policy_context(data, bundle_digest='a'*64,
                             feedback_scopes=(('pcm', 'l10n/messages_pcm.properties'),))


def test_existing_rule_identity_or_exact_predicate_is_not_redefined(tmp_path):
    from dataclasses import replace
    context = build_policy_context(corpus(tmp_path), bundle_digest='a'*64)
    result = validate_policy_manifest(manifest(), context=context, evidence_ids=('review_comment:42',))
    for existing in (result['rules'][0], {'id': result['rules'][0]['id'], 'severity': 'warning'}):
        with pytest.raises(PolicyProposalError, match='existing'):
            validate_policy_manifest(manifest(), context=replace(context, existing_rules=(existing,)),
                                     evidence_ids=('review_comment:42',))
