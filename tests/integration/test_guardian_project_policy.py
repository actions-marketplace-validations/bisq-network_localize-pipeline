"""Private policy authoring uses real coordinator accounting and source authority."""
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from localize.guardian.codex import CodexUsage
from localize.guardian.diagnostics import failure_audit
from localize.guardian.models import GuardianMode, RecurrenceCandidate
from localize.guardian.prevention_runtime import PreventionAuthorResult
from localize.guardian.project_policy import PolicyProposalError
from localize.guardian.state import GuardianState
from tests.unit.test_guardian_project_policy import corpus, manifest
from tests.unit.test_guardian_prevention_runtime import (
    _FakeBroker, _coordinator, _repository_policy, _open_source_kwargs,
    _historical_prevention_source,
)

NOW = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)


class PolicyAuthor:
    model = 'gpt-6-astra'
    max_attempts = 1

    def __init__(self, mutate=None):
        self.calls = 0
        self.mutate = mutate

    def run(self, *, workspace, scope, **kwargs):
        assert scope == 'project_config'
        self.calls += 1
        request = json.loads((workspace / 'request.json').read_text())
        assert 'config' not in request
        (workspace / 'proposal.json').write_text(manifest(locale='ru'))
        if self.mutate:
            self.mutate(workspace)
        return PreventionAuthorResult(attempts=1, usage=CodexUsage(cost_usd=.25))


def setup(tmp_path, state, *, historical=False, mutate=None, accepted=True):
    data = corpus(tmp_path, accepted=accepted)
    config = data.root / 'config.yaml'
    config.write_text(config.read_text().replace('pcm', 'ru'))
    (data.root / 'repo/l10n/messages_pcm.properties').rename(data.root / 'repo/l10n/messages_ru.properties')
    author = PolicyAuthor(mutate)
    coordinator = _coordinator(state=state, tmp_path=tmp_path, broker=_FakeBroker(), author=author)
    def no_authority(*args, **kwargs):
        pytest.fail('Project-only authoring touched a public broker, checkout, signing, or tests')
    coordinator.broker_factory = no_authority
    coordinator.checkout_factory = no_authority
    coordinator.publish_credential_environment = no_authority
    coordinator.test_runner = object()
    run_id = state.start_run(repository='acme/translations', locale='ru',
                             mode=GuardianMode.PROPOSE_PREVENTION, started_at=NOW)
    source_kwargs = _open_source_kwargs()
    revision_id = 1
    checks = []
    if historical:
        source, revision_id = _historical_prevention_source(state, now=NOW)
        source_kwargs = dict(source_pulls=(source,), source_event_revision_ids=(revision_id,),
                             require_exact_sources_still_closed=lambda *args: checks.append('historical'))
    else:
        source_kwargs['require_exact_open_source_authority'] = lambda *args: checks.append('live')
    source_kwargs.update(real_data=data, policy_config_digest='a' * 64)
    kwargs = dict(policy=_repository_policy(), recurrence_candidates=(RecurrenceCandidate(
        scope='project_config', summary='Keep reviewed Private chat wording',
        evidence_feedback_ids=('review_comment:42',)),),
        evidence_revision_ids={'review_comment:42': revision_id}, run_id=run_id,
        observed_at=NOW, require_live_lease=lambda: None,
        require_current_base_unchanged=lambda: None, **source_kwargs)
    return coordinator, author, kwargs, checks, data


@pytest.mark.parametrize('historical', [False, True])
def test_live_and_historical_policy_drafts_are_private_bounded_and_idempotent(tmp_path, historical):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state, historical=historical)
        original = (data.root / 'config.yaml').read_bytes()
        first = coordinator.propose(**kwargs)
        assert first.failures == ()
        assert len(first.policy_proposals) == 1
        assert first.drafts == ()
        assert len(checks) == 2
        path = tmp_path / 'policy-proposals'
        proposal = next(path.glob('*/proposal.json'))
        payload = json.loads(proposal.read_text())
        assert payload['status'] == 'needs_operator_review'
        assert payload['model'] == 'gpt-6-astra'
        assert payload['identity']['bundle_digest'] == 'a' * 64
        assert (proposal.parent / 'proposal.yaml').read_text().startswith('semantic_quality_rules:')
        assert proposal.stat().st_mode & 0o777 == 0o600
        assert proposal.parent.stat().st_mode & 0o777 == 0o700
        again = coordinator.propose(**kwargs)
        assert again.failures == ()
        assert again.policy_proposals == first.policy_proposals
        assert again.skipped == 1
        assert author.calls == 1
        assert state.model_calls_committed_for_day(NOW.date()) == 1
        assert state.cost_for_day(NOW.date()) == .25
        assert (data.root / 'config.yaml').read_bytes() == original
        actions = state._connection.execute('SELECT action,status FROM actions').fetchall()
        assert [tuple(row) for row in actions] == [('project_policy_proposal', 'pending')]
        assert state._connection.execute('SELECT COUNT(*) FROM prevention_draft_events').fetchone()[0] == 0


@pytest.mark.parametrize('mutation', ['extra_file', 'changed_input', 'symlink', 'unsafe_field', 'stale_config'])
def test_untrusted_output_and_stale_context_cannot_create_review_artifact(tmp_path, mutation):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        def mutate(workspace):
            if mutation == 'extra_file':
                (workspace / 'rules.py').write_text('raise RuntimeError()')
            elif mutation == 'changed_input':
                (workspace / 'request.json').write_text('{}')
            elif mutation == 'symlink':
                (workspace / 'proposal.json').unlink()
                (workspace / 'proposal.json').symlink_to(workspace / 'request.json')
            elif mutation == 'unsafe_field':
                (workspace / 'proposal.json').write_text(manifest(locale='ru', signing_key='model-owned'))
            else:
                (data.root / 'config.yaml').write_text('supported_locales: []\n')
        author.mutate = mutate
        result = coordinator.propose(**kwargs)
        assert result.failures
        assert result.policy_proposals == ()
        assert not (tmp_path / 'policy-proposals').exists()
        assert state.model_calls_committed_for_day(NOW.date()) == 1


def test_missing_accepted_control_is_reviewable_but_evidence_incomplete(tmp_path):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state, accepted=False)
        result = coordinator.propose(**kwargs)
        assert result.failures == ()
        proposal = next((tmp_path / 'policy-proposals').glob('*/proposal.json'))
        assert json.loads(proposal.read_text())['status'] == 'evidence_incomplete'


def test_exhausted_shared_model_budget_cannot_start_policy_author(tmp_path):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        coordinator.max_model_calls_per_day = 1
        state.try_reserve_model_call(run_id=kwargs['run_id'], daily_limit=1,
                                    model='gpt-6-sol', purpose='assessment', reserved_at=NOW)
        result = coordinator.propose(**kwargs)
        assert result.failures == ('PreventionRuntimeError',)
        assert author.calls == 0
        assert result.policy_proposals == ()


def test_policy_route_reuses_real_author_confinement_and_selected_model(tmp_path, monkeypatch):
    import subprocess
    from localize.guardian import prevention_runtime
    from localize.guardian.models import CodexAuthMode
    from localize.guardian.prevention_runtime import PreventionCodexAuthor
    from pathlib import Path

    seen = []
    def process(argv, **kwargs):
        assert argv[argv.index('--model') + 1] == 'gpt-6-astra'
        assert 'model_reasoning_effort="high"' in argv
        assert 'default_permissions="guardian_prevention_author"' in argv
        assert kwargs['shell'] is False
        assert kwargs['env']['CODEX_API_KEY'] == 'model-key'
        assert not {'GH_TOKEN', 'GITHUB_TOKEN', 'SSH_AUTH_SOCK', 'GIT_ASKPASS'} & kwargs['env'].keys()
        assert 'Write ONLY proposal.json' in kwargs['input']
        workspace = Path(argv[argv.index('-C') + 1])
        (workspace / 'proposal.json').write_text(manifest(locale='ru'))
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}, 'cost_usd': .25,
        }), '')
    monkeypatch.setattr(prevention_runtime, 'run_bounded_process', process)
    monkeypatch.setenv('GH_TOKEN', 'must-not-reach-author')
    monkeypatch.setenv('SSH_AUTH_SOCK', '/must-not-reach-author')
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        coordinator.author = PreventionCodexAuthor(model='gpt-6-astra', reasoning_effort='high',
                                                  auth_mode=CodexAuthMode.API_KEY, max_attempts=1)
        result = coordinator.propose(**kwargs)
        assert result.failures == ()
        assert len(seen) == 1
        assert len(result.policy_proposals) == 1


def test_policy_source_revalidation_failure_prevents_artifact(tmp_path):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        def changed(*args):
            checks.append('check')
            if len(checks) > 1:
                raise ValueError('source changed')
        kwargs['require_exact_open_source_authority'] = changed
        result = coordinator.propose(**kwargs)
        assert result.failures
        assert not (tmp_path / 'policy-proposals').exists()
        assert author.calls == 1


def test_unknown_feedback_binding_cannot_spend_model_budget(tmp_path):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        kwargs['recurrence_candidates'] = (replace(kwargs['recurrence_candidates'][0],
                                                   evidence_feedback_ids=('review_comment:99',)),)
        result = coordinator.propose(**kwargs)
        assert result.failures == ('PolicyProposalError',)
        assert author.calls == 0
        assert state.model_calls_committed_for_day(NOW.date()) == 0
        candidate, = kwargs.pop('recurrence_candidates')
        with pytest.raises(PolicyProposalError, match='^Policy feedback binding is unavailable\\.$'):
            coordinator._propose_project_policy(candidate=candidate, **kwargs)


def test_private_policy_author_failure_keeps_redacted_contextual_audit(tmp_path):
    def fail_author(workspace):
        raise RuntimeError('private-author-output-and-secret')

    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state, mutate=fail_author)
        with failure_audit(state):
            result = coordinator.propose(**kwargs)
        assert result.failures == ('RuntimeError',)
        assert result.policy_proposals == ()
        assert author.calls == 1
        assert state.model_calls_committed_for_day(NOW.date()) == 1
        failure = state.latest_health('guardian-failure')
        assert failure is not None
        assert failure.status == 'failed'
        assert failure.details['exception'] == 'RuntimeError'
        assert failure.details['repository'] == kwargs['policy'].base_repo
        assert failure.details['run_id'] == kwargs['run_id']
        assert any(':_propose_project_policy:' in location for location in failure.details['locations'])
        assert 'private-author-output-and-secret' not in json.dumps(failure.details)
        assert state._connection.execute("SELECT count(*) FROM health WHERE component='guardian-failure'").fetchone()[0] == 1
        assert not (tmp_path / 'policy-proposals').exists()


@pytest.mark.parametrize('malformation', [
    'list', 'null', 'string', 'missing_status', 'unknown_status', 'list_status', 'dict_status',
])
def test_malformed_existing_policy_artifact_cannot_recover_audit(tmp_path, malformation):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        assert coordinator.propose(**kwargs).failures == ()
        proposal = next((tmp_path / 'policy-proposals').glob('*/proposal.json'))
        payload = json.loads(proposal.read_text())
        if malformation == 'missing_status':
            del payload['status']
        elif malformation.endswith('_status'):
            payload['status'] = {'unknown_status': 'approved', 'list_status': [], 'dict_status': {}}[malformation]
        else:
            payload = {'list': [], 'null': None, 'string': 'not an artifact'}[malformation]
        proposal.write_text(json.dumps(payload))
        state._connection.execute("DELETE FROM actions WHERE action='project_policy_proposal'")
        state._connection.execute("DELETE FROM health WHERE component='project-policy'")
        state._connection.commit()

        replay = coordinator.propose(**kwargs)
        assert replay.failures == ('PolicyProposalError',)
        assert replay.policy_proposals == ()
        assert author.calls == 1
        assert state.model_calls_committed_for_day(NOW.date()) == 1
        assert state.latest_health('project-policy') is None
        assert state._connection.execute("SELECT count(*) FROM actions WHERE action='project_policy_proposal'").fetchone()[0] == 0


@pytest.fixture
def controller_runtime(tmp_path):
    from tests.unit.test_guardian_controller import runtime
    return runtime.__wrapped__(tmp_path)


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('candidate_count', [1, 2])
def test_actual_controller_routes_policy_candidate_and_trusted_config(tmp_path, controller_runtime, historical, candidate_count):
    from tests.unit import test_guardian_controller as fixture
    from localize.guardian.codex import GuardianRecurrenceCandidate
    from localize.guardian.prevention_runtime import PreventionCoordinator

    base, head, checkout, provider, broker, sequence = controller_runtime
    class Assessment(fixture.FakeCodexDriver):
        def run(self, task, *, success_observer=None, **kwargs):
            final = None
            def persist(attempt, usage, result):
                nonlocal final
                decision = replace(result.feedback[0], verdict='needs_human', replacements=(),
                                   report_reason='policy_conflict', decision_required=True)
                final = replace(result, feedback=(decision,), recurrence_candidates=tuple(
                    GuardianRecurrenceCandidate(scope='project_config', summary=f'Review exact project wording {index}',
                                                evidence_feedback_ids=(decision.feedback_id,))
                    for index in range(candidate_count)))
                if success_observer:
                    success_observer(attempt, usage, final)
            super().run(task, success_observer=persist, **kwargs)
            return final

    class ObservedAuthor(PolicyAuthor):
        def run(self, *, workspace, **kwargs):
            self.calls += 1
            request = json.loads((workspace / 'request.json').read_text())
            observed = request['observations'][0]
            (workspace / 'proposal.json').write_text(json.dumps({'entries': [dict(
                locale=observed['locale_code'], key=observed['key'], source_value=observed['source_value'],
                forbidden_value=observed['new_value'],
                evidence_feedback_ids=[request['feedback'][0]['feedback_id']],
            )]}))
            return PreventionAuthorResult(attempts=1, usage=CodexUsage(cost_usd=.25))

    def no_public_authority(*args, **kwargs):
        pytest.fail('Project policy must not use the pipeline publication path')
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        author = ObservedAuthor()
        prevention = PreventionCoordinator(
            state=state, checkout_factory=no_public_authority, broker_factory=no_public_authority,
            author=author, test_runner=object(), model_credential_provider=lambda: 'model-key',
            publish_credential_environment=no_public_authority, signing_key='unchanged-key',
            signing_environment=None, max_drafts=1, reservation_usd=1, daily_limit_usd=2,
            max_model_calls_per_day=2, temporary_root=tmp_path, now=lambda: fixture.NOW,
        )
        kwargs = {}
        if historical:
            policy = replace(fixture._historical_policy(), prevention=fixture._prevention_policy())
            config = fixture._config(GuardianMode.PROPOSE_PREVENTION, policies=(policy,))
            kwargs = dict(
                historical_snapshot_provider=fixture.FakeHistoricalSnapshotProvider(
                    (fixture._snapshot(pull=fixture._pull(state='closed')),)),
                historical_checkout_factory=fixture.FakeHistoricalCheckoutFactory(base, head, tmp_path),
                current_base_provider=fixture.FakeCurrentBaseProvider(),
            )
            provider = fixture.FakeSnapshotProvider(())
        else:
            config = fixture._config(GuardianMode.PROPOSE_PREVENTION)
        result = fixture._controller(tmp_path=tmp_path, state=state, config=config,
                                     checkout=checkout, provider=provider, driver=Assessment(),
                                     broker=broker, prevention_runner=prevention, **kwargs).poll_once()
        assert result.prevention_failures == ()
        assert result.prevention_items_deferred == candidate_count - 1
        assert state.latest_health('guardian').status == 'ok'
        assert len(result.project_policy_proposals) == 1
        assert result.applied_commits == ()
        assert result.prevention_drafts_created == 0
        assert author.calls == 1
        proposal = next((tmp_path / 'policy-proposals').glob('*/proposal.json'))
        payload = json.loads(proposal.read_text())
        assert len(payload['identity']['bundle_digest']) == 64
        assert payload['identity']['repository'] == 'acme/widgets'
        assert payload['status'] == 'evidence_incomplete'
        assert state.model_calls_committed_for_day(fixture.NOW.date()) == 2


@pytest.mark.parametrize('historical', [False, True])
def test_policy_authoring_cap_defers_new_work_and_reuses_existing_artifacts(tmp_path, historical):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state, historical=historical)
        coordinator.max_drafts = 1
        first = coordinator.propose(**kwargs)
        assert first.failures == ()
        second_kwargs = dict(kwargs, recurrence_candidates=(replace(
            kwargs['recurrence_candidates'][0], summary='Another reviewed Private chat policy candidate'),))
        with failure_audit(state):
            second = coordinator.propose(**second_kwargs)
        assert second.deferred == 1
        assert second.failures == ()
        assert second.policy_proposals == ()
        assert author.calls == 1
        assert state.latest_health('guardian-failure') is None
        replay = coordinator.propose(**kwargs)
        assert replay.policy_proposals == first.policy_proposals
        assert replay.skipped == 1 and replay.failures == ()
        assert author.calls == 1
        coordinator.begin_poll()
        resumed = coordinator.propose(**second_kwargs)
        assert resumed.failures == () and resumed.deferred == 0
        assert len(resumed.policy_proposals) == 1
        assert author.calls == 2
        assert state.model_calls_committed_for_day(NOW.date()) == 2
        assert {row[0] for row in state._connection.execute(
            "SELECT status FROM actions WHERE action='project_policy_proposal'")} == {'pending'}


def test_artifact_first_crash_recovers_audit_and_status_without_model_spend(tmp_path):
    with GuardianState(tmp_path / 'state.sqlite3') as state:
        coordinator, author, kwargs, checks, data = setup(tmp_path, state)
        result = coordinator.propose(**kwargs)
        assert result.failures == ()
        # Simulate interruption after the atomic artifact rename but before audit/status.
        state._connection.execute("DELETE FROM actions WHERE action='project_policy_proposal'")
        state._connection.execute("DELETE FROM health WHERE component='project-policy'")
        state._connection.commit()
        replay = coordinator.propose(**kwargs)
        assert replay.failures == ()
        assert author.calls == 1
        assert state.latest_health('project-policy').details['path'] == result.policy_proposals[0]
        assert state._connection.execute("SELECT count(*) FROM actions WHERE action='project_policy_proposal'").fetchone()[0] == 1
