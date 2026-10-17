"""Real controller, files, SQLite, and replay; only network/model are faked."""

from dataclasses import replace

import pytest

from localize.guardian.models import GuardianMode
from localize.guardian.state import GuardianState
from tests.unit.test_guardian_controller import (
    COMMIT_SHA,
    TARGET_PATH,
    FakeCodexDriver,
    TwoReplacementCodexDriver,
    _config,
    _controller,
    _feedback,
    _pull,
    _snapshot,
)
from tests.unit.test_guardian_controller import runtime as _runtime_fixture
from tests.unit.test_guardian_controller import _policy
from localize.guardian.models import TrustedActor
from tests.unit.test_guardian_reporting import observed_followups, recent_followups

runtime = _runtime_fixture


class InsufficientEvidenceDriver(FakeCodexDriver):
    """Assess every newly admitted event, including acknowledged bot replies."""

    def run(self, task, **kwargs):
        import json

        observer = kwargs.pop("success_observer", None)
        ids = json.loads((task.evidence_dir / "manifest.json").read_text())["feedback_ids"]

        def decisions(result):
            return replace(result, feedback=tuple(replace(
                result.feedback[0], feedback_id=identifier, verdict="needs_human",
                replacements=(), report_reason="insufficient_evidence", decision_required=True,
            ) for identifier in ids))

        def completed(attempt, usage, result):
            if observer:
                observer(attempt, usage, decisions(result))

        return decisions(super().run(task, success_observer=completed, **kwargs))


def test_unchanged_evidence_followups_converge_after_assessment(tmp_path, runtime):
    """Five natural followups keep one published decision and its summary link."""
    _, _, checkout, provider, broker, _ = runtime
    policy = replace(_policy(), trusted_bots={"ru": (TrustedActor("coderabbitai[bot]", 999, "Bot"),)})
    root = replace(_feedback(body="A Swahili reviewer should assess `umetumwa`."),
                   author_login="coderabbitai[bot]", author_id=999, author_type="Bot")
    feedback = [root]
    driver = InsufficientEvidenceDriver()
    config = replace(_config(GuardianMode.APPLY_OWNED_TRANSLATIONS), repositories=(policy,))
    config = replace(config, limits=replace(config.limits, max_model_calls_per_day=20, daily_cost_limit_usd=20))
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path, state=state,
            config=config,
            checkout=checkout, provider=provider, driver=driver, broker=broker,
        )
        provider.snapshots = (_snapshot(feedback=tuple(feedback)),)
        assert controller.poll_once().failures == ()
        original_summary = broker.feedback_summary.body
        for reply in (item for item in observed_followups() if not item["resolved"]):
            feedback.append(replace(root, source_id=reply["id"], body=reply["body"], in_reply_to_id=44))
            provider.snapshots = (_snapshot(feedback=tuple(feedback)),)
            outcome = controller.poll_once()
            assert outcome.failures == () and outcome.applied_commits == ()
            assert len(broker.feedback_reports) == 1
            assert broker.feedback_summary.body == original_summary
        assert len(driver.calls) == 6
        assert len(state.latest_event_revisions()) == 6
        assert all(state.latest_feedback_report(revision.revision_id) is not None
                   for revision in state.latest_event_revisions())
        # A fresh controller and SQLite connection retain the publication
        # aliases; process memory is not the convergence mechanism.
        with GuardianState(tmp_path / "state.sqlite3") as restarted_state:
            restarted = _controller(tmp_path=tmp_path, state=restarted_state, config=config,
                                    checkout=checkout, provider=provider, driver=driver, broker=broker)
            assert restarted.poll_once().failures == ()
            assert len(driver.calls) == 6 and len(broker.feedback_reports) == 1
        # A new literal is material evidence, even inside the same known grammar.
        feedback.append(replace(feedback[-1], source_id="9991", body=feedback[-1].body.replace("umetumwa", "another-word")))
        provider.snapshots = (_snapshot(feedback=tuple(feedback)),)
        assert controller.poll_once().failures == ()
        assert len(broker.feedback_reports) == 2
        # An edited repeat also remains publicly accountable.
        feedback[-2] = replace(feedback[-2], updated_at="2026-08-30T11:00:00Z",
                               body=feedback[-2].body.replace("translation concern", "finding"))
        provider.snapshots = (_snapshot(feedback=tuple(feedback)),)
        assert controller.poll_once().failures == ()
        assert f"review_comment:{feedback[-2].source_id}" in _posted_feedback_ids(state)
        assert len(broker.feedback_reports) <= 4


def test_edited_resolved_acknowledgement_is_reported(convergence_reporting):
    ctx = convergence_reporting
    observed = [item for item in observed_followups() if item["resolved"]]
    reply = replace(ctx.replies[0], body=observed[0]["body"])
    no_op = dict(outcome="not_applicable", verdict="reject", report_reason="not_applicable", decision_required=False)
    ctx.flush((ctx.root, reply), changes=no_op)
    assert len(ctx.broker.feedback_reports) == 1  # Only the original finding.
    edited = replace(reply, body=observed[1]["body"], updated_at="2026-08-30T11:00:00Z")
    ctx.flush((ctx.root, edited), changes=no_op)
    assert len(ctx.broker.feedback_reports) == 2
    assert f"review_comment:{reply.source_id}" in _posted_feedback_ids(ctx.state)
    ctx.flush((ctx.root, edited), changes=no_op)
    assert len(ctx.broker.feedback_reports) == 2


def _posted_feedback_ids(state):
    return {row[0] for row in state._connection.execute(
        "SELECT json_extract(payload_json, '$.feedback_id') FROM feedback_report_deliveries WHERE status = 'posted'"
    )}


@pytest.fixture
def convergence_reporting(tmp_path, runtime):
    """Real authority checks and outbox, with explicit completed assessments."""
    from types import SimpleNamespace
    from localize.guardian.authorization import authorize_feedback
    from tests.unit.test_guardian_controller import NOW, FakeSnapshotProvider

    _, _, checkout, _, broker, _ = runtime
    policy = replace(_policy(), trusted_bots={"ru": (TrustedActor("coderabbitai[bot]", 999, "Bot"),)})
    config = replace(_config(GuardianMode.APPLY_OWNED_TRANSLATIONS), repositories=(policy,))
    root = replace(_feedback(body="A Swahili reviewer should assess `umetumwa`."),
                   author_login="coderabbitai[bot]", author_id=999, author_type="Bot")
    bodies = [item["body"] for item in observed_followups() if not item["resolved"]]
    replies = tuple(replace(root, source_id=str(50 + index), body=body, in_reply_to_id=44)
                    for index, body in enumerate(bodies))
    details = dict(outcome="needs_human", verdict="needs_human", report_reason="insufficient_evidence",
                   decision_required=True, changed_keys=0, commit_sha=None,
                   report_policy_digest="policy", report_config_digest="config")
    with GuardianState(tmp_path / "state.sqlite3") as state:
        assert state.acquire_lease(name="guardian:poll", owner="test-owner", ttl_seconds=60, now=NOW)

        def flush(feedback, *, pull=None, changes=None):
            snapshot = _snapshot(feedback=tuple(feedback), pull=pull)
            authorized = authorize_feedback(policy=policy, snapshot=snapshot,
                                             path_locales={TARGET_PATH: "ru"}, changed_locales=("ru",))
            source = GuardianController._open_pull_authority_reference(
                policy=policy, snapshot=snapshot, authorized=authorized,
            )
            run_id = state.start_run(repository=policy.base_repo, locale="ru",
                                     mode=GuardianMode.APPLY_OWNED_TRANSLATIONS, started_at=NOW)
            current = []
            for event in authorized.events:
                revision = state.record_feedback_event(event, observed_at=NOW)
                state.record_action(run_id=run_id, event_revision_id=revision.revision_id,
                                    action="assess", status="completed", occurred_at=NOW,
                                    details=details | (changes or {}))
                current.append((event, revision))
            controller = _controller(tmp_path=tmp_path, state=state, config=config,
                                     checkout=checkout, provider=FakeSnapshotProvider((snapshot,)),
                                     driver=FakeCodexDriver(), broker=broker)
            controller._flush_feedback_reports(policy=policy, snapshot=snapshot, current=current,
                                               open_source=source, lease_owner="test-owner",
                                               policy_digest=(changes or {}).get("report_policy_digest", "policy"))
            return current

        from localize.guardian.controller import GuardianController
        yield SimpleNamespace(flush=flush, root=root, replies=replies, state=state,
                              broker=broker, details=details)


@pytest.mark.parametrize("control", [
    "new_head", "new_base", "root_edit", "missing_root", "different_root", "mixed",
    "unknown_prose", "new_literal", "new_language", "new_config", "new_policy",
    "held_edits", "deferred_edits", "changed_reason", "changed_outcome", "edited_reply",
])
def test_material_followup_controls_are_public(convergence_reporting, control):
    ctx = convergence_reporting
    ctx.flush((ctx.root, ctx.replies[0]))
    assert len(ctx.broker.feedback_reports) == 1
    reply, root, pull, changes = ctx.replies[1], ctx.root, None, {}
    if control == "new_head":
        pull = _pull(head_sha="e" * 40)
    elif control == "new_base":
        pull = _pull(base_sha="e" * 40)
    elif control == "root_edit":
        root = replace(root, body=root.body + " New evidence.")
    elif control == "different_root":
        root, reply = replace(root, source_id="99"), replace(reply, in_reply_to_id=99)
    elif control in {"mixed", "unknown_prose"}:
        reply = replace(reply, body=reply.body.replace("\n\n", " Please change the other label.\n\n", 1)
                        if control == "mixed" else reply.body + "\nAnother placeholder is missing.")
    elif control == "new_literal":
        reply = replace(reply, body=reply.body.replace("umetumwa", "new-wording"))
    elif control == "new_language":
        reply = replace(reply, body=reply.body.replace("Swahili", "French"))
    elif control == "new_config":
        changes = {"report_config_digest": "new-config"}
    elif control == "new_policy":
        changes = {"report_policy_digest": "new-policy"}
    elif control == "held_edits":
        changes = {"held_value_edits": 1}
    elif control == "deferred_edits":
        changes = {"deferred_value_edits": 1}
    elif control == "changed_reason":
        changes = {"report_reason": "glossary_conflict"}
    elif control == "changed_outcome":
        changes = {"outcome": "translation_batch_deferred"}
    elif control == "edited_reply":
        reply = replace(ctx.replies[0], body=ctx.replies[1].body)
    ctx.flush((reply,) if control == "missing_root" else (root, reply), pull=pull, changes=changes)
    assert len(ctx.broker.feedback_reports) >= 2
    assert ({f"review_comment:{reply.source_id}", f"review_comment:{root.source_id}"}
            & _posted_feedback_ids(ctx.state))


@pytest.mark.parametrize("status", ["posted", "pending"])
def test_legacy_outbox_bytes_survive_convergence_upgrade(convergence_reporting, status):
    import json
    from localize.guardian.reporting import report_body, report_key

    ctx = convergence_reporting
    body = report_body(ctx.details, repository="acme/widgets", feedback_id="review_comment:44", pull_number=12)
    key = report_key(repository_id=42, pull_number=12, feedback_id="review_comment:44", body=body)
    payload = dict(body=body, repository_id=42, pr_number=12, feedback_id="review_comment:44")
    ctx.state.queue_feedback_report(key, payload)
    if status == "posted":
        ctx.state.finish_feedback_report(key, result=dict(url="https://github.com/acme/widgets/pull/12#discussion_r200", body=body))
    original = ctx.state.feedback_report_delivery(key)["payload_json"]
    ctx.flush((ctx.root, ctx.replies[0]))
    assert ctx.state.feedback_report_delivery(key)["payload_json"] == original
    assert json.loads(original) == payload
    count = len(ctx.broker.feedback_reports)
    ctx.flush((ctx.root, *ctx.replies[:2]))
    assert len(ctx.broker.feedback_reports) == count
    assert ctx.state.feedback_reporting_counts()[0] == 0


def test_ambiguous_context_does_not_suppress_delivery(convergence_reporting):
    ctx = convergence_reporting
    ctx.broker.report_error = RuntimeError("acknowledgement lost")
    with pytest.raises(RuntimeError):
        ctx.flush((ctx.root,))
    row = ctx.state._connection.execute("SELECT payload_json FROM feedback_report_deliveries WHERE status = 'pending'").fetchone()
    import json
    identity = json.loads(row["payload_json"])["convergence_key"]
    assert ctx.state.posted_convergent_feedback_report(identity) is None
    ctx.broker.report_error = None
    ctx.flush((ctx.root, ctx.replies[0]))
    assert len(ctx.broker.feedback_reports) == 1
    assert ctx.state.feedback_reporting_counts()[0] == 0


def test_convergent_receipt_has_no_intermediate_pending_write(convergence_reporting):
    ctx = convergence_reporting
    ctx.flush((ctx.root,))
    # Reject even a transient pending alias. A two-transaction queue/finish
    # sequence would fail here and could produce a duplicate after a crash.
    ctx.state._connection.execute("""CREATE TRIGGER alias_must_arrive_posted
        BEFORE INSERT ON feedback_report_deliveries
        WHEN json_extract(NEW.payload_json, '$.converged_to') IS NOT NULL
          AND NEW.status != 'posted'
        BEGIN SELECT RAISE(ABORT, 'pending alias'); END""")
    ctx.flush((ctx.root, ctx.replies[0]))
    assert len(ctx.broker.feedback_reports) == 1
    database = ctx.state._connection.execute("PRAGMA database_list").fetchone()[2]
    with GuardianState(database) as restarted:
        aliases = restarted._connection.execute("""SELECT status FROM feedback_report_deliveries
            WHERE json_extract(payload_json, '$.converged_to') IS NOT NULL""").fetchall()
        assert [row[0] for row in aliases] == ["posted"]
        assert restarted.feedback_reporting_counts()[0] == 0


def test_return_to_previous_decision_is_a_new_public_context(convergence_reporting):
    ctx = convergence_reporting
    ctx.flush((ctx.root, ctx.replies[0]))
    assert len(ctx.broker.feedback_reports) == 1
    ctx.flush((ctx.root,), changes=dict(outcome="not_applicable", verdict="reject",
                                       report_reason="not_applicable", decision_required=False))
    previous = len(ctx.broker.feedback_reports)
    ctx.flush((ctx.root, ctx.replies[1]))
    assert len(ctx.broker.feedback_reports) == previous + 1
    assert f"review_comment:{ctx.replies[1].source_id}" in _posted_feedback_ids(ctx.state)
    ctx.flush((ctx.root, *ctx.replies[1:3]))
    assert len(ctx.broker.feedback_reports) == previous + 1


@pytest.mark.parametrize("change", [None, "literal", "language", "mixed"])
def test_fresh_swahili_hold_variant_keeps_material_feedback_visible(convergence_reporting, change):
    ctx = convergence_reporting
    ctx.flush((ctx.root, ctx.replies[0]))
    assert len(ctx.broker.feedback_reports) == 1
    actual = next(item for item in recent_followups() if item["id"] == "4302441045")
    body = actual["body"]
    if change == "literal":
        body = body.replace("a translation defect.", "a translation defect in `new wording`.")
    elif change == "language":
        body = body.replace("Swahili", "Hungarian")
    elif change == "mixed":
        body = body.replace("Keep the translation", "Please change the other label. Keep the translation")
    reply = replace(ctx.replies[1], source_id=actual["id"], body=body)
    current = ctx.flush((ctx.root, ctx.replies[0], reply))
    assert len(ctx.broker.feedback_reports) == (1 if change is None else 2)
    revision = next(revision for event, revision in current if event.event_id == actual["id"])
    assessment = ctx.state.latest_feedback_report(revision.revision_id)
    assert assessment["verdict"] == "needs_human" and assessment["decision_required"] is True
    assert "maintainer decision still required" in ctx.broker.feedback_summary.body


@pytest.mark.parametrize(
    "reason", ["as_suggested", "alternative_source_fidelity", "alternative_other"]
)
def test_held_accepted_decision_is_normalized_before_replay_and_persistence(
    tmp_path, runtime, reason
):
    """A later hold must not claim that a new correction was accepted."""
    _, _, checkout, provider, broker, _ = runtime

    class AcceptedDecisionDriver(DecisionDriver):
        @staticmethod
        def decision(result):
            return replace(result, feedback=(replace(
                result.feedback[0], report_reason=reason, decision_required=True
            ),))

    with GuardianState(tmp_path / "state.sqlite3") as state:
        common = dict(
            tmp_path=tmp_path, state=state,
            checkout=checkout, provider=provider, broker=broker,
        )
        first = _controller(
            **common, config=_config(GuardianMode.OBSERVE),
            driver=AcceptedDecisionDriver(),
        ).poll_once()
        assert first.failures == () and first.applied_commits == ()
        revision = state.latest_event_revisions()[0]
        original = state.latest_feedback_report(revision.revision_id)
        replay = state.held_feedback_report_for_revision(
            revision.revision_id, original["report_policy_digest"]
        )
        assert replay["report_reason"] == "insufficient_evidence"
        assert replay["verdict"] == "needs_human"
        provider.snapshots = (_snapshot(pull=_pull(head_sha="d" * 40)),)
        second = _controller(
            **common, config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            driver=FakeCodexDriver(),
        ).poll_once()
        assert second.failures == () and second.applied_commits == ()
        held_reports = [reply.body for reply in broker.feedback_reports.values()
                        if "Maintainer decision needed" in reply.body]
        assert held_reports
        assert all("insufficient to authorize" in body for body in held_reports)
        revision = state.latest_event_revisions()[0]
        details = state.latest_feedback_report(revision.revision_id)
        assert details["verdict"] == "needs_human"
        assert details["report_reason"] == "insufficient_evidence"
        assert details["decision_required"] is True


class DecisionDriver(FakeCodexDriver):
    @staticmethod
    def decision(result):
        return replace(
            result,
            feedback=(
                replace(
                    result.feedback[0],
                    verdict="needs_human",
                    replacements=(),
                    report_reason="glossary_conflict",
                    decision_required=True,
                    rationale="Private explanation /Users/operator/secrets password=do-not-publish",
                ),
            ),
        )

    def run(self, task, **kwargs):
        observer = kwargs.pop("success_observer", None)

        def completed(attempt, usage, result):
            if observer:
                observer(attempt, usage, self.decision(result))

        return self.decision(super().run(task, success_observer=completed, **kwargs))


def test_human_explanation_and_summary_are_durable_without_repeated_model_calls(
    tmp_path, runtime
):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=driver,
            broker=broker,
        )
        first = controller.poll_once()
        assert first.failures == ()
        assert first.applied_commits == ()
        assert len(broker.feedback_reports) == 1
        body = next(iter(broker.feedback_reports.values())).body
        assert "configured glossary" in body and "maintainer decision" in body
        assert "/Users/" not in body and "password" not in body
        assert "maintainer decision still required" in broker.feedback_summary.body
        second = controller.poll_once()
        assert second.failures == () and second.runs_started == 0
        assert len(driver.calls) == 1 and len(broker.feedback_reports) == 1


def test_lost_ack_retries_reporting_not_assessment(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    broker.report_error = RuntimeError("response lost after comment creation")
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=driver,
            broker=broker,
        )
        first = controller.poll_once()
        assert first.failures == ("RuntimeError",)
        assert state.latest_health("feedback-reporting").status == "failed"
        assert len(broker.feedback_reports) == 1
        broker.report_error = None
        assert controller.poll_once().failures == ()
        assert len(driver.calls) == 1 and len(broker.feedback_reports) == 1
        assert state.latest_health("feedback-reporting").status == "ok"


def test_new_head_cannot_make_model_silently_overrule_held_decision(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    with GuardianState(tmp_path / "state.sqlite3") as state:
        common = dict(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            broker=broker,
        )
        assert _controller(**common, driver=DecisionDriver()).poll_once().failures == ()
        provider.snapshots = (_snapshot(pull=_pull(head_sha="d" * 40)),)
        second = _controller(**common, driver=FakeCodexDriver()).poll_once()
        assert second.failures == ()
        assert second.applied_commits == () and second.prepared_value_edits == 0
        assert len(broker.feedback_reports) == 1


def test_observation_does_not_publish_explanations(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    with GuardianState(tmp_path / "state.sqlite3") as state:
        result = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.OBSERVE),
            checkout=checkout,
            provider=provider,
            driver=DecisionDriver(),
            broker=broker,
        ).poll_once()
        assert result.failures == ()
        assert broker.feedback_reports == {} and broker.feedback_summary is None


def test_published_correction_report_recovers_on_exact_successor_without_model_call(
    tmp_path, runtime
):
    _, _, checkout, provider, broker, _ = runtime
    driver = FakeCodexDriver()
    broker.report_error = RuntimeError("report acknowledgement lost")
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=driver,
            broker=broker,
        )
        first = controller.poll_once()
        assert first.applied_commits == (COMMIT_SHA,)
        assert state.feedback_reporting_counts()[0] == 1
        provider.snapshots = (_snapshot(pull=_pull(head_sha=COMMIT_SHA)),)
        broker.report_error = None
        second = controller.poll_once()
        assert second.failures == () and second.applied_commits == ()
        assert len(driver.calls) == 1 and len(broker.feedback_reports) == 1
        assert state.feedback_reporting_counts()[0] == 0
        assert broker.feedback_summary is not None


def test_withdrawn_feedback_clears_summary_without_claiming_policy_approval(
    tmp_path, runtime
):
    _, _, checkout, provider, broker, _ = runtime
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=DecisionDriver(),
            broker=broker,
        )
        assert controller.poll_once().failures == ()
        provider.snapshots = (_snapshot(feedback=()),)
        assert controller.poll_once().failures == ()
        assert "No current authorized feedback reports" in broker.feedback_summary.body
        assert "does not approve" in broker.feedback_summary.body
        assert state.feedback_reporting_counts()[1] == 1


def test_summary_can_return_to_previously_posted_body(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=DecisionDriver(),
            broker=broker,
        )
        assert controller.poll_once().failures == ()
        first = broker.feedback_summary.body
        provider.snapshots = (_snapshot(feedback=(_feedback(source_id="45"),)),)
        assert controller.poll_once().failures == ()
        assert broker.feedback_summary.body != first
        provider.snapshots = (_snapshot(),)
        assert controller.poll_once().failures == ()
        assert broker.feedback_summary.body == first


def test_report_retry_leaves_active_queue_when_pull_leaves_open_intake(
    tmp_path, runtime
):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    broker.report_error = RuntimeError("response lost")
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            driver=driver,
            broker=broker,
        )
        assert controller.poll_once().failures == ("RuntimeError",)
        assert state.feedback_reporting_counts() == (1, 1)
        provider.snapshots = ()
        broker.report_error = None
        assert controller.poll_once().failures == ()
        assert state.feedback_reporting_counts() == (0, 1)
        assert len(driver.calls) == 1
        # A reopened eligible pull can reconcile the same exact comment.
        provider.snapshots = (_snapshot(),)
        assert controller.poll_once().failures == ()
        assert len(broker.feedback_reports) == 1 and len(driver.calls) == 1


def test_failed_processing_has_safe_deferred_report_intent(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    driver = DecisionDriver()
    with GuardianState(tmp_path / "state.sqlite3") as state:
        config = _config(GuardianMode.APPLY_OWNED_TRANSLATIONS)
        config = replace(
            config, limits=replace(config.limits, max_model_calls_per_day=1)
        )
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=config,
            checkout=checkout,
            provider=provider,
            driver=driver,
            broker=broker,
        )
        assert controller.poll_once().failures == ()
        provider.snapshots = (_snapshot(feedback=(_feedback(source_id="45"),)),)
        controller.poll_once()
        # Next intake reports the durable failure without an extra model call.
        controller.poll_once()
        assert len(driver.calls) == 1
        assert any(
            "Processing did not complete" in reply.body
            and "Deferred work" in reply.body
            for reply in broker.feedback_reports.values()
        )


def test_failed_reassessment_cannot_hide_existing_glossary_decision(tmp_path, runtime):
    _, _, checkout, provider, broker, _ = runtime
    with GuardianState(tmp_path / "state.sqlite3") as state:
        common = dict(
            tmp_path=tmp_path,
            state=state,
            config=_config(GuardianMode.APPLY_OWNED_TRANSLATIONS),
            checkout=checkout,
            provider=provider,
            broker=broker,
        )
        assert _controller(**common, driver=DecisionDriver()).poll_once().failures == ()
        provider.snapshots = (_snapshot(pull=_pull(head_sha="d" * 40)),)
        failing = _controller(
            **common, driver=FakeCodexDriver(error=RuntimeError("private failure"))
        )
        assert failing.poll_once().failures == ("RuntimeError",)
        assert state.feedback_reporting_counts()[1] == 1
        assert "maintainer decision still required" in broker.feedback_summary.body
        failing.poll_once()
        assert "maintainer decision still required" in broker.feedback_summary.body
        assert state.feedback_reporting_counts()[1] == 1


def test_partial_glossary_alternative_explicitly_holds_remaining_edits(
    tmp_path, runtime
):
    base, head, checkout, provider, broker, _ = runtime
    for root in (base, head):
        source = root / "l10n/messages_en.properties"
        source.write_text(source.read_text() + "alpha=Alpha\n")
        target = root / TARGET_PATH
        target.write_text(target.read_text() + "alpha=Старый альфа\n")

    class AlternativeDriver(TwoReplacementCodexDriver):
        @staticmethod
        def _with_alpha(result):
            result = TwoReplacementCodexDriver._with_alpha(result)
            return replace(
                result,
                feedback=(
                    replace(
                        result.feedback[0],
                        report_reason="alternative_glossary",
                        decision_required=True,
                    ),
                ),
            )

    config = _config(GuardianMode.APPLY_OWNED_TRANSLATIONS)
    config = replace(config, limits=replace(config.limits, max_value_edits_per_run=1))
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(
            tmp_path=tmp_path,
            state=state,
            config=config,
            checkout=checkout,
            provider=provider,
            driver=AlternativeDriver(),
            broker=broker,
        )
        first = controller.poll_once()
        assert first.applied_commits == (COMMIT_SHA,)
        assert first.deferred_value_edits == 1
        assert any(
            "1 remaining correction needs maintainer approval" in reply.body
            for reply in broker.feedback_reports.values()
        )
        provider.snapshots = (_snapshot(pull=_pull(head_sha=COMMIT_SHA)),)
        second = controller.poll_once()
        assert second.failures == () and second.applied_commits == ()
        assert state.feedback_reporting_counts()[1] == 1
        assert "maintainer decision still required" in broker.feedback_summary.body
