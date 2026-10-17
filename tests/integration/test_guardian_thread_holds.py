"""A completed thread decision owns subsequent bot evidence before proposals."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from localize.guardian.models import GuardianMode, TrustedActor
from localize.guardian.state import GuardianState
from tests.integration.test_guardian_feedback_reporting import InsufficientEvidenceDriver
from tests.unit.test_guardian_controller import _config, _controller, _feedback, _policy, _snapshot
from tests.unit.test_guardian_controller import runtime as _runtime_fixture

runtime = _runtime_fixture

FRESH_REPLY = ("@hiciefte The finding remains unconfirmed. No new evidence supports a correction. "
               "Leave the translation and configuration unchanged pending a maintainer decision."
               "\n\n_You are interacting with an AI system._\n\n"
               "<!-- This is an auto-generated reply by CodeRabbit -->")


@pytest.fixture
def held_thread(tmp_path, runtime):
    from types import SimpleNamespace

    base, head, checkout, provider, broker, _ = runtime
    policy = replace(_policy(), trusted_bots={"ru": (TrustedActor("coderabbitai[bot]", 999, "Bot"),)})
    root = replace(_feedback(body="A Swahili reviewer should assess `umetumwa`."),
                   author_login="coderabbitai[bot]", author_id=999, author_type="Bot")
    driver = InsufficientEvidenceDriver()
    config = replace(_config(GuardianMode.APPLY_OWNED_TRANSLATIONS), repositories=(policy,))
    with GuardianState(tmp_path / "state.sqlite3") as state:
        controller = _controller(tmp_path=tmp_path, state=state, config=config,
                                 checkout=checkout, provider=provider, driver=driver, broker=broker)
        provider.snapshots = (_snapshot(feedback=(root,)),)
        assert controller.poll_once().failures == ()
        yield SimpleNamespace(**locals())


@pytest.mark.parametrize("body", [FRESH_REPLY, "An unseen paraphrase.",
    FRESH_REPLY + " Change the translation now.", "Use `another-word`.",
    "A Hungarian reviewer supplied new evidence."])
def test_thread_hold_admits_new_evidence_without_model_or_public_reply(held_thread, body):
    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=body, in_reply_to_id=44)
    summary = ctx.broker.feedback_summary.body
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == ()
    assert len(ctx.driver.calls) == 1
    assert len(ctx.broker.feedback_reports) == 1 and ctx.broker.feedback_summary.body == summary
    revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == reply.source_id)
    assert revision.body == body
    row = ctx.state._connection.execute(
        "SELECT details_json FROM actions WHERE event_revision_id = ? ORDER BY action_id DESC LIMIT 1",
        (revision.revision_id,),
    ).fetchone()
    details = json.loads(row[0])
    assert details["outcome"] == "thread_human_review"
    assert details["thread_hold"]["classification"] == "new_evidence"
    assert "verdict" not in details and "report_reason" not in details and "confidence" not in details
    total, pending = ctx.state.thread_hold_evidence_for_operator(limit=1)
    assert total == 1 and pending[0]["event_id"] == reply.source_id
    assert pending[0]["root_action_id"] and pending[0]["url"].endswith("discussion_r4302557939")
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 1
    assert ctx.state._connection.execute("SELECT COUNT(*) FROM actions WHERE event_revision_id = ?",
                                         (revision.revision_id,)).fetchone()[0] == 1


def test_held_bot_evidence_does_not_swallow_unrelated_feedback(held_thread):
    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    other = replace(ctx.root, source_id="77", body="An unrelated actionable finding.")
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply, other)),)
    assert ctx.controller.poll_once().failures == ()
    assert len(ctx.driver.calls) == 2
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 1
    revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == "77")
    assert ctx.state.latest_feedback_report(revision.revision_id)["verdict"] == "needs_human"


def test_failed_reply_history_and_ambiguous_delivery_are_preserved_under_hold(held_thread):
    from localize.guardian.authorization import authorize_feedback
    from tests.unit.test_guardian_controller import NOW, TARGET_PATH

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    snapshot = _snapshot(feedback=(ctx.root, reply))
    authorized = authorize_feedback(policy=ctx.policy, snapshot=snapshot,
                                     path_locales={TARGET_PATH: "ru"}, changed_locales=("ru",))
    event = next(item for item in authorized.events if item.event_id == reply.source_id)
    revision = ctx.state.record_feedback_event(event, observed_at=NOW)
    run = ctx.state.start_run(repository=event.repository, locale="ru", mode=ctx.config.mode, started_at=NOW)
    ctx.state.record_action(run_id=run, event_revision_id=revision.revision_id, action=ctx.config.mode.value,
        status="failed", occurred_at=NOW, details={"outcome": "orchestration_failure",
            "report_outcome": "failed", "report_reason": "processing_failure", "decision_required": False})
    ctx.state.finish_run(run, status="failed", summary="Historical validation failure.", finished_at=NOW)
    before = tuple(ctx.state._connection.execute("SELECT * FROM actions WHERE run_id = ?", (run,)).fetchone())
    ctx.state.queue_feedback_report("f" * 64, {"feedback_id": event.feedback_id,
        "repository_id": ctx.policy.base_repo_id, "pr_number": event.pr_number, "body": "Ambiguous prior intent."})
    pending = dict(ctx.state.feedback_report_delivery("f" * 64))
    ctx.provider.snapshots = (snapshot,)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 1
    assert ctx.state.latest_feedback_report(revision.revision_id)["report_reason"] == "processing_failure"
    assert len(ctx.broker.feedback_reports) == 1
    assert dict(ctx.state.feedback_report_delivery("f" * 64)) == pending
    assert tuple(ctx.state._connection.execute("SELECT * FROM actions WHERE run_id = ?", (run,)).fetchone()) == before
    with GuardianState(ctx.tmp_path / "state.sqlite3") as restarted:
        assert restarted.thread_hold_evidence_for_operator()[0] == 1
        controller = _controller(tmp_path=ctx.tmp_path, state=restarted, config=ctx.config,
            checkout=ctx.checkout, provider=ctx.provider, driver=ctx.driver, broker=ctx.broker)
        assert controller.poll_once().failures == () and len(ctx.driver.calls) == 1


def test_trusted_human_reply_is_actionable_alongside_held_bot_evidence(held_thread):
    ctx = held_thread
    bot = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    human = replace(_feedback(source_id="78", body="Native reviewer supplies new evidence."), in_reply_to_id=44)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, bot, human)),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 2
    revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == "78")
    assert ctx.state.latest_feedback_report(revision.revision_id)["verdict"] == "needs_human"
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 1


@pytest.mark.parametrize("change", ["policy", "config", "mode", "head", "base", "root", "decision"])
def test_thread_hold_reopens_when_its_authority_changes(held_thread, change):
    from tests.unit.test_guardian_controller import NOW, _pull

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 1
    config, root, pull = ctx.config, ctx.root, _pull()
    if change == "policy":
        config = replace(config, limits=replace(config.limits, min_apply_confidence=0.99))
    elif change == "config":
        path = ctx.base / ".localize/config.yaml"
        path.write_text(path.read_text() + "\n# Deliberate governed configuration revision after maintainer review.\n")
    elif change == "mode":
        config = replace(config, mode=GuardianMode.OBSERVE)
    elif change in {"head", "base"}:
        pull = replace(pull, **{change + "_sha": "d" * 40})
    elif change == "root":
        root = replace(root, body=root.body + " New material source evidence.")
    else:
        revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == "44")
        run = ctx.state.start_run(repository=root.repository, locale="ru", mode=config.mode, started_at=NOW)
        ctx.state.record_action(run_id=run, event_revision_id=revision.revision_id, action=config.mode.value,
                                status="completed", occurred_at=NOW,
                                details={"verdict": "reject", "outcome": "not_applicable"})
        ctx.state.finish_run(run, status="completed", summary="New root decision.", finished_at=NOW)
    ctx.provider.snapshots = (_snapshot(feedback=(root, reply), pull=pull),)
    controller = _controller(tmp_path=ctx.tmp_path, state=ctx.state, config=config,
                             checkout=ctx.checkout, provider=ctx.provider, driver=ctx.driver, broker=ctx.broker)
    assert controller.poll_once().failures == ()
    assert len(ctx.driver.calls) == 2
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 0


def test_new_root_decision_releases_unassessed_evidence_without_resurrecting_hold(held_thread):
    from tests.unit.test_guardian_controller import NOW, FakeCodexDriver

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == ()
    root_revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == "44")
    old = dict(ctx.state.latest_feedback_report(root_revision.revision_id))
    run = ctx.state.start_run(repository=ctx.root.repository, locale="ru", mode=ctx.config.mode, started_at=NOW)
    ctx.state.record_action(run_id=run, event_revision_id=root_revision.revision_id, action=ctx.config.mode.value,
        status="completed", occurred_at=NOW,
        details={**old, "outcome": "not_applicable", "verdict": "reject", "report_reason": "not_applicable", "decision_required": False})
    ctx.state.finish_run(run, status="completed", summary="Superseding assessment.", finished_at=NOW)
    revision = next(item for item in ctx.state.latest_event_revisions() if item.event_id == reply.source_id)
    assert ctx.state.thread_hold_for_revision(revision.revision_id) is None
    class ReleasedDriver(FakeCodexDriver):
        @staticmethod
        def decide(result):
            return replace(result, feedback=tuple(replace(item, verdict="reject", replacements=(),
                report_reason="not_applicable", decision_required=False) for item in result.feedback))
        def run(self, task, **kwargs):
            observer = kwargs.pop("success_observer", None)
            return self.decide(super().run(task, success_observer=(
                lambda attempt, usage, result: observer(attempt, usage, self.decide(result))) if observer else None, **kwargs))
    driver = ReleasedDriver()
    controller = _controller(tmp_path=ctx.tmp_path, state=ctx.state, config=ctx.config,
                             checkout=ctx.checkout, provider=ctx.provider, driver=driver, broker=ctx.broker)
    assert controller.poll_once().failures == () and len(driver.calls) == 1
    assert ctx.state.latest_feedback_report(revision.revision_id)["verdict"] == "reject"
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 0


@pytest.mark.parametrize("change", ["missing_config", "failed", "running", "nonheld", "held_edits", "deferred", "recurrence", "wrong_reason", "unbound"])
def test_incomplete_or_conflicting_seed_never_invents_thread_hold(held_thread, change):
    from tests.unit.test_guardian_controller import NOW

    ctx = held_thread
    revision = ctx.state.latest_event_revisions()[0]
    details = dict(ctx.state.latest_feedback_report(revision.revision_id))
    if change == "missing_config":
        details.pop("report_config_digest")
    elif change == "unbound":
        details["report_config_digest"] = "unbound"
    elif change == "held_edits":
        details["held_value_edits"] = 1
    elif change == "deferred":
        details["deferred_value_edits"] = 1
    elif change == "recurrence":
        details["recurrence_candidates"] = 1
    elif change == "wrong_reason":
        details["report_reason"] = "glossary_conflict"
    elif change == "nonheld":
        details.update(verdict="reject", outcome="not_applicable", decision_required=False)
    run = ctx.state.start_run(repository=ctx.root.repository, locale="ru", mode=ctx.config.mode, started_at=NOW)
    ctx.state.record_action(run_id=run, event_revision_id=revision.revision_id, action=ctx.config.mode.value,
        status="failed" if change == "failed" else "completed", occurred_at=NOW, details=details)
    if change != "running":
        ctx.state.finish_run(run, status="failed" if change == "failed" else "completed", summary="Conflicting seed.", finished_at=NOW)
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 2
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 0


@pytest.mark.parametrize("change", ["bot_name", "actor", "unbound_config"])
def test_reply_and_current_config_authority_are_indispensable(held_thread, monkeypatch, change):
    import localize.guardian.controller as controller_module

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    if change == "bot_name":
        reply = replace(reply, author_login="another-bot[bot]")
    elif change == "actor":
        # An authorized second actor still cannot inherit this root's hold.
        policy = replace(ctx.policy, trusted_bots={"ru": (*ctx.policy.trusted_bots["ru"], TrustedActor("second[bot]", 998, "Bot"))})
        config = replace(ctx.config, repositories=(policy,))
        # Establish the root under this exact policy before trying its reply.
        controller = _controller(tmp_path=ctx.tmp_path, state=ctx.state, config=config,
            checkout=ctx.checkout, provider=ctx.provider, driver=ctx.driver, broker=ctx.broker)
        assert controller.poll_once().failures == ()
        ctx.controller, ctx.config = controller, config
        reply = replace(reply, author_id=998)
    else:
        monkeypatch.setattr(controller_module, "_base_pipeline_config_bundle_digest", lambda **_: None)
    calls = len(ctx.driver.calls)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == calls + 1
    assert ctx.state.thread_hold_evidence_for_operator()[0] == 0


def test_pending_publication_prevents_thread_hold_partition(held_thread, monkeypatch):
    from types import SimpleNamespace
    from tests.unit.test_guardian_controller import NOW

    ctx = held_thread
    monkeypatch.setattr(ctx.state, "pending_publications", lambda **_: (object(),))
    monkeypatch.setattr(ctx.state, "thread_hold_seed", lambda *_a, **_k: pytest.fail("Pending publication must veto seed lookup"))
    assert not ctx.controller._hold_thread_replies(policy=ctx.policy, snapshot=None,
        current=(), actionable=(), scope=SimpleNamespace(config_bundle_digest="a" * 64),
        run_id="unused", observed_at=NOW, lease_owner="unused")


def test_status_labels_unassessed_evidence_and_keeps_review_body_private(held_thread, monkeypatch, capsys):
    from contextlib import nullcontext
    from localize.guardian import cli

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body="Sensitive complete review body.", in_reply_to_id=44)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply)),)
    assert ctx.controller.poll_once().failures == ()
    monkeypatch.setattr(cli, "_load_config_or_raise", lambda _: ctx.config)
    monkeypatch.setattr(cli, "_locked_existing_state", lambda _: nullcontext(ctx.state))
    assert cli.main(["status", "--config", str(ctx.tmp_path / "config.yaml"), "--human-evidence-limit", "1"]) == 0
    output = capsys.readouterr().out
    assert "thread evidence awaiting human review: 1" in output
    assert "4302557939" in output and "root_action_id" in output and "new_evidence" in output
    assert "Sensitive complete review body" not in output and "awaiting_human_review" in output


def test_operator_evidence_pagination_and_edited_deleted_revisions(held_thread):
    ctx = held_thread
    replies = tuple(replace(ctx.root, source_id=str(80+i), body="Unknown evidence.", in_reply_to_id=44) for i in range(3))
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, *replies)),)
    assert ctx.controller.poll_once().failures == ()
    total, first = ctx.state.thread_hold_evidence_for_operator(limit=2)
    assert total == 3 and len(first) == 2
    total, last = ctx.state.thread_hold_evidence_for_operator(limit=2, after_revision_id=first[-1]["revision_id"])
    assert total == 3 and len(last) == 1
    edited = replace(replies[0], body="New literal `changed`.", updated_at="2026-08-31T10:00:00Z")
    deleted = replace(replies[1], deleted=True)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, edited, deleted, replies[2])),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 1
    total, current = ctx.state.thread_hold_evidence_for_operator()
    assert total == 2 and {item["event_id"] for item in current} == {"80", "82"}
    assert all(item["classification"] == "new_evidence" for item in current)
    for args in ({"limit": 0}, {"limit": 101}, {"after_revision_id": -1}):
        with pytest.raises(ValueError):
            ctx.state.thread_hold_evidence_for_operator(**args)


def test_returning_to_an_exact_old_head_uses_latest_observation_not_revision_order(held_thread):
    from tests.unit.test_guardian_controller import _pull

    ctx = held_thread
    reply = replace(ctx.root, source_id="4302557939", body=FRESH_REPLY, in_reply_to_id=44)
    a = _snapshot(feedback=(ctx.root, reply))
    ctx.provider.snapshots = (a,)
    assert ctx.controller.poll_once().failures == ()
    original = next(item for item in ctx.state.latest_event_revisions() if item.event_id == reply.source_id)
    ctx.provider.snapshots = (_snapshot(feedback=(ctx.root, reply), pull=replace(_pull(), head_sha="d" * 40)),)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 2
    actions = ctx.state._connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
    stable = ctx.controller.poll_once()
    assert stable.failures == () and stable.runs_started == 0 and len(ctx.driver.calls) == 2
    assert ctx.state._connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == actions
    ctx.provider.snapshots = (a,)
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 2
    total, page = ctx.state.thread_hold_evidence_for_operator()
    assert total == 1 and page[0]["revision_id"] == original.revision_id
    assert ctx.state.thread_hold_for_revision(original.revision_id) is not None
    count = ctx.state._connection.execute("SELECT COUNT(*) FROM actions WHERE event_revision_id = ?",
                                         (original.revision_id,)).fetchone()[0]
    assert ctx.controller.poll_once().failures == () and len(ctx.driver.calls) == 2
    assert ctx.state._connection.execute("SELECT COUNT(*) FROM actions WHERE event_revision_id = ?",
                                        (original.revision_id,)).fetchone()[0] == count


def test_actual_failed_swahili_reply_uses_current_completed_root_without_backfill(tmp_path, runtime, monkeypatch):
    """Replay authentic public bytes and persisted decisions into a fresh offline DB."""
    from types import SimpleNamespace
    from localize.guardian.models import FeedbackEvent
    from tests.unit.test_guardian_controller import NOW, FakePreventionRunner
    import localize.guardian.controller as controller_module

    fixture = json.loads((Path(__file__).parent / "fixtures/guardian_5086_thread_hold.json").read_text())
    assert all(fixture["checks"].values()) and fixture["root_edited_history"]
    _, _, checkout, provider, broker, _ = runtime
    base_config = _config(GuardianMode.PROPOSE_PREVENTION)
    policy = replace(base_config.repositories[0], base_repo="bisq-network/bisq2", base_repo_id=372760852)
    config = replace(base_config, repositories=(policy,))
    seed = fixture["seed"]["details"]
    monkeypatch.setattr(controller_module, "_patch_policy_digest", lambda *_: seed["report_policy_digest"])
    def event(which):
        revision, public = fixture["current_" + which + "_revision"], fixture[which]
        return FeedbackEvent(repository=revision["repository"], pr_number=5086, kind="review_comment",
            event_id=revision["event_id"], author=revision["author"], author_id=revision["author_id"],
            author_type=revision["author_type"], body=public["body"], head_sha=revision["head_sha"],
            base_sha=revision["base_sha"], locale="sw", updated_at=revision["event_updated_at"],
            path=revision["path"], line=revision["line"], html_url=revision["html_url"])
    root, reply = event("root"), event("reply")
    with GuardianState(tmp_path / "offline.sqlite3") as state:
        root_revision = state.record_feedback_event(root, observed_at=NOW)
        reply_revision = state.record_feedback_event(reply, observed_at=NOW)
        assert root_revision.revision_hash == fixture["current_root_revision"]["revision_hash"]
        assert reply_revision.revision_hash == fixture["current_reply_revision"]["revision_hash"]
        for revision, original in ((root_revision, fixture["seed"]), (reply_revision, fixture["failed"])):
            run = state.start_run(repository=policy.base_repo, locale="sw", mode=config.mode, started_at=NOW)
            state.record_action(run_id=run, event_revision_id=revision.revision_id, action=config.mode.value,
                                status=original["status"], occurred_at=NOW, details=original["details"])
            state.finish_run(run, status=original["run_status"], summary="Sanitized persisted fixture.", finished_at=NOW)
        before = state._connection.execute("SELECT * FROM actions WHERE status = 'failed'").fetchone()
        assert state.acquire_lease(name="guardian:poll", owner="offline", ttl_seconds=60, now=NOW)
        controller = _controller(tmp_path=tmp_path, state=state, config=config, checkout=checkout,
                                 provider=provider, driver=InsufficientEvidenceDriver(), broker=broker,
                                 prevention_runner=FakePreventionRunner())
        snapshot = SimpleNamespace(feedback=(SimpleNamespace(kind=SimpleNamespace(value="review_comment"),
            source_id=root.event_id, in_reply_to_id=None), SimpleNamespace(kind=SimpleNamespace(value="review_comment"),
            source_id=reply.event_id, in_reply_to_id=int(root.event_id))))
        run = state.start_run(repository=policy.base_repo, locale="sw", mode=config.mode, started_at=NOW)
        held = controller._hold_thread_replies(policy=policy, snapshot=snapshot,
            current=((root, root_revision), (reply, reply_revision)), actionable=((reply, reply_revision),),
            scope=SimpleNamespace(config_bundle_digest=seed["report_config_digest"]), run_id=run,
            observed_at=NOW, lease_owner="offline")
        assert held == {reply_revision.revision_id}
        assert state.thread_hold_evidence_for_operator()[0] == 1
        assert not controller.codex_driver.calls and not broker.feedback_reports
        assert not controller.prevention_runner.propose_calls
        assert tuple(state._connection.execute("SELECT * FROM actions WHERE status = 'failed'").fetchone()) == tuple(before)
