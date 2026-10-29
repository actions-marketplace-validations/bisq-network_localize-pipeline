"""Exercise reporting through the real HTTP broker, including ambiguous writes."""

import json
import subprocess

import httpx
import pytest

from localize.guardian.github import (
    GitHubAPIError,
    GitHubAuthenticationError,
    PolicyViolation,
)
from tests.unit.test_guardian_github import (
    BASE_SHA,
    HEAD_SHA,
    EXPECTED_ACTOR,
    _json_response,
    _policy,
    _pr_payload,
    _repo_payload,
    _user_payload,
    _write_broker,
)


@pytest.fixture
def boundary(monkeypatch):
    state = {
        "comments": [],
        "writes": [],
        "lost_ack": False,
        "pull": _pr_payload(1),
        "actor": _user_payload(),
        "patch_error": None,
        "parent_pull": "https://api.github.test/repos/acme/app/pulls/1",
        "repository": _repo_payload(),
        "read_hook": None,
    }

    def handler(request):
        path, method = request.url.path, request.method
        if method == "GET" and state["read_hook"]:
            response = state["read_hook"](request, state)
            if response is not None:
                return response
        if path == "/user":
            return _json_response(request, state["actor"])
        if path == "/repos/acme/app":
            return _json_response(request, state["repository"])
        if path == "/repos/acme/app/pulls/1":
            return _json_response(request, state["pull"])
        if path == "/repos/acme/app/pulls/comments/12":
            return _json_response(
                request, {"id": 12, "pull_request_url": state["parent_pull"]}
            )
        if path.startswith("/repos/acme/app/pulls/comments/") and method == "GET":
            comment = next(c for c in state["comments"] if c["id"] == int(path.rsplit("/", 1)[1]))
            return _json_response(request, dict(comment))
        collections = {
            "/repos/acme/app/pulls/1/comments",
            "/repos/acme/app/issues/1/comments",
        }
        if path in collections and method == "GET":
            review = "/pulls/" in path
            return _json_response(
                request,
                [
                    c
                    for c in state["comments"]
                    if bool(c.get("in_reply_to_id")) == review
                ],
            )
        if path.startswith("/repos/acme/app/issues/comments/"):
            comment = next(
                c for c in state["comments"] if c["id"] == int(path.rsplit("/", 1)[1])
            )
            if method == "GET":
                return _json_response(request, dict(comment))
            assert method == "PATCH"
            if state["patch_error"] == "before":
                raise httpx.ReadTimeout("request lost before PATCH", request=request)
            comment["body"] = json.loads(request.content)["body"]
            state["writes"].append((method, path))
            if state["patch_error"] == "after":
                raise httpx.ReadTimeout("response lost after PATCH", request=request)
            return _json_response(request, dict(comment))
        if method == "POST" and path in {
            "/repos/acme/app/pulls/1/comments/12/replies",
            "/repos/acme/app/issues/1/comments",
        }:
            number = 100 + len(state["comments"])
            review = path.endswith("/replies")
            comment = {
                "id": number,
                "user": _user_payload(),
                "body": json.loads(request.content)["body"],
                "html_url": f"https://github.test/acme/app/pull/1#{'discussion_r' if review else 'issuecomment-'}{number}",
            }
            if review:
                comment["in_reply_to_id"] = 12
            state["comments"].append(comment)
            state["writes"].append((method, path))
            if state["lost_ack"]:
                state["lost_ack"] = False
                raise httpx.ReadTimeout("lost response", request=request)
            return _json_response(request, dict(comment), status=201)
        raise AssertionError(f"Unexpected write or endpoint: {method} {path}")

    monkeypatch.setattr(
        "localize.guardian.credentials.run_bounded_process",
        lambda *a, **k: subprocess.CompletedProcess(
            ["test-helper"], 0, stdout="test-token", stderr=""
        ),
    )
    broker = _write_broker(
        policy=_policy(),
        token_command=("test-helper",),
        base_url="https://api.github.test",
        transport=httpx.MockTransport(handler),
    )
    return broker, state


def arguments():
    return dict(
        pull_number=1,
        expected_head_sha=HEAD_SHA,
        expected_base_sha=BASE_SHA,
        expected_actor=EXPECTED_ACTOR,
        before_create=lambda: None,
        before_recover=lambda: None,
    )


def report_arguments():
    return dict(
        **arguments(),
        report_id="f" * 64,
        feedback_id="review_comment:12",
        details={
            "outcome": "needs_human",
            "report_reason": "glossary_conflict",
            "decision_required": True,
        },
    )


def test_thread_reply_recovers_lost_response_without_duplicate(boundary):
    broker, state = boundary
    state["lost_ack"] = True
    with pytest.raises(GitHubAPIError):
        broker.post_feedback_report(**report_arguments())
    result = broker.post_feedback_report(**report_arguments())
    assert not result.created
    assert len(state["writes"]) == 1
    assert "glossary" in result.body and "maintainer decision" in result.body
    assert "#discussion_r12" in result.body


@pytest.mark.parametrize("kind", ["report", "legacy", "summary"])
def test_exact_receipt_recovery_never_calls_source_checkout(boundary, kind):
    broker, state = boundary

    def post(callback):
        if kind == "summary":
            return broker.post_feedback_summary(**{**arguments(), "before_create": callback}, reports=[], previous_body=None)
        return broker.post_feedback_report(**{**report_arguments(), "before_create": callback})

    original = post(lambda: None)
    if kind == "legacy":
        from localize.guardian.reporting import legacy_report_body
        state["comments"][0]["body"] = ("<!-- localize-guardian:feedback:" + "f" * 64 + " -->\n"
            + legacy_report_body(report_arguments()["details"], repository="acme/app",
                                 feedback_id="review_comment:12", pull_number=1, web_base_url=broker.web_base_url))

    def forbidden_checkout():
        raise AssertionError("Read-only receipt recovery must not materialize source checkouts")

    recovered = post(forbidden_checkout)
    assert recovered.comment_id == original.comment_id and not recovered.created
    assert len(state["writes"]) == 1


@pytest.mark.parametrize("change", ["closed", "head", "base", "actor", "repository", "body", "writer", "type", "parent", "url", "identity", "deleted"])
def test_receipt_recovery_rechecks_after_collection_reads(boundary, change):
    """A matching list entry cannot bypass final authority or direct reread."""
    broker, state = boundary
    broker.post_feedback_report(**report_arguments())
    original_writes = list(state["writes"])

    def race(request, remote):
        if request.url.path == "/repos/acme/app/pulls/1/comments":
            page = json.loads(json.dumps(remote["comments"]))
            if change == "closed":
                remote["pull"]["state"] = "closed"
            elif change in {"head", "base"}:
                remote["pull"][change]["sha"] = "f" * 40
            elif change == "actor":
                remote["actor"] = _user_payload(actor_id=999)
            elif change == "repository":
                remote["repository"]["id"] = 999
            elif change == "body":
                remote["comments"][0]["body"] += " external annotation"
            elif change in {"writer", "type"}:
                remote["comments"][0]["user"].update({"id": 999} if change == "writer" else {"type": "Bot"})
            elif change == "parent":
                remote["comments"][0]["in_reply_to_id"] = 99
            elif change == "url":
                remote["comments"][0]["html_url"] = "https://evil.test/comment"
            return _json_response(request, page)
        if request.url.path == "/repos/acme/app/pulls/comments/100":
            if change == "deleted":
                return _json_response(request, {}, status=404)
            if change == "identity":
                return _json_response(request, {**remote["comments"][0], "id": 101,
                    "html_url": "https://github.test/acme/app/pull/1#discussion_r101"})

    state["read_hook"] = race
    def no_write():
        raise AssertionError("Receipt conflicts must not enter publication fallback")
    with pytest.raises((PolicyViolation, GitHubAuthenticationError, GitHubAPIError)):
        broker.post_feedback_report(**{**report_arguments(), "before_create": no_write})
    assert state["writes"] == original_writes


@pytest.mark.parametrize("failure", ["duplicate_later_page", "unsafe_link", "repeated_link", "page_bound", "page_http_error"])
def test_receipt_recovery_requires_complete_safe_pagination(boundary, monkeypatch, failure):
    broker, state = boundary
    broker.post_feedback_report(**report_arguments())
    if failure == "page_bound":
        monkeypatch.setattr("localize.guardian.github._MAX_PAGINATION_PAGES", 1)

    def pages(request, remote):
        if request.url.path != "/repos/acme/app/pulls/1/comments":
            return None
        if request.url.params.get("page") == "2":
            if failure == "page_http_error":
                return _json_response(request, {}, status=503)
            return _json_response(request, [{**remote["comments"][0], "id": 101,
                "html_url": "https://github.test/acme/app/pull/1#discussion_r101"}])
        link = "https://api.github.test/repos/acme/app/pulls/1/comments?page=2"
        if failure == "unsafe_link":
            link = "https://evil.test/comments?page=2"
        elif failure == "repeated_link":
            link = "https://api.github.test/repos/acme/app/pulls/1/comments"
        return _json_response(request, remote["comments"], headers={"Link": f'<{link}>; rel="next"'})

    state["read_hook"] = pages
    with pytest.raises((PolicyViolation, GitHubAPIError)):
        broker.post_feedback_report(**report_arguments())
    assert len(state["writes"]) == 1


@pytest.mark.parametrize("kind", ["report", "legacy", "summary"])
def test_receipt_recovery_requires_local_intent_and_lease_callback(boundary, kind):
    broker, state = boundary
    def post(**overrides):
        if kind == "summary":
            return broker.post_feedback_summary(**{**arguments(), **overrides}, reports=[], previous_body=None)
        return broker.post_feedback_report(**{**report_arguments(), **overrides})
    post()
    if kind == "legacy":
        from localize.guardian.reporting import legacy_report_body
        state["comments"][0]["body"] = ("<!-- localize-guardian:feedback:" + "f" * 64 + " -->\n"
            + legacy_report_body(report_arguments()["details"], repository="acme/app", feedback_id="review_comment:12", pull_number=1, web_base_url=broker.web_base_url))
    def invalid_intent():
        raise PolicyViolation("Durable local source or lease is no longer valid")
    with pytest.raises(PolicyViolation):
        post(before_recover=invalid_intent)
    assert len(state["writes"]) == 1


@pytest.mark.parametrize("route", ["post", "patch"])
def test_receipt_fast_path_preserves_full_write_callback_protocol(boundary, route):
    broker, state = boundary
    if route == "patch":
        prior = broker.post_feedback_summary(**arguments(), reports=[], previous_body=None)
        reports = [{"url": "https://github.test/acme/app/pull/1#discussion_r1", "disposition": "needs_human", "decision_required": True}]
    calls = []
    def full_source():
        calls.append(len(state["writes"]))
    if route == "post":
        broker.post_feedback_report(**{**report_arguments(), "before_create": full_source})
        assert calls == [0, 0]
    else:
        broker.post_feedback_summary(**{**arguments(), "before_create": full_source}, reports=reports, previous_body=prior.body)
        assert calls == [1, 1, 1]
    assert state["writes"][-1][0] == ("POST" if route == "post" else "PATCH")


@pytest.mark.parametrize("change", ["closed", "head", "base", "actor", "thread"])
def test_report_revalidates_all_write_authority(boundary, change):
    broker, state = boundary
    if change == "closed":
        state["pull"]["state"] = "closed"
    elif change in {"head", "base"}:
        state["pull"][change]["sha"] = "f" * 40
    elif change == "actor":
        state["actor"] = _user_payload(actor_id=999)
    else:
        state["parent_pull"] = "https://api.github.test/repos/acme/app/pulls/2"
    with pytest.raises((PolicyViolation, GitHubAuthenticationError)):
        broker.post_feedback_report(**report_arguments())
    assert not state["writes"]


def test_edited_feedback_callback_stops_post_after_remote_reads(boundary):
    broker, state = boundary
    calls = []

    def changed():
        calls.append(True)
        if len(calls) == 2:
            raise PolicyViolation("feedback was deleted or edited")

    with pytest.raises(PolicyViolation):
        broker.post_feedback_report(**{**report_arguments(), "before_create": changed})
    assert not state["writes"]


def test_summary_is_single_managed_comment_and_preserves_external_edits(boundary):
    broker, state = boundary
    report = broker.post_feedback_report(**report_arguments())
    reports = [
        {
            "url": report.html_url,
            "disposition": "needs_human",
            "decision_required": True,
        }
    ]
    first = broker.post_feedback_summary(
        **arguments(), reports=reports, previous_body=None
    )
    again = broker.post_feedback_summary(
        **arguments(), reports=reports, previous_body=None
    )
    assert not again.created
    reports[0]["disposition"] = "applied_alternative"
    updated = broker.post_feedback_summary(
        **arguments(), reports=reports, previous_body=first.body
    )
    assert updated.comment_id == first.comment_id
    assert not updated.created
    assert "maintainer decision still required" in updated.body
    state["comments"][-1]["body"] += "\nHuman annotation"
    reports[0]["disposition"] = "needs_human"
    with pytest.raises(PolicyViolation):
        broker.post_feedback_summary(
            **arguments(), reports=reports, previous_body=updated.body
        )
    assert state["comments"][-1]["body"].endswith("Human annotation")
    assert len(state["comments"]) == 2


@pytest.mark.parametrize("route,spoof", [
    ("summary", "actor"), ("summary", "type"),
    ("thread", "actor"), ("thread", "type"), ("thread", "parent"),
])
def test_copied_markers_outside_writer_and_route_do_not_block_reports(boundary, route, spoof):
    broker, state = boundary
    def post():
        if route == "summary":
            return broker.post_feedback_summary(**arguments(), reports=[], previous_body=None)
        return broker.post_feedback_report(**report_arguments())
    first = post()
    legitimate = dict(state["comments"][0])
    rogue = {**legitimate, "id": 987}
    if spoof == "parent":
        rogue["in_reply_to_id"] = 99
    else:
        rogue["user"] = {**rogue["user"], **({"id": 999} if spoof == "actor" else {"type": "Bot"})}
    state["comments"][:] = [rogue]
    replacement = post()
    assert replacement.created
    assert replacement.comment_id != first.comment_id
    recovered = post()
    assert not recovered.created
    assert recovered.comment_id == replacement.comment_id
    assert state["comments"][0] == rogue


def test_review_summary_uses_linked_issue_comment(boundary):
    broker, state = boundary
    result = broker.post_feedback_report(
        **{**report_arguments(), "feedback_id": "review:12"}
    )
    assert "#pullrequestreview-12" in result.body
    assert state["writes"] == [("POST", "/repos/acme/app/issues/1/comments")]


@pytest.mark.parametrize("failure", ["before", "after"])
def test_summary_retries_known_exact_versions_after_ambiguous_patch(boundary, failure):
    from localize.guardian.reporting import summary_body

    broker, state = boundary
    reports = [
        {
            "url": "https://github.test/acme/app/pull/1#discussion_r100",
            "disposition": "needs_human",
            "decision_required": True,
        }
    ]
    first = broker.post_feedback_summary(
        **arguments(), reports=reports, previous_body=None
    )
    reports[0]["disposition"] = "applied_alternative"
    candidate = summary_body(
        reports, repository="acme/app", pull_number=1, web_base_url=broker.web_base_url
    )
    state["patch_error"] = failure
    with pytest.raises(GitHubAPIError):
        broker.post_feedback_summary(
            **arguments(), reports=reports, previous_body=(first.body, candidate)
        )
    state["patch_error"] = None
    result = broker.post_feedback_summary(
        **arguments(), reports=reports, previous_body=(first.body, candidate)
    )
    assert result.body == candidate and result.comment_id == first.comment_id
    assert len(state["comments"]) == 1


def commit_arguments():
    return {**{key: value for key, value in arguments().items() if key != "before_recover"},
            "commit_sha": HEAD_SHA, "action_id": "private-action-123",
            "event_revision_id": "private-event-456"}


@pytest.mark.parametrize("kind", ["commit", "report", "summary"])
def test_public_writes_contain_only_human_readable_content(boundary, kind):
    broker, state = boundary
    if kind == "commit":
        result = broker.post_commit_reply(**commit_arguments())
    elif kind == "report":
        result = broker.post_feedback_report(**report_arguments())
        assert f"/commit/{HEAD_SHA}" in result.body
        assert "Assessed revision" in result.body
    else:
        result = broker.post_feedback_summary(**arguments(), reports=[], previous_body=None)
    assert result.body.startswith("🤖 **Localize Guardian")
    for private in ("<!--", "private-action-123", "private-event-456", "f" * 64, "review_comment:12"):
        assert private not in result.body
    assert len(state["writes"]) == 1


@pytest.mark.parametrize("kind", ["commit", "report"])
def test_legacy_comments_are_recovered_without_republishing_metadata(boundary, kind):
    broker, state = boundary
    if kind == "commit":
        broker.post_commit_reply(**commit_arguments())
        marker = "<!-- localize-guardian:v1 action=private-action-123 event=private-event-456 -->"
        legacy_body = (marker + "\n🤖 **Localize Guardian:** Applied a validated translation-only "
                       f"correction in [`{HEAD_SHA[:12]}`](https://github.test/translator-bot/app/commit/{HEAD_SHA}). "
                       "The review thread remains open for reviewer confirmation.")
        def post():
            return broker.post_commit_reply(**commit_arguments())
    else:
        from localize.guardian.reporting import legacy_report_body
        broker.post_feedback_report(**report_arguments())
        legacy_body = ("<!-- localize-guardian:feedback:" + "f" * 64 + " -->\n"
                       + legacy_report_body(report_arguments()["details"], repository="acme/app",
                                            feedback_id="review_comment:12", pull_number=1,
                                            web_base_url=broker.web_base_url))
        def post():
            return broker.post_feedback_report(**report_arguments())
    state["comments"][0]["body"] = legacy_body
    recovered = post()
    assert recovered.body == legacy_body and not recovered.created
    assert len(state["writes"]) == 1


@pytest.mark.parametrize("tamper", ["edited", "heading", "duplicate"])
def test_readable_report_identity_preserves_conflict_guards(boundary, tamper):
    broker, state = boundary
    broker.post_feedback_report(**report_arguments())
    if tamper == "edited":
        state["comments"][0]["body"] += "\nAn external annotation"
    elif tamper == "heading":
        state["comments"][0]["body"] = state["comments"][0]["body"].replace(
            "🤖 **Localize Guardian — Maintainer decision needed**", "Edited heading")
    else:
        state["comments"].append({**state["comments"][0], "id": 101,
                                  "html_url": "https://github.test/acme/app/pull/1#discussion_r101"})
    with pytest.raises(PolicyViolation):
        broker.post_feedback_report(**report_arguments())
    assert len(state["writes"]) == 1


def test_legacy_summary_migrates_only_from_exact_known_body(boundary):
    broker, state = boundary
    first = broker.post_feedback_summary(**arguments(), reports=[], previous_body=None)
    legacy_body = "<!-- localize-guardian:feedback-summary:v1 -->\n" + first.body
    state["comments"][0]["body"] = legacy_body
    with pytest.raises(PolicyViolation, match="edited externally"):
        broker.post_feedback_summary(**arguments(), reports=[], previous_body=first.body)
    updated = broker.post_feedback_summary(**arguments(), reports=[], previous_body=legacy_body)
    assert updated.comment_id == first.comment_id and not updated.created
    assert "<!--" not in updated.body
    assert len(state["comments"]) == 1


def test_legacy_report_key_migration_recovers_existing_comment(boundary):
    from localize.guardian.reporting import legacy_report_body, report_body, report_key
    broker, state = boundary
    kwargs = report_arguments()
    old_text = legacy_report_body(kwargs["details"], repository="acme/app",
                                  feedback_id=kwargs["feedback_id"], pull_number=1,
                                  web_base_url=broker.web_base_url)
    old_key = report_key(repository_id=broker.policy.repository_id, pull_number=1,
                         feedback_id=kwargs["feedback_id"], body=old_text)
    new_text = report_body(kwargs["details"], repository="acme/app",
                           feedback_id=kwargs["feedback_id"], pull_number=1,
                           web_base_url=broker.web_base_url)
    new_key = report_key(repository_id=broker.policy.repository_id, pull_number=1,
                         feedback_id=kwargs["feedback_id"], body=new_text)
    assert old_key != new_key
    state["comments"].append({
        "id": 100, "user": _user_payload(), "in_reply_to_id": 12,
        "html_url": "https://github.test/acme/app/pull/1#discussion_r100",
        "body": f"<!-- localize-guardian:feedback:{old_key} -->\n{old_text}",
    })
    recovered = broker.post_feedback_report(**{**kwargs, "report_id": new_key})
    assert not recovered.created and recovered.comment_id == 100
    assert state["writes"] == []


def test_summary_is_brief_without_hiding_counts_or_hold_decisions(boundary):
    broker, state = boundary
    reports = [{"url": f"https://github.test/acme/app/pull/1#discussion_r{100 + index}",
                "disposition": "applied"} for index in range(100)]
    reports[-1].update(disposition="needs_human", decision_required=True)
    result = broker.post_feedback_summary(**arguments(), reports=reports, previous_body=None)
    assert "Please review: 99 applied; 1 needs human" in result.body
    assert "maintainer decision still required" in result.body
    assert "#discussion_r199" in result.body
    assert result.body.count("[Details ") == 3
    assert "97 more outcomes included in the counts" in result.body
    assert len(result.body) < 600


def test_summary_for_completed_noops_requests_no_action(boundary):
    broker, _ = boundary
    result = broker.post_feedback_summary(**arguments(), previous_body=None, reports=[
        {"url": "https://github.test/acme/app/pull/1#discussion_r100", "disposition": "already_addressed"},
    ])
    assert "No action needed" in result.body and "Please review" not in result.body


def test_compact_summary_validates_even_unlinked_outcomes(boundary):
    broker, state = boundary
    reports = [{"url": f"https://github.test/acme/app/pull/1#discussion_r{100 + index}",
                "disposition": "applied"} for index in range(4)]
    reports[-1]["url"] = "https://attacker.invalid/"
    with pytest.raises(ValueError, match="Invalid feedback summary entry"):
        broker.post_feedback_summary(**arguments(), reports=reports, previous_body=None)
    assert state["writes"] == []
