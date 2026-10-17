"""Review coverage must survive bot conversations without gaining false proof."""

from dataclasses import replace
import json

import pytest

from localize.guardian.github import (
    CodeRabbitCoverageStatus as Status,
    FeedbackKind,
    FeedbackRevision,
    _coderabbit_coverage,
)


HEAD = "1" * 40
OLD = "2" * 40


def feedback(body, *, kind=FeedbackKind.ISSUE_COMMENT, day=1, commit=None):
    return FeedbackRevision(
        repository="acme/app", pull_number=1, kind=kind, source_id=str(day),
        node_id=None, author_login="coderabbitai[bot]", author_id=136622811,
        author_type="Bot", body=body, created_at=f"2026-10-{day:02d}T00:00:00Z",
        updated_at=f"2026-10-{day:02d}T00:00:00Z",
        html_url=f"https://github.com/acme/app/pull/1#issuecomment-{day}",
        commit_id=commit,
    )


def summary(*, head=HEAD, selected=2, skipped=0):
    coverage = json.dumps({"sourceCommitId": head, "coveredCommitId": head, "kind": "reviewed"})
    return (
        "<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\n"
        "<!-- recent_review_start -->\n"
        "No actionable comments were generated in the recent review. 🎉\n"
        f"<summary><strong>📒 Files selected for processing: {selected}</strong></summary>\n"
        f"<summary><strong>🚧 Files skipped from review as they are similar to previous changes: {skipped}</strong></summary>\n"
        "<!-- recent_review_end -->\n"
        f"<!-- final_review_risk_coverage:{coverage} -->"
    )


def test_resolved_reply_does_not_erase_rate_limit():
    limited = feedback("## Review limit reached\nReview was rate limited.")
    acknowledgement = feedback(
        "@translator The finding remains withdrawn. No correction is needed.\n\n"
        "✅ Review thread resolved.", kind=FeedbackKind.REVIEW_COMMENT, day=2,
        commit=HEAD,
    )
    result = _coderabbit_coverage([limited, acknowledgement], head_sha=HEAD)
    assert result.status is Status.RATE_LIMITED
    assert result.source_revision_id == limited.revision_id


@pytest.mark.parametrize("kind,body", [
    (FeedbackKind.REVIEW_COMMENT, "The finding is fixed."),
    (FeedbackKind.REVIEW, ""),
    (FeedbackKind.ISSUE_COMMENT, "Thanks for the clarification."),
])
def test_conversation_alone_is_not_review_coverage(kind, body):
    assert _coderabbit_coverage([feedback(body, kind=kind, commit=HEAD)], head_sha=HEAD).status is Status.ABSENT


@pytest.mark.parametrize("selected,skipped", [(2, 2), (1, 1), (90, 1), (0, 0)])
def test_skipped_or_empty_review_never_claims_complete_coverage(selected, skipped):
    assert _coderabbit_coverage([feedback(summary(selected=selected, skipped=skipped))], head_sha=HEAD).status is Status.SKIPPED


def test_ancillary_docstring_skip_does_not_hide_completed_review():
    body = summary(selected=90) + "\nDocstring Coverage: skipped; 90 files unsupported."
    assert _coderabbit_coverage([feedback(body)], head_sha=HEAD).status is Status.REVIEWED


@pytest.mark.parametrize("body", [
    summary(head=OLD),
    "## Walkthrough\nI reviewed the changed values.",
    "<!-- final_review_risk_coverage:{malformed} -->",
])
def test_missing_or_stale_commit_proof_does_not_claim_current_review(body):
    assert _coderabbit_coverage([feedback(body)], head_sha=HEAD).status is Status.ABSENT


def test_current_review_supersedes_previous_limit():
    limited = feedback("## Review limit reached")
    reviewed = feedback(summary(), day=2)
    result = _coderabbit_coverage([limited, reviewed], head_sha=HEAD)
    assert result.status is Status.REVIEWED
    assert result.source_revision_id == reviewed.revision_id


def test_newer_limit_overrides_older_completed_review():
    assert _coderabbit_coverage([
        feedback(summary()), feedback("## Review limit reached", day=2),
    ], head_sha=HEAD).status is Status.RATE_LIMITED


def test_substantive_review_binds_to_api_commit():
    body = "**Actionable comments posted: 1**\n\nCorrect the spelling."
    assert _coderabbit_coverage([feedback(body, kind=FeedbackKind.REVIEW, commit=HEAD)], head_sha=HEAD).status is Status.REVIEWED
    assert _coderabbit_coverage([feedback(body, kind=FeedbackKind.REVIEW, commit=OLD)], head_sha=HEAD).status is Status.ABSENT


def test_user_impersonation_and_deleted_review_are_not_proof():
    item = feedback(summary())
    for invalid in [replace(item, author_type="User"), replace(item, deleted=True)]:
        assert _coderabbit_coverage([invalid], head_sha=HEAD).status is Status.ABSENT


def test_updated_summary_without_proof_invalidates_previous_coverage():
    old = feedback(summary())
    newer = replace(old, body="<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\nReview in progress.", updated_at="2026-10-02T00:00:00Z")
    assert _coderabbit_coverage([old, newer], head_sha=HEAD).status is Status.ABSENT


@pytest.mark.parametrize("prefix", ["Here is the old summary format:\n", "## Walkthrough\n"])
def test_quoted_marker_is_not_review_proof(prefix):
    marker = summary().splitlines()[-1]
    for quoted in [f"```html\n{marker}\n```", f"> {marker}", f"~~~\n{marker}\n~~~"]:
        assert _coderabbit_coverage([feedback(prefix + quoted)], head_sha=HEAD).status is Status.ABSENT


def test_same_second_contradictory_coverage_is_conservative():
    reviewed = feedback(summary())
    for notice,expected in [("## Review limit reached", Status.RATE_LIMITED), ("## Review skipped", Status.SKIPPED)]:
        limit = replace(feedback(notice), source_id="2")
        for items in [[reviewed, limit], [limit, reviewed]]:
            assert _coderabbit_coverage(items, head_sha=HEAD).status is expected
