# Project policy profiles

These profiles contain project-specific translation policy. Select the appropriate
profile for the translator or review and apply its rules to the installation's
chosen operator policy. Editing a reference profile does not update a separate
private operator snapshot.

The `known-typo`, `known-stale`, `known-saved`, and `known-unlock` rules reject
specific previously reviewed translations only when the key, locale, English
source, and defective target all match. They do not validate arbitrary wording
or prove that every privacy promise, warning state, or action is preserved.
Changed English wording stops those rules from applying and needs fresh review.
The PCM spelling rules are an optional defense alongside the accepted-correction
and translation-memory handling, not a replacement for it.

Pinned originals and corrections are recorded in
[`bisq_review_regressions.json`](../tests/fixtures/project_policy/bisq_review_regressions.json).
Constructed alternative-wording and URL/name controls are labelled separately
from observed corrections; they are not claims of native-speaker approval.

Rules do not enable additional translation locales. The Mobile reference profile
currently selects neither `pcm` nor `hi`; its PCM and Hindi rules become active
only in an installation that explicitly selects those locales. Keep the selected
locale list and deployed policy snapshot aligned when applying a proposal.
