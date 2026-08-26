#!/usr/bin/env bash
# Build the filtered history for the PUBLIC GitHub repository.
#
# Sibling of build_anon_bundle.sh and deliberately not the same thing. That
# script exists to hide who we are, for a double-blind venue. This one does
# not: JAIR is single-blind, the paper carries our names, and the whole point
# of the public repo is that a reader can check the commit trail against a real
# URL. So identity, dates and timezones are all left alone here.
#
# What it still removes, because anonymity was never the only reason those
# paths were excluded:
#
#   1. Private correspondence and external review notes. Nobody consented to
#      those being published, and a public push is permanent: GitHub caches
#      unreachable objects and anyone can have cloned before a later fix.
#      The JAIR cover letter counts: it is a message to an editor, and it
#      names the concurrent submission outright.
#   2. Build artifacts that bloat history (two 19 MB bundles and the zips).
#   3. Every reference to the concurrent submission that is still under
#      double-blind review elsewhere. Its forum id and title sit in
#      paper/references.bib and tmlr_submission/overleaf/references.bib in two
#      commits. Publishing them under our own names would identify its authors
#      to its own reviewers, which is the exact thing removing the citation
#      from the paper was meant to prevent. Removing it from HEAD is not
#      enough; git history is the whole point of this repository.
#      HANDOFF_2026-08-16.md is dropped entirely rather than scrubbed: it
#      discusses that submission's shared authorship repeatedly, and
#      whack-a-mole on a file whose subject is the thing you are hiding is how
#      leaks survive.
#   4. The internal cluster IP, and a shared student account belonging to
#      someone who is not an author. Note (b) below for what is deliberately
#      left alone: the bare hostname stays, because scrubbing it would rewrite
#      the PBS job ids in the result files.
#
# Filtering is DETERMINISTIC. Re-running this after adding commits gives the
# already-published commits the same hashes, so the next push is a
# fast-forward rather than a rewrite, and the hashes in the paper's commit
# table stay valid.
#
# This script does NOT push. It builds and verifies, then prints the command.
#
# Usage:  bash code/phase3/scripts/build_public_repo.sh [outdir]

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-$REPO/../rdmerge-public.git}"
WORK="$(mktemp -d)"

FR_PATH="${GIT_FILTER_REPO:-git-filter-repo}"
if command -v "$FR_PATH" >/dev/null 2>&1; then
  FR=("$FR_PATH")
elif [ -f "$FR_PATH" ]; then
  FR=(python "$FR_PATH")
else
  echo "need git-filter-repo: pip install git-filter-repo, or set" >&2
  echo "GIT_FILTER_REPO=/path/to/git-filter-repo" >&2; exit 2
fi

# No AUTHOR identity rules: the names stay, that is the point of this repo.
# But the anonymiser's text rules were not all about anonymity, and dropping
# them wholesale was a mistake this script made once already. Two kinds remain:
#
#   a. The concurrent submission. The title is broken in two places rather than
#      matched whole, because it is line-wrapped in the bib entry and a single
#      pattern would have to span the newline. Breaking either half is enough
#      to make it unsearchable. The topic phrase is a REGEX rule, not a
#      literal one: --replace-text is case-sensitive by default, and three
#      different casings of it existed in history. Two builds leaked on a
#      casing a literal rule missed, so the rule now matches all of them
#      rather than growing one line per variant discovered.
#   b. Infrastructure, but only where it costs nothing. The internal IP and the
#      shared student account go: the account belongs to a third party who is
#      not an author and never consented to appear here, and the IP is RFC1918
#      so it is useless to a reader and needless exposure otherwise. Both occur
#      only in prose and scripts, never in data.
#
#      The bare hostname jiit-master is deliberately NOT scrubbed. It is
#      embedded in every PBS job id in roughly 200 result files
#      ("pbs_jobid": "42592.jiit-master"), so a rule for it would rewrite the
#      experimental record itself, and it discloses nothing: the institution is
#      already on the paper's title page. Corrupting provenance data to hide a
#      hostname the reader can infer from the author list is a bad trade.
#
# Note the self-reference: this script lives in the repository it filters, so
# the literal strings below get rewritten inside the published copy of this
# file. That is why every term in the step-4 scan must be one of these LHSs.
cat > "$WORK/rules.txt" <<'RULES'
REDACTED-FORUM-ID==>REDACTED-FORUM-ID
regex:(?i)REDACTED==>REDACTED
REDACTED==>REDACTED
regex:(?i)(tmlr |openreview )?a concurrent anonymous submission==>a concurrent anonymous submission
concurrentanon==>concurrentanon
CLUSTER-HOST==>CLUSTER-HOST
STUDENT-ACCOUNT==>STUDENT-ACCOUNT
RULES

# One author, spelled four ways. History carries the GitHub noreply address,
# the real address, one typo of it, and the cluster account: per CLAUDE.md that
# account has no git identity of its own, so commits made there fell back to
# the login name, which is why four 2026-05-20 paper commits look like someone
# else made them. They are all the same person, and the cluster form also
# carries the hostname inside the email.
cat > "$WORK/mailmap.txt" <<'MAILMAP'
K144U <pathaksankalp04@gmail.com> <95154157+K144U@users.noreply.github.com>
K144U <pathaksankalp04@gmail.com> <pathaksankalp@gmail.com>
K144U <pathaksankalp04@gmail.com> <sanjay.g@jiit-master.cm.cluster>
MAILMAP

# Assistant trailers. GitHub parses Co-Authored-By and lists the address in it
# as a repository CONTRIBUTOR, which is the whole reason an @claude account
# showed up on the contributor list next to the author. Strip the trailer and
# the session URL, which is a live link into a private transcript.
#
# Only those two exact line forms. Prose mentioning CLAUDE.md must survive:
# that is a real tracked file here, and roughly ten commit subjects describe
# editing it. A blanket rule on the word would rewrite the project's own
# history of its own documentation.
#
# The message rules are a SUPERSET of the text rules, not a replacement: the
# scrubbing above has to apply to commit messages too, and passing
# --replace-message a file that omitted them would silently un-scrub them.
cp "$WORK/rules.txt" "$WORK/msgrules.txt"
cat >> "$WORK/msgrules.txt" <<'MSGRULES'
regex:(?m)^Co-Authored-By: Claude.*\n?==>
regex:(?m)^Claude-Session: .*\n?==>
MSGRULES

echo "[1/5] mirror clone"
rm -rf "$OUT"
git clone --mirror -q "$REPO" "$OUT"
cd "$OUT"

echo "[2/5] drop private paths and build artifacts, scrub the other submission"
"${FR[@]}" \
  --invert-paths \
  --path "SentEmail/" --path "final review/" \
  --path "handoff_2026-05-25_review_to_garg.md" \
  --path "handoff_to_garg_2026-06-13.md" \
  --path "notes/garg_message_2026-06-14.md" \
  --path "notes/garg_message_2026-06-23.md" \
  --path "Alignment Forum research note.pdf" \
  --path-glob "A_Rate_Distortion_*.pdf" \
  --path "POST_ACCEPTANCE_TODO.md" \
  --path "tmlr_submission/supplementary/audit-trail.bundle" \
  --path "tmlr_submission/supplementary.zip" \
  --path "tmlr_submission/overleaf_tmlr.zip" \
  --path "tmlr_submission/paper.pdf" \
  --path "JAIR/overleaf_jair.zip" \
  --path "JAIR/COVER_LETTER.md" \
  --path "HANDOFF_2026-08-16.md" \
  --replace-text "$WORK/rules.txt" \
  --replace-message "$WORK/msgrules.txt" \
  --mailmap "$WORK/mailmap.txt" \
  --force >/dev/null

echo "[3/5] verify the private paths are gone from every commit"
# Every check below this line is `git ... | grep -q`, so a git that ERRORS
# produces no output and the check reports "clean". That is the worst possible
# failure mode for a script whose output authorises a permanent public push,
# and it is reachable: filter-repo packs every ref, and if it leaves $GIT_DIR
# without a refs/ directory git refuses the repository outright. So assert the
# repo is readable and non-empty before believing anything the checks say.
mkdir -p refs
git rev-parse --git-dir >/dev/null 2>&1 || {
  echo "  ABORT: $OUT is not a readable git repo; checks would be vacuous" >&2
  exit 3; }
ncommits="$(git rev-list --count --all 2>/dev/null || echo 0)"
[ "$ncommits" -gt 0 ] 2>/dev/null || {
  echo "  ABORT: no commits reachable; checks would be vacuous" >&2; exit 3; }
echo "  repo readable, $ncommits commits"
gone=0
for p in "SentEmail" "final review" "handoff_2026-05-25_review_to_garg.md" \
         "handoff_to_garg_2026-06-13.md" "notes/garg_message_2026-06-14.md" \
         "notes/garg_message_2026-06-23.md" "POST_ACCEPTANCE_TODO.md" \
         "JAIR/COVER_LETTER.md" "HANDOFF_2026-08-16.md"; do
  if git log --all --oneline -- "$p" | grep -q .; then
    echo "  LEAK: $p still reachable in history" >&2; gone=1
  fi
done
[ "$gone" -eq 0 ] || { echo "ABORT" >&2; exit 3; }
echo "  clean"

echo "[4/5] verify no concurrent-submission or infrastructure leak survives"
# Same self-reference trap as the anonymiser: this script is itself in the
# repository it scans, so every term below must be one the rules above rewrite,
# or the scan reports a leak against its own source.
#
# The other submission's bare number is still NOT checked: it occurs by
# coincidence as a float in dozens of result JSONs, and a rule for it would
# corrupt data. The check is bounded to the phrase form instead, which no
# float can produce. The number is not written anywhere in this file either:
# a comment explaining what we scrub, next to the number itself, hands a
# reader the lookup key that the scrubbing exists to withhold.
# Matching is case-insensitive on purpose: the leak that got through the first
# build was a lowercase "REDACTED" against a capitalised rule, so an
# uncovered case variant must fail the build rather than pass it.
PAT='REDACTED-FORUM-ID|REDACTED|REDACTED|concurrentanon'
PAT="$PAT"'|a concurrent anonymous submission|172[.]16[.]176[.]120|STUDENT-ACCOUNT'
leak=0
while read -r o; do
  [ "$(git cat-file -t "$o" 2>/dev/null)" = "blob" ] || continue
  if git cat-file blob "$o" 2>/dev/null | grep -qaiE "$PAT"; then
    echo "  LEAK in blob $o ($(git rev-list --objects --all | grep "^$o " | cut -d' ' -f2-))" >&2
    leak=1
  fi
done < <(git rev-list --objects --all | awk '{print $1}' | sort -u)
git log --all --format='%B' | grep -qaiE "$PAT" && {
    echo "  LEAK in a commit message" >&2; leak=1; }
[ "$leak" -eq 0 ] || { echo "ABORT: not pushed" >&2; exit 3; }
echo "  clean"

echo "[4b/5] verify one identity and no assistant trailers"
# GitHub builds the contributor list from BOTH the author field and the
# Co-Authored-By trailers, so either one alone is enough to put a face on the
# repository that should not be there. Check both, and check every commit
# rather than the tip: `git log -1` reports the most recent identity and says
# nothing about the other 228.
ids="$(git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u)"
if [ "$(printf '%s\n' "$ids" | grep -c .)" -ne 1 ]; then
  echo "  LEAK: more than one identity in history:" >&2
  printf '    %s\n' $(printf '%s\n' "$ids") >&2
  echo "ABORT: not pushed" >&2; exit 3
fi
echo "  identity: $ids (all $ncommits commits, author and committer)"
if git log --all --format='%B' | grep -qaiE '^(Co-Authored-By|Claude-Session):'; then
  echo "  LEAK: an assistant trailer survives in a commit message" >&2
  echo "ABORT: not pushed" >&2; exit 3
fi
echo "  no assistant trailers"

echo "[5/5] result"
echo "  commits:  $(git rev-list --count --all)"
echo "  identity: $(git log -1 --format='%an <%ae>')"
echo "  size:     $(du -sm "$OUT" | cut -f1) MB"
echo "  map:      $OUT/filter-repo/commit-map"
echo
echo "To publish (this script deliberately does not):"
echo "  git -C $OUT push --mirror \\"
echo "    https://github.com/K144U/Initialisation-Sets-the-Conditioning-of-LoRA-Merging.git"
