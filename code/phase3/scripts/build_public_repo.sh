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
#   2. Build artifacts that bloat history (two 19 MB bundles and the zips).
#   3. Every reference to the concurrent a concurrent anonymous submission, which is still under
#      double-blind review elsewhere. Its forum id and title sit in
#      paper/references.bib and tmlr_submission/overleaf/references.bib in two
#      commits. Publishing them under our own names would identify its authors
#      to its own reviewers, which is the exact thing removing the citation
#      from the paper was meant to prevent. Removing it from HEAD is not
#      enough; git history is the whole point of this repository.
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

# Only the 9930 scrub. No identity rules: the names stay.
# The title is broken in two places rather than matched whole, because it is
# line-wrapped in the bib entry and a single pattern would have to span the
# newline. Breaking either half is enough to make it unsearchable.
cat > "$WORK/rules.txt" <<'RULES'
REDACTED-FORUM-ID==>REDACTED-FORUM-ID
REDACTED==>REDACTED
REDACTED==>REDACTED
a concurrent anonymous submission==>a concurrent anonymous submission
concurrentanon==>concurrentanon
RULES

echo "[1/5] mirror clone"
rm -rf "$OUT"
git clone --mirror -q "$REPO" "$OUT"
cd "$OUT"

echo "[2/5] drop private paths and build artifacts, scrub 9930"
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
  --replace-text "$WORK/rules.txt" \
  --replace-message "$WORK/rules.txt" \
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
         "notes/garg_message_2026-06-23.md" "POST_ACCEPTANCE_TODO.md"; do
  if git log --all --oneline -- "$p" | grep -q .; then
    echo "  LEAK: $p still reachable in history" >&2; gone=1
  fi
done
[ "$gone" -eq 0 ] || { echo "ABORT" >&2; exit 3; }
echo "  clean"

echo "[4/5] verify no reference to the concurrent submission survives"
# Same self-reference trap as the anonymiser: this script is itself in the
# repository it scans, so every term below must be one the rules above rewrite.
# That is why the bare number 9930 is NOT checked: it occurs legitimately as a
# float in dozens of result JSONs, and a rule for it would corrupt data.
PAT='REDACTED-FORUM-ID|REDACTED|REDACTED|concurrentanon'
leak=0
while read -r o; do
  [ "$(git cat-file -t "$o" 2>/dev/null)" = "blob" ] || continue
  if git cat-file blob "$o" 2>/dev/null | grep -qaE "$PAT"; then
    echo "  LEAK in blob $o" >&2; leak=1
  fi
done < <(git rev-list --objects --all | awk '{print $1}' | sort -u)
git log --all --format='%B' | grep -qaE "$PAT" && {
    echo "  LEAK in a commit message" >&2; leak=1; }
[ "$leak" -eq 0 ] || { echo "ABORT: not pushed" >&2; exit 3; }
echo "  clean"

echo "[5/5] result"
echo "  commits:  $(git rev-list --count --all)"
echo "  identity: $(git log -1 --format='%an <%ae>')"
echo "  size:     $(du -sm "$OUT" | cut -f1) MB"
echo "  map:      $OUT/filter-repo/commit-map"
echo
echo "To publish (this script deliberately does not):"
echo "  git -C $OUT push --mirror \\"
echo "    https://github.com/K144U/Initialisation-Sets-the-Conditioning-of-LoRA-Merging.git"
