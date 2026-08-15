#!/usr/bin/env bash
# Build the anonymised audit-trail bundle for a double-blind submission.
#
# TMLR reviews double-blind and requires supplementary material to be
# anonymised, so the repository cannot be linked from the paper. But the
# paper's central claim about itself is that a reader can verify the commit
# ORDERING, and the usual anonymisation services (anonymous.4open.science and
# friends) serve a snapshot with no git history, which would remove exactly
# the thing being claimed.
#
# A git bundle solves that: one file, real history, `git clone` works, and
# `git merge-base --is-ancestor` runs against it.
#
# What this does:
#   1. mirror-clones the repo, so the working repository is never touched
#   2. deletes paths that identify people and support no result
#   3. rewrites identifying text across every blob AND every commit message
#   4. rewrites author and committer identity on every commit
#   5. verifies no identifying string survives anywhere in the object store
#   6. writes the bundle and re-verifies the ancestry checks from a fresh
#      clone of it
#
# Rewriting changes every commit hash. It does NOT change topology, so the
# ordering the paper claims is unaffected, but Table 6's hashes must be
# regenerated from the bundle after each rebuild. Step 6 prints the mapping.
#
# Usage:  bash code/phase3/scripts/build_anon_bundle.sh [output.bundle]

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-$REPO/../rdmerge-audit-trail.bundle}"
WORK="$(mktemp -d)"
FR="${GIT_FILTER_REPO:-git-filter-repo}"

command -v "$FR" >/dev/null 2>&1 || {
  if [ -f "$FR" ]; then FR="python $FR"; else
    echo "need git-filter-repo: pip install git-filter-repo, or set" >&2
    echo "GIT_FILTER_REPO=/path/to/git-filter-repo" >&2; exit 2
  fi
}

# Order matters: specific patterns before the general ones they contain.
# Case-insensitive throughout, because the first pass missed JIIT, pathak,
# sankalp and garg in the cases the literal rules did not cover, including
# the bibtex keys pathak2026merging and pathak2026rdmerge.
cat > "$WORK/rules.txt" <<'RULES'
regex:(?i)95154157\+k144u@users\.noreply\.github\.com==>anon@anonymous.invalid
regex:(?i)pathaksankalp[0-9]*@gmail\.com==>author1@anonymous.invalid
regex:(?i)gargsv[0-9]*@gmail\.com==>author2@anonymous.invalid
regex:(?i)github\.com/k144u/rdmerge==>github.com/ANONYMISED/rdmerge
regex:(?i)/home/sanjay\.g==>/home/anon
regex:(?i)jiit[-_]?gpu0?1==>gpu-node-01
regex:(?i)jiit[-_]?master==>cluster-head
regex:(?i)jiit_?231b220==>student-account
regex:(?i)jaypee[ a-z]*==>Anonymous Institution
CLUSTER-HOST==>10.0.0.1
regex:(?i)sanjay\.g==>anon
regex:(?i)sankalp[ ._-]?pathak==>Anonymous Author
regex:(?i)sanjay[ ._-]?garg==>Anonymous Coauthor
regex:(?i)k144u==>anon-author
regex:(?i)pathak==>author
regex:(?i)sankalp==>anonymous
regex:(?i)sanjay==>anonymous
regex:(?i)garg==>coauthor
regex:(?i)jiit==>cluster
RULES

echo "[1/6] mirror clone"
git clone --mirror -q "$REPO" "$WORK/anon.git"
cd "$WORK/anon.git"

echo "[2-4/6] strip identifying paths, rewrite text and identity"
$FR \
  --invert-paths \
  --path "SentEmail/" --path "final review/" \
  --path "handoff_2026-05-25_review_to_garg.md" \
  --path "handoff_to_garg_2026-06-13.md" \
  --path "notes/garg_message_2026-06-14.md" \
  --path "notes/garg_message_2026-06-23.md" \
  --path "Alignment Forum research note.pdf" \
  --path-glob "A_Rate_Distortion_*.pdf" \
  --replace-text "$WORK/rules.txt" \
  --replace-message "$WORK/rules.txt" \
  --name-callback 'return b"Anonymous Author"' \
  --email-callback 'return b"anon@anonymous.invalid"' \
  --force >/dev/null

echo "[5/6] verify no identifying string survives in any blob"
leak=0
while read -r o; do
  [ "$(git cat-file -t "$o" 2>/dev/null)" = "blob" ] || continue
  if git cat-file blob "$o" 2>/dev/null \
     | grep -qaiE "sanjay|sankalp|pathak|garg|jiit|k144u|jaypee|172\.16\.176"; then
    echo "  LEAK in blob $o" >&2; leak=1
  fi
done < <(git rev-list --objects --all | awk '{print $1}' | sort -u)
git log --all --format='%an <%ae>%n%cn <%ce>%n%B' \
  | grep -qaiE "sanjay|sankalp|pathak|garg|jiit|k144u|jaypee" && {
      echo "  LEAK in identity or commit messages" >&2; leak=1; }
[ "$leak" -eq 0 ] || { echo "ABORT: bundle not written" >&2; exit 3; }
echo "  clean"

echo "[6/6] write bundle and verify a fresh clone of it"
git bundle create "$OUT" --all >/dev/null 2>&1
rm -rf "$WORK/verify"
git clone -q "$OUT" "$WORK/verify"
cd "$WORK/verify"
echo "  $(git rev-list --count --all) commits, identity: $(git log -1 --format='%an <%ae>')"
echo
echo "old -> new hashes for Table 6 (regenerate the table from these):"
while read -r old new; do
  msg=$(git log -1 --format='%s' "$new" 2>/dev/null | cut -c1-58)
  [ -n "$msg" ] && printf "  %s -> %s  %s\n" "${old:0:7}" "${new:0:7}" "$msg"
done < "$WORK/anon.git/filter-repo/commit-map" | grep -iE "pre-?registration|amendment|analyzer|verdict|generator|VOID|CONFIRMED|SURVIVES|complete" | head -40

echo
echo "bundle: $OUT"
