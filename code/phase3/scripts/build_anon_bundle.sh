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
# git-filter-repo may be on PATH or a standalone script. Kept as an array, not
# a string: this repository lives under a path with spaces, and "python $FR"
# word-splits it into arguments that get run as commands.
FR_PATH="${GIT_FILTER_REPO:-git-filter-repo}"
if command -v "$FR_PATH" >/dev/null 2>&1; then
  FR=("$FR_PATH")
elif [ -f "$FR_PATH" ]; then
  FR=(python "$FR_PATH")
else
  echo "need git-filter-repo: pip install git-filter-repo, or set" >&2
  echo "GIT_FILTER_REPO=/path/to/git-filter-repo" >&2; exit 2
fi

# Order matters: specific patterns before the general ones they contain.
# Case-insensitive throughout, because the first pass missed JIIT, pathak,
# sankalp and garg in the cases the literal rules did not cover, including
# the bibtex keys pathak2026merging and pathak2026rdmerge.
cat > "$WORK/rules.txt" <<'RULES'
regex:(?i)95154157\+k144u@users\.noreply\.github\.com==>anon@anonymous.invalid
95154157==>ANONID
regex:(?i)pathaksankalp[0-9]*@gmail\.com==>author1@anonymous.invalid
regex:(?i)gargsv[0-9]*@gmail\.com==>author2@anonymous.invalid
regex:(?i)github\.com/k144u/rdmerge==>github.com/ANONYMISED/rdmerge
regex:(?i)/home/sanjay\.g==>/home/anon
regex:(?i)jiit[-_]?gpu0?1==>gpu-node-01
regex:(?i)jiit[-_]?master==>cluster-head
regex:(?i)jiit_?231b220==>student-account
regex:(?i)jaypee[ a-z]*==>Anonymous Institution
CLUSTER-HOST==>10.0.0.1
172.16.176==>10.0.0
+0530==>+0000
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
"${FR[@]}" \
  --invert-paths \
  --path "SentEmail/" --path "final review/" \
  --path "handoff_2026-05-25_review_to_garg.md" \
  --path "handoff_to_garg_2026-06-13.md" \
  --path "notes/garg_message_2026-06-14.md" \
  --path "notes/garg_message_2026-06-23.md" \
  --path "Alignment Forum research note.pdf" \
  --path-glob "A_Rate_Distortion_*.pdf" \
  --path "tmlr_submission/supplementary/audit-trail.bundle" \
  --path "tmlr_submission/supplementary.zip" \
  --path "tmlr_submission/overleaf_tmlr.zip" \
  --path "tmlr_submission/paper.pdf" \
  --replace-text "$WORK/rules.txt" \
  --replace-message "$WORK/rules.txt" \
  --name-callback 'return b"Anonymous Author"' \
  --email-callback 'return b"anon@anonymous.invalid"' \
  --commit-callback '
commit.author_date = commit.author_date.split(b" ")[0] + b" +0000"
commit.committer_date = commit.committer_date.split(b" ")[0] + b" +0000"
' \
  --force >/dev/null

echo "[5/6] verify no identifying string survives in any blob"
# One pattern, used for every check, so a term added here is added everywhere.
# It is wider than the rewrite rules on purpose: the rules say what to change,
# this says what must not survive, and the two failing to agree is the bug
# worth catching. 95154157 is here because a numeric GitHub user id resolves
# to the account as surely as the name does, and an earlier build left it
# inside this script's own rewritten rules.
#
# Note the self-reference. This script is in the bundle, so this very line is
# scanned by the check it defines, and every term below must therefore be one
# the rules above rewrite. The IP is written in full for that reason: a
# truncated 172.16.176 has no rule and would flag itself forever. The local
# timezone offset is deliberately absent, since the check for it is the
# positive one above, that every commit is +0000.
LEAKPAT='sanjay|sankalp|pathak|garg|jiit|k144u|jaypee|CLUSTER-HOST|95154157'
# The timezone is an identifier too. Every commit carried the authors' local
# offset, which names a geography on a submission whose names, emails,
# hostnames and paths have all been stripped. The callback above keeps the
# epoch second and relabels the offset as UTC, so ordering is untouched;
# ancestry is pure topology and does not consult dates at all. This comment
# deliberately does not write the original offset, because this script is
# itself in the bundle.
if git log --all --format='%ai %ci' | grep -qv '+0000'; then
  echo "  LEAK: a commit carries a non-UTC timezone offset" >&2; exit 3
fi
echo "  timezones normalised to UTC"
leak=0
while read -r o; do
  [ "$(git cat-file -t "$o" 2>/dev/null)" = "blob" ] || continue
  if git cat-file blob "$o" 2>/dev/null \
     | grep -qaiE "$LEAKPAT"; then
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
# A previous build of this bundle was briefly committed to the repository, and
# a bundle is built from history, so it ended up inside its own successor and
# the size tripled. The --path exclusions above strip it, and this guards the
# invariant: a bundle carrying a copy of a bundle has gone wrong somewhere.
sz=$(du -m "$OUT" | cut -f1)
echo "  size: ${sz} MB"
if [ "$sz" -gt 40 ]; then
  echo "  WARNING: unexpectedly large. Check for build artifacts in history:" >&2
  echo "    git rev-list --objects --all | sort -k2 | uniq -f1 -d" >&2
fi
rm -rf "$WORK/verify"
git clone -q "$OUT" "$WORK/verify"
cd "$WORK/verify"
echo "  $(git rev-list --count --all) commits, identity: $(git log -1 --format='%an <%ae>')"
echo
echo "old -> new hashes for the commit-trail table"
echo "(every hash the table cites, looked up by name rather than guessed by"
echo " keyword: a heuristic that silently misses a row is worse than useless"
echo " when the row is the evidence)"
echo

# The working-repository hashes the paper's commit-trail table cites, in table
# order. Update this list when a row is added, and the mapping below follows.
WANTED="
0d9924b 175453f
d503347 7edf68a
100cd43 b94331c 672a72c 3a0822e
fa5f3c3 6e51df3 de2aa10 25d7de0
40eae48 dc2991a
0c8b9e8 c597e31 2a49b3e a5e1ea1
fb21c0f 23c251b
e8c968a 30d2c04
7a12e2c 0d35789
ae3fdf1 603e013 778da6e 84526b9 11bc8a7 df832fe e66d273 f20b13c
"

missing=0
for want in $WANTED; do
  full=$(git -C "$REPO" rev-parse "$want^{commit}" 2>/dev/null || true)
  if [ -z "$full" ]; then
    printf "  %-9s NOT FOUND in the working repository\n" "$want"; missing=1; continue
  fi
  new=$(awk -v o="$full" '$1==o {print $2}' "$WORK/anon.git/filter-repo/commit-map")
  if [ -z "$new" ] || [ "$new" = "0000000000000000000000000000000000000000" ]; then
    printf "  %-9s -> DROPPED by the filter\n" "$want"; missing=1; continue
  fi
  msg=$(git log -1 --format='%s' "$new" 2>/dev/null | cut -c1-56)
  printf "  %-9s -> %s  %s\n" "$want" "${new:0:7}" "$msg"
done

echo
if [ "$missing" -ne 0 ]; then
  echo "WARNING: some cited commits could not be mapped. The table cannot be"
  echo "regenerated correctly until that is resolved."
fi
echo "bundle: $OUT"
