#!/usr/bin/env bash
# Daily audit of entries the scan wrote (status pending_audit). Identical in both catalog repos.
# Meant for a script-only cron job after both daily scans:  bash tools/daily_audit.sh
# Steps: git precheck (same rules as the daily scan) -> catalog_tool.py cross-sync (reverse links for the other
# catalog's cross relations) -> pattern-sync (derive catalog/patterns + entry `pattern` from the var forest)
# -> tools/catalog_audit.py run -> build views -> validate -> stage only catalog/ + views/
# -> commit "daily audit YYYY-MM-DD: <repo>" -> push -> verify remote.
# Exits 0 without committing when nothing is pending and neither sync changed anything. Needs JUDGE_API_KEY.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(basename "$PWD" | sed 's/_error_review$//')
DAY=$(date -u +%F)
say() { echo "$(date -u +%FT%TZ) $*"; }

# 1. git precheck: clean main, fetch ok, HEAD == origin/main (never pull/reset/force)
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { say "BLOCKED: not on main"; exit 1; }
[ -z "$(git status --porcelain -uall)" ] || { say "BLOCKED: working tree not clean"; git status --short | head; exit 1; }
git fetch -q origin main || { say "BLOCKED: fetch failed"; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { say "BLOCKED: HEAD differs from origin/main"; exit 1; }

# 2. reverse links for cross relations the other catalog made (deterministic, no gateway calls)
python3 tools/catalog_tool.py cross-sync > /tmp/daily_audit_cross_$REPO.json || { say "FAILED: cross-sync"; exit 1; }
CROSS=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['added']))" /tmp/daily_audit_cross_$REPO.json)
[ "$CROSS" = 0 ] || say "cross-sync: added $CROSS reverse cross link(s)"

# 2b. pattern layer: membership is derived from var relations, so a relation the scan wrote last night
# puts its entry in the right family here (deterministic, no gateway calls, idempotent).
python3 tools/catalog_tool.py pattern-sync > /tmp/daily_audit_pattern_$REPO.json || { say "FAILED: pattern-sync"; exit 1; }
PATS=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d['created']) + len(d['membership_changed']) + d['entries_touched'])" /tmp/daily_audit_pattern_$REPO.json)
[ "$PATS" = 0 ] || say "pattern-sync: $PATS pattern/entry change(s)"

PENDING=$(python3 tools/catalog_tool.py stats | python3 -c "import json,sys; print(json.load(sys.stdin)['status'].get('pending_audit', 0))")
if [ "$PENDING" = 0 ] && [ "$CROSS" = 0 ] && [ "$PATS" = 0 ]; then say "nothing pending; no commit"; exit 0; fi

# 3-5. audit, rebuild views, validate
if [ "$PENDING" != 0 ]; then
  say "auditing $PENDING pending entries"
  python3 tools/catalog_audit.py run || { say "FAILED: audit run"; exit 1; }
fi
python3 tools/catalog_tool.py build > /dev/null || { say "FAILED: build"; exit 1; }
python3 tools/catalog_tool.py validate > /tmp/daily_audit_validate_$REPO.json || {
  say "FAILED: validate"; head -c 2000 /tmp/daily_audit_validate_$REPO.json; exit 1; }

# 6. commit only the catalog deliverables, push, verify
git add catalog/entries catalog/audit catalog/patterns views
git diff --cached --quiet && { say "no changes after audit; no commit"; exit 0; }
git -c core.whitespace=-blank-at-eol diff --cached --check -- . ':!*.csv' ':!*.jsonl' || { say "FAILED: whitespace check"; exit 1; }
git commit -q -m "daily audit $DAY: $REPO" || { say "FAILED: commit"; exit 1; }
git push -q origin HEAD:main || { say "FAILED: push (local commit $(git rev-parse --short HEAD) not synced)"; exit 1; }
REMOTE=$(git ls-remote origin refs/heads/main | cut -f1)
[ "$REMOTE" = "$(git rev-parse HEAD)" ] && say "OK: pushed $(git rev-parse --short HEAD)" || { say "FAILED: remote $REMOTE != local"; exit 1; }
