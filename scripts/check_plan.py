#!/usr/bin/env python3
'''Check backlog execution plan.'''
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

TICKETS_DIR = Path("/home/maxbogus/Repositories/contest/docs/backlog/tickets")
GATES = {
    "T1.1.1":["T1.0.2"],"T1.1.3":["T1.0.2"],"T1.2.2":["T1.1.1","T1.1.3"],
    "T1.4.7":["T1.2.2"],"T3.6.13":["T3.0.1"],"T3.6.11":["T3.6.13"],
    "T3.6.4":["T3.6.11"],"T3.6.7":["T3.6.4"],"T3.6.14":["T3.6.4"],
    "T2.6.7":["T2.0.1"],"T2.6.10":["T2.6.7"],"T2.6.11":["T2.6.10"],"T2.6.14":["T2.6.11"],
}
TIER = {"T3.6.13":["T1.1.1"],"T2.6.7":["T3.6.4"]}

#: Statuses that mean "no further action on this ticket".
#: Anything in this set is filtered out of --next/--rice and reported as
#: not-startable by can_start(). Added in M-009 follow-up:
#: duplicate, blocked, deferred, wont_fix are equally terminal as done.
TERMINAL_STATUSES = frozenset({"done", "duplicate", "blocked", "deferred", "wont_fix"})

#: Statuses that mean "active right now" (used by cmd_next/cmd_rice).
ACTIVE_STATUSES = frozenset({"backlog", "in_progress"})


def unfinished_in_prefix(prefix: str) -> list[dict]:
    """Return active (non-terminal, non-in_progress... wait, in_progress IS
    considered active here) tickets whose id starts with 'T{1,2,3}.'.

    Active = status in ACTIVE_STATUSES (backlog, in_progress).
    Terminal statuses (done, duplicate, blocked, deferred, wont_fix) are excluded.
    """
    result = []
    for t in get_all():
        tid = t.get("id", "")
        if not tid.startswith(f"{prefix}."):
            continue
        if t.get("status") in ACTIVE_STATUSES:
            result.append(t)
    return result


def _stage_visible(t: dict) -> bool:
    """Stage gating: T1 -> T2 -> T3 (T{n+1} hidden until Tn is done)."""
    tid = t.get("id", "")
    if tid.startswith("T1."):
        return True
    if tid.startswith("T2."):
        return len(unfinished_in_prefix("T1")) == 0
    if tid.startswith("T3."):
        return len(unfinished_in_prefix("T1")) == 0 and len(unfinished_in_prefix("T2")) == 0
    return True  # T0 and others: always visible

sys.path.insert(0, str(Path(__file__).parent))
try:
    import rice as _rice
except ImportError:
    _rice = None

def parse_fm(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"): return {}
    end = text.find("\n---", 4)
    if end < 0: return {}
    fm, nk = {}, None
    for line in text[4:end].split("\n"):
        if line.startswith("  ") and nk:
            s = line.strip()
            if ": " in s:
                k,_,v = s.partition(": ")
                if k and v: fm.setdefault(nk,{})[k] = v
            continue
        nk = None
        if ":" in line and not line.startswith(" "):
            k,_,v = line.partition(": "); v = v.strip()
            if v == "": nk = k; fm[k] = {}
            else: fm[k] = v
    return fm

def get_ticket(tid):
    matches = list(TICKETS_DIR.glob(f"{tid}*.md"))
    if not matches:
        # Fallback: check archive for done tickets
        archive_dir = TICKETS_DIR.parent / "archive"
        matches = list(archive_dir.glob(f"{tid}*.md"))
        if not matches:
            return None
    return {"path": matches[0], "id": tid, **parse_fm(matches[0])}

def get_all():
    results, seen = [], set()
    for p in sorted(TICKETS_DIR.glob("*.md")):
        m = re.match(r"^(T\d+(?:\.\d+)+|M-\d+)", p.stem)
        if not m: continue
        tid = m.group(1)
        if tid in seen: continue
        t = get_ticket(tid)
        if t: seen.add(tid); results.append(t)
    return results

def can_start(tid):
    t = get_ticket(tid)
    if not t: return False, ["not found"]
    s = t.get("status","unknown")
    if s in TERMINAL_STATUSES: return False, [f"already {s} (terminal)"]
    if s == "in_progress": return False, [f"already {s}"]
    bl = []
    if not tid.startswith("M-"):
        for d in GATES.get(tid,[]):
            dt = get_ticket(d)
            if not dt or dt.get("status") not in TERMINAL_STATUSES:
                bl.append(f"dep {d} not done")
    for b,rs in TIER.items():
        if b == tid:
            for r in rs:
                rt = get_ticket(r)
                if not rt or rt.get("status") not in TERMINAL_STATUSES:
                    bl.append(f"tier: {r}")
    return len(bl)==0, bl

def _confidence(t):
    r = t.get("rice", {})
    if isinstance(r, dict):
        c = r.get("C")
        if c is not None: return float(c)
    return 0.8

def _effort_h(t):
    e = t.get("effort", t.get("effort_h", "8.0"))
    try: return float(e)
    except (ValueError, TypeError): return 8.0

def _deps_norm(t):
    raw = t.get("depends_on", [])
    if isinstance(raw, str): return set(re.findall(r"T\d+\.\d+\.\d+", raw))
    if isinstance(raw, dict): return set(re.findall(r"T\d+\.\d+\.\d+", str(raw)))
    return set(raw)

def _blocks_norm(t):
    raw = t.get("blocks", [])
    if isinstance(raw, str): return set(re.findall(r"T\d+\.\d+\.\d+", raw))
    if isinstance(raw, dict): return set(re.findall(r"T\d+\.\d+\.\d+", str(raw)))
    return set(raw)

def _to_rice(t):
    return {
        "id": t["id"],
        "status": t.get("status", "unknown"),
        "priority": t.get("priority", "P1"),
        "repo": t.get("repo", "?"),
        "confidence": _confidence(t),
        "effort_h": _effort_h(t),
        "depends_on": _deps_norm(t),
        "blocks": _blocks_norm(t),
    }

def cmd_gate(tid):
    t = get_ticket(tid)
    if not t: print(f"X Ticket {tid} not found"); sys.exit(1)
    ok, reasons = can_start(tid)
    if ok:
        print(f"OK GATE PASSED: {tid}")
        print(f"   status={t.get('status','?')}, repo={t.get('repo','?')}")
        deps = GATES.get(tid,[])
        if deps: print(f"   deps: {', '.join(deps)} (all done)")
        sys.exit(0)
    else:
        print(f"X GATE BLOCKED: {tid}")
        print(f"   status={t.get('status','?')}")
        for r in reasons: print(f"   - {r}")
        print("\nTo unblock, complete:")
        for dep in GATES.get(tid,[]):
            dt = get_ticket(dep)
            if dt and dt.get("status") != "done":
                print(f"  -> {dep}: {dt.get('title','')}")
        sys.exit(2)

def cmd_status():
    all_t = get_all()
    by_status = defaultdict(list)
    for t in all_t: by_status[t.get("status","unknown")].append(t["id"])
    print("=" * 60)
    print(f"BACKLOG: {len(all_t)} tickets total")
    print("=" * 60)
    for s in ["done","in_progress","backlog"]:
        ts = by_status.get(s,[])
        short = ", ".join(ts[:5])
        print(f"  [{s.upper()}] ({len(ts)}): {short}{'...' if len(ts)>5 else ''}")
    all_rice = [_to_rice(t) for t in all_t]
    ready = [t for t in all_t
             if t.get("status") in ACTIVE_STATUSES and can_start(t["id"])[0]]
    ready_rice = [_to_rice(t) for t in ready]
    ranked = _rice.rank(ready_rice, all_rice) if _rice and ready_rice else []
    print()
    if ranked:
        print("TOP-3 READY (RICE):")
        for i, (t, s) in enumerate(ranked[:3], 1):
            marker = " <-- TAKE" if i == 1 else ""
            print(f"  {i}. [{t['id']}] rice={s:.2f} [{t.get('repo','?')}] {t.get('title','')[:40]}{marker}")
        top = ranked[0][0]
        print(f"\n  Gate: python scripts/check_plan.py --gate {top['id']}")
    else:
        print("  (all blocked by dependencies)")

def cmd_next(n=3, repo=None, json_out=False):
    all_t = get_all()
    all_rice = [_to_rice(t) for t in all_t]
    ready = [t for t in all_t
             if t.get("status") in ACTIVE_STATUSES and can_start(t["id"])[0]]
    # Stage gating: T2 hidden while T1 open; T3 hidden while T1 or T2 open
    ready = [t for t in ready if _stage_visible(t)]
    if repo: ready = [t for t in ready if t.get("repo") == repo]
    ready_rice = [_to_rice(t) for t in ready]

    # Warn if stage filter hid anything
    t1_open = unfinished_in_prefix("T1")
    t2_open = unfinished_in_prefix("T2")
    if t1_open:
        print("\n\u26a0\ufe0f T2/T3 hidden: T1 still has open tickets:",
              ", ".join(t["id"] for t in t1_open[:5]),
              f"(+{len(t1_open)-5} more)" if len(t1_open) > 5 else "")
    elif t2_open:
        print("\n\u26a0\ufe0f T3 hidden: T2 still has open tickets:",
              ", ".join(t["id"] for t in t2_open[:5]),
              f"(+{len(t2_open)-5} more)" if len(t2_open) > 5 else "")
    ranked = _rice.rank(ready_rice, all_rice) if _rice and ready_rice else []
    if json_out:
        data = {"ready": [{"id": t["id"], "rice": s, "title": t.get("title",""), "repo": t.get("repo","?"), "priority": t.get("priority","?"), "effort_h": _effort_h(t)} for t, s in ranked[:n]]}
        print(json.dumps(data, indent=2)); return
    if not ranked: print("No ready tickets."); return
    for i, (t, s) in enumerate(ranked[:n], 1):
        marker = " <-- TAKE" if i == 1 else ""
        print(f"{i}. [{t['id']}] rice={s:.2f} {t.get('title','')[:40]}{marker}")

def cmd_rice(repo=None, json_out=False):
    all_t = get_all()
    all_rice = [_to_rice(t) for t in all_t]
    backlog = [t for t in all_t if t.get("status") in ACTIVE_STATUSES]
    if repo: backlog = [t for t in backlog if t.get("repo") == repo]
    bl_rice = [_to_rice(t) for t in backlog]
    ranked = _rice.rank(bl_rice, all_rice) if _rice and bl_rice else []
    if json_out:
        data = {"backlog": [{"id": t["id"], "rice": s, "title": t.get("title",""), "repo": t.get("repo","?"), "priority": t.get("priority","?"), "effort_h": _effort_h(t)} for t, s in ranked]}
        print(json.dumps(data, indent=2)); return
    for t, s in ranked:
        print(f"{t['id']:10s} rice={s:6.2f} [{t.get('repo','?')}] {t.get('title','')[:40]}")

def cmd_archive(tid):
    t = get_ticket(tid)
    if not t: print(f"X Ticket {tid} not found"); sys.exit(1)
    if t.get("status") not in TERMINAL_STATUSES:
        print(f"X Only terminal-status tickets can be archived. "
              f"Current: {t.get('status')}"); sys.exit(1)
    src = t["path"]
    dst = Path("/home/maxbogus/Repositories/contest/docs/backlog/archive") / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    import subprocess, shutil
    r = subprocess.run(["git", "mv", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0: shutil.move(str(src), str(dst))
    print(f"Archived: {src.name} -> docs/backlog/archive/")

def cmd_suggestions():
    nexts = [t for t in get_all() if t.get("status") not in ("done","in_progress") and can_start(t["id"])[0]]
    print("=" * 60)
    print(f"AI AGENT PROMPTS (next {min(3,len(nexts))} tickets):")
    print("=" * 60)
    for t in nexts[:3]:
        tid = t["id"]
        print(f"\n--- {tid}: {t.get('title','?')} ---")
        print(f"Repo: {t.get('repo','contest')}, P={t.get('priority','?')}, Effort={t.get('effort','?')}h")
        print(f"Gate: python scripts/check_plan.py --gate {tid}")
        tid_ = tid.lower().replace(".","_")
        print(f"Suggested: uv run pytest tests/test_{tid_}.py --tb=short")
        print(f"  then: uv run ruff check . && git add -A && git commit")
        print(f"commit: feat({t.get('repo','task')}): {tid} {t.get('title','')[:40]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", metavar="ID")
    p.add_argument("--status", metavar="ID")
    p.add_argument("--next", action="store_true")
    p.add_argument("--rice", action="store_true")
    p.add_argument("--archive", metavar="ID")
    p.add_argument("--suggestions", action="store_true")
    p.add_argument("--repo", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("-n", type=int, default=3)
    a = p.parse_args()
    if a.gate: cmd_gate(a.gate)
    elif a.status:
        t = get_ticket(a.status)
        if not t: print(f"Not found: {a.status}"); sys.exit(1)
        ok,reasons = can_start(a.status)
        print(f"{a.status}: status={t.get('status','?')}, can_start={ok}")
        if not ok:
            for r in reasons: print(f"  blocked: {r}")
    elif a.next: cmd_next(n=a.n, repo=a.repo, json_out=a.json)
    elif a.rice: cmd_rice(repo=a.repo, json_out=a.json)
    elif a.archive: cmd_archive(a.archive)
    elif a.suggestions: cmd_suggestions()
    else: cmd_status()
