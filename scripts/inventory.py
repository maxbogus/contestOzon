#!/usr/bin/env python3
"""Backlog inventory: LLM-optimized profiling of the contest backlog."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_plan import TICKETS_DIR, can_start
import rice as rice_module


# ─── Frontmatter parser ───────────────────────────────────────────────────────

def parse_fm(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    fm, nk = {}, None
    for line in text[4:end].split("\n"):
        if line.startswith("  ") and nk:
            s = line.strip()
            if ": " in s:
                k, _, v = s.partition(": ")
                if k and v:
                    fm.setdefault(nk, {})[k] = v
            continue
        nk = None
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(": ")
            v = v.strip()
            if v == "":
                nk = k
                fm[k] = {}
            else:
                fm[k] = v
    return fm


def load_all_with_rice() -> list[dict]:
    tickets = []
    for p in sorted(TICKETS_DIR.glob("T*.md")):
        m = re.match(r"^(T\d+(?:\.\d+)+)", p.stem)
        if not m:
            continue
        tid = m.group(1)
        fm = parse_fm(p.read_text(encoding="utf-8"))
        if not fm:
            continue

        deps_raw = fm.get("depends_on", [])
        if isinstance(deps_raw, str):
            deps_raw = re.findall(r"T\d+\.\d+\.\d+", deps_raw)
        elif isinstance(deps_raw, dict):
            deps_raw = re.findall(r"T\d+\.\d+\.\d+", str(deps_raw))

        blocks_raw = fm.get("blocks", [])
        if isinstance(blocks_raw, str):
            blocks_raw = re.findall(r"T\d+\.\d+\.\d+", blocks_raw)
        elif isinstance(blocks_raw, dict):
            blocks_raw = re.findall(r"T\d+\.\d+\.\d+", str(blocks_raw))

        rice_dict = fm.get("rice", {})
        if isinstance(rice_dict, dict):
            confidence = float(rice_dict.get("C", "0.8"))
        else:
            confidence = 0.8

        effort_raw = fm.get("effort", "8.0")
        try:
            effort_h = float(effort_raw)
        except (ValueError, TypeError):
            effort_h = 8.0

        tickets.append({
            "id": tid,
            "status": fm.get("status", "unknown"),
            "priority": fm.get("priority", "P1"),
            "repo": fm.get("repo", "contest"),
            "confidence": confidence,
            "effort_h": effort_h,
            "depends_on": set(deps_raw),
            "blocks": set(blocks_raw),
            "title": fm.get("title", ""),
        })
    return tickets


# ─── Critical path ─────────────────────────────────────────────────────────────

def compute_critical_path(tickets: list[dict]) -> list[str]:
    by_id = {t["id"]: t for t in tickets}
    adj: dict[str, list[str]] = defaultdict(list)
    for t in tickets:
        for dep in t.get("depends_on", set()):
            adj[dep].append(t["id"])

    roots = []
    for t in tickets:
        if t.get("priority") != "P0":
            continue
        deps = t.get("depends_on", set())
        if not deps or all(by_id.get(d, {}).get("priority") != "P0" for d in deps):
            roots.append(t["id"])

    if not roots:
        return []

    def dfs(node: str, path: list[str], visited: set) -> list[str]:
        best = path[:]
        for child in adj.get(node, []):
            if child in visited:
                continue
            if by_id.get(child, {}).get("priority") != "P0":
                continue
            visited.add(child)
            candidate = dfs(child, path + [child], visited)
            if len(candidate) > len(best):
                best = candidate
            visited.discard(child)
        return best

    all_paths = []
    for root in roots:
        path = dfs(root, [root], {root})
        all_paths.append(path)

    return max(all_paths, key=len) if all_paths else []


# ─── Markdown output ───────────────────────────────────────────────────────────

def emit_markdown(tickets: list[dict]) -> str:
    by_status = defaultdict(list)
    by_repo = defaultdict(list)
    by_priority = defaultdict(list)
    for t in tickets:
        by_status[t["status"]].append(t)
        by_repo[t.get("repo", "?")].append(t)
        by_priority[t["priority"]].append(t)

    total = len(tickets)
    done = len(by_status["done"])
    backlog = len(by_status["backlog"])
    in_progress = len(by_status["in_progress"])
    blocked = len(by_status.get("blocked", []))

    ready = [t for t in tickets if t["status"] == "backlog" and can_start(t["id"])[0]]
    ranked = rice_module.rank(ready, tickets)
    top_ready = ranked[:5]
    critical = compute_critical_path(tickets)
    effort_total = sum(t["effort_h"] for t in tickets if t["status"] == "backlog")

    effort_by_repo: dict[str, float] = {}
    for repo in by_repo:
        h = sum(t["effort_h"] for t in by_repo[repo] if t["status"] == "backlog")
        if h > 0:
            effort_by_repo[repo] = h

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Contest Backlog Inventory\n",
        f"> Generated: {ts}  |  Total: **{total}**\n",
        "## SUMMARY\n",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total | {total} |",
        f"| Done | {done} |",
        f"| Backlog | {backlog} |",
        f"| In Progress | {in_progress} |",
        f"| Blocked | {blocked} |\n",
        "```json",
        json.dumps({
            "total": total, "done": done, "backlog": backlog,
            "in_progress": in_progress, "blocked": blocked,
            "by_repo": {k: len(v) for k, v in by_repo.items()},
            "by_priority": {k: len(v) for k, v in by_priority.items()},
        }, indent=2),
        "```\n",
        "## BY REPO\n",
        "| Repo | Tickets | Backlog | Done |",
        "|---|---|---|---|",
    ]
    for repo in sorted(by_repo):
        ts2 = by_repo[repo]
        b = sum(1 for t in ts2 if t["status"] == "backlog")
        d = sum(1 for t in ts2 if t["status"] == "done")
        lines.append(f"| {repo} | {len(ts2)} | {b} | {d} |")

    lines += [
        "",
        "## BY PRIORITY\n",
        "| Priority | Count |",
        "|---|---:|",
    ]
    for p in ["P0", "P1", "P2", "P3"]:
        lines.append(f"| {p} | {len(by_priority[p])} |")

    lines += [
        "",
        "## TOP-5 READY (RICE)\n",
        "> Ready = status=backlog + all deps done + tier gate passed.\n",
        "| Rank | ID | Repo | RICE | Effort | Priority | Title |",
        "|---|---|---|---|---|---|---|",
    ]
    if top_ready:
        for i, (t, s) in enumerate(top_ready, 1):
            title = t.get("title", "")[:50]
            if len(t.get("title", "")) > 50:
                title += "..."
            lines.append(f"| {i} | {t['id']} | {t.get('repo', '?')} | {s:.2f} | {t['effort_h']:.1f}h | {t['priority']} | {title} |")
    else:
        lines.append("*No ready tickets (all blocked by dependencies).*")

    lines += [
        "",
        "## CRITICAL PATH (P0 chain)\n",
    ]
    if critical:
        lines.append(" → ".join(f"`{tid}`" for tid in critical))
        lines.append(f"\n*Length: {len(critical)} P0 tickets.*")
    else:
        lines.append("*No P0 critical path found.*")

    lines += [
        "",
        "## ESTIMATED EFFORT REMAINING\n",
        "| Repo | Backlog Hours |",
        "|---|---:|",
    ]
    for repo in sorted(effort_by_repo):
        lines.append(f"| {repo} | {effort_by_repo[repo]:.1f}h |")
    lines.append(f"| **Total** | **{effort_total:.1f}h** |\n")

    lines += ["## NEXT ACTION\n"]
    if top_ready:
        t, s = top_ready[0]
        lines.append(f"**Take: `{t['id']}`** — {t.get('title', '')}")
        lines.append(f"- RICE: {s:.2f} | Priority: {t['priority']} | Effort: {t['effort_h']:.1f}h")
        lines.append(f"- Gate: `python scripts/check_plan.py --gate {t['id']}`")
    else:
        lines.append("*No ready tickets. Complete dependencies first.*")

    return "\n".join(lines)


def emit_json(tickets: list[dict]) -> str:
    by_status = defaultdict(list)
    by_repo = defaultdict(list)
    for t in tickets:
        by_status[t["status"]].append(t["id"])
        by_repo[t.get("repo", "?")].append(t["id"])
    ready = [t for t in tickets if t["status"] == "backlog" and can_start(t["id"])[0]]
    ranked = rice_module.rank(ready, tickets)
    return json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "total": len(tickets),
        "by_status": dict(by_status),
        "by_repo": {k: len(v) for k, v in by_repo.items()},
        "critical_path": compute_critical_path(tickets),
        "ready": [
            {"id": t["id"], "rice": s, "priority": t["priority"],
             "effort_h": t["effort_h"], "repo": t.get("repo", "?")}
            for t, s in ranked[:10]
        ],
    }, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tickets = load_all_with_rice()
    if args.repo:
        tickets = [t for t in tickets if t.get("repo") == args.repo]
    if not tickets:
        print("No tickets match.", file=sys.stderr)
        return 1
    if args.json:
        print(emit_json(tickets))
    else:
        print(emit_markdown(tickets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
