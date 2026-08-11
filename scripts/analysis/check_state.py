#!/usr/bin/env python3
"""Repository hygiene gate for experiment work.

Run before and after every experiment:
    python3 scripts/analysis/check_state.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STATUS = {"todo", "running", "blocked", "done", "dead"}
REQUIRED_RUN_HEADINGS = [
    "## Metadata",
    "## Question",
    "## Protocol",
    "## Commands",
    "## Live Log",
    "## Result",
    "## Decision",
    "## Sync Checklist",
]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def parse_queue(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|") or "---" in line or " ID " in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "id": cells[0].strip("`"),
                "status": cells[1].strip("`"),
                "config": cells[3],
                "record": cells[4].strip("`"),
                "raw": raw,
            }
        )
    return rows


def referenced_paths(text: str) -> list[str]:
    found = re.findall(r"`((?:configs|scripts|experiments)/[^`\s]+)`", text)
    found += re.findall(r"\b((?:configs|scripts|experiments)/[^\s)）]+)", text)
    return [p.rstrip("`.,;:") for p in found]


def check_required_files(errors: list[str]) -> None:
    for rel in [
        "AGENTS.md",
        "docs/STATE.md",
        "docs/LEDGER.md",
        "experiments/README.md",
        "experiments/QUEUE.md",
        "experiments/RUN_TEMPLATE.md",
        "scripts/analysis/check_state.py",
    ]:
        if not (ROOT / rel).exists():
            fail(errors, f"missing required file: {rel}")


def check_no_root_clutter(errors: list[str]) -> None:
    root_scripts = [p.name for p in (ROOT / "scripts").iterdir() if p.is_file() and p.name != "README.md"]
    if root_scripts:
        fail(errors, "scripts/ root contains executable clutter: " + ", ".join(sorted(root_scripts)))
    root_configs = [p.name for p in (ROOT / "configs").glob("*.yaml")]
    if root_configs:
        fail(errors, "configs/ root contains yaml clutter: " + ", ".join(sorted(root_configs)))


def check_config_bases(errors: list[str]) -> None:
    for p in (ROOT / "configs").rglob("*.yaml"):
        if not p.is_file():
            continue
        for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line.startswith("base:"):
                continue
            target = (p.parent / line.split(":", 1)[1].strip()).resolve()
            if not target.exists():
                fail(errors, f"broken config base: {p.relative_to(ROOT)}:{line_no} -> {target}")


def check_queue(errors: list[str]) -> None:
    queue = ROOT / "experiments/QUEUE.md"
    rows = parse_queue(queue)
    if not rows:
        fail(errors, "experiments/QUEUE.md has no experiment rows")
        return
    running = [r for r in rows if r["status"] == "running"]
    if len(running) > 1:
        fail(errors, "only one running experiment is allowed: " + ", ".join(r["id"] for r in running))
    state_text = (ROOT / "docs/STATE.md").read_text(encoding="utf-8")
    for row in rows:
        if row["status"] not in ALLOWED_STATUS:
            fail(errors, f"invalid queue status for {row['id']}: {row['status']}")
        if row["status"] in {"running", "done", "dead"}:
            record = ROOT / row["record"]
            if not record.exists():
                fail(errors, f"{row['status']} experiment missing run record: {row['record']}")
        if row["status"] == "running" and row["id"] not in state_text:
            fail(errors, f"running experiment {row['id']} is not mentioned in docs/STATE.md")


def check_run_records(errors: list[str]) -> None:
    run_dir = ROOT / "experiments/runs"
    for p in sorted(run_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        for heading in REQUIRED_RUN_HEADINGS:
            if heading not in text:
                fail(errors, f"run record {p.relative_to(ROOT)} missing heading: {heading}")
        for rel in referenced_paths(text):
            if rel.startswith("configs/") and not (ROOT / rel).exists():
                fail(errors, f"run record {p.relative_to(ROOT)} references missing path: {rel}")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_no_root_clutter(errors)
    check_config_bases(errors)
    check_queue(errors)
    check_run_records(errors)
    if errors:
        print("STATE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("STATE CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
