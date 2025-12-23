"""
Refresh requirements.txt pins to whatever is currently latest-installable.

What it does:
  1) Reads requirements.txt (top-level deps only).
  2) Installs/upgrades those packages (pip install -U ...).
  3) Writes requirements.txt with pinned versions (name==installed_version).

Usage (PowerShell):
  python .\admin_scripts\refresh_requirements.py

Notes:
  - This intentionally pins ONLY the top-level packages listed in requirements.txt,
    not the full transitive dependency tree (i.e., it's not "pip freeze").
  - Run this inside the venv/conda env you deploy with.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REQ_FILE_DEFAULT = Path(__file__).resolve().parents[1] / "requirements.txt"


_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _iter_top_level_names(requirements_txt: str) -> list[str]:
    names: list[str] = []
    for raw in requirements_txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Keep it simple: grab leading distribution name; ignore specifiers/extras/markers.
        m = _NAME_RE.match(line)
        if not m:
            raise ValueError(f"Unrecognized requirement line: {raw!r}")
        names.append(m.group(1))
    return names


def _pip_install_upgrade(pkgs: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *pkgs]
    subprocess.check_call(cmd)


def _get_installed_version(dist_name: str) -> str:
    # metadata.version accepts normalized names; it's OK if case differs.
    return metadata.version(dist_name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Upgrade top-level deps and rewrite requirements.txt pins.")
    ap.add_argument("--requirements", default=str(REQ_FILE_DEFAULT), help="Path to requirements.txt")
    ap.add_argument("--no-install", action="store_true", help="Only rewrite pins based on already-installed versions")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written, but don't write the file")
    args = ap.parse_args()

    req_path = Path(args.requirements).resolve()
    if not req_path.exists():
        print(f"requirements file not found: {req_path}", file=sys.stderr)
        return 2

    original = req_path.read_text(encoding="utf-8")
    pkgs = _iter_top_level_names(original)
    if not pkgs:
        print("No requirements found (file is empty or only comments).", file=sys.stderr)
        return 2

    if not args.no_install:
        _pip_install_upgrade(pkgs)

    lines_out: list[str] = []
    for name in pkgs:
        ver = _get_installed_version(name)
        lines_out.append(f"{name}=={ver}")
    new_text = "\n".join(lines_out) + "\n"

    if args.dry_run:
        print(new_text, end="")
        return 0

    req_path.write_text(new_text, encoding="utf-8")
    print(f"Wrote {req_path} ({len(lines_out)} pinned packages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


