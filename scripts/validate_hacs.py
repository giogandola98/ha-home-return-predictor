"""Offline HACS checks for this repository (fast local complement to ``hacs/action``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HACS_KEYS = {
    "name",
    "content_in_root",
    "zip_release",
    "filename",
    "hide_default_branch",
    "country",
    "homeassistant",
    "persistent_directory",
    "render_readme",
    "hacs",
}
MANIFEST_REQUIRED = {"domain", "name", "version", "documentation", "issue_tracker", "codeowners"}


def main() -> int:
    errors: list[str] = []
    hacs = json.loads((ROOT / "hacs.json").read_text())
    if not isinstance(hacs.get("name"), str):
        errors.append("hacs.json: 'name' must be a string")
    if unknown := set(hacs) - HACS_KEYS:
        errors.append(f"hacs.json: unknown keys {sorted(unknown)}")

    integrations = list((ROOT / "custom_components").glob("*/manifest.json"))
    if len(integrations) != 1:
        errors.append("expected exactly one integration in custom_components/")
    for path in integrations:
        manifest = json.loads(path.read_text())
        if missing := MANIFEST_REQUIRED - set(manifest):
            errors.append(f"{path.name}: missing {sorted(missing)}")
        if manifest.get("domain") != path.parent.name:
            errors.append(f"{path.name}: domain does not match directory name")

    for error in errors:
        print(f"ERROR: {error}")
    print("HACS offline validation:", "failed" if errors else "ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
