"""Launch a retained figure or HMM comparison script with explicitly configured local data roots."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def roots(path):
    """Read explicit absolute data roots from JSON; reject unknown keys and placeholders."""
    values = json.loads(Path(path).read_text())
    allowed = set(json.loads((ROOT / "paths.example.json").read_text()))
    if set(values) - allowed:
        raise ValueError("Unknown data-root keys")
    for key, value in values.items():
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or value.startswith(("/path/to/", "/configure/"))
        ):
            raise ValueError(f"Configure an absolute path for {key}")
    return values


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    script = (ROOT / a.script).resolve()
    if not script.is_relative_to(ROOT / "workflows") or not script.is_file():
        p.error("Choose a script inside workflows")
    values = roots(a.paths)
    args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    raise SystemExit(
        subprocess.call([sys.executable, str(script), *args], env={**os.environ, **values})
    )
