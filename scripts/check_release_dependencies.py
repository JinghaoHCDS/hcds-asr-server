"""Read-only release check for source pins and the locked MCP SDK contract."""
from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from asr_server.registry import ALIGNER_SOURCE, MODEL_SOURCES


def check_release_dependencies(root: Path) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    gpu_lines = (root / "requirements/wsl-gpu-cu128.txt").read_text(encoding="utf-8").splitlines()
    git_lines = [line.strip() for line in gpu_lines if "git+https://" in line and not line.lstrip().startswith("#")]
    if not git_lines or any(not re.search(r"@[0-9a-f]{40}$", line) for line in git_lines):
        raise ValueError("GPU Git dependencies must use full commit pins")
    sources = [*MODEL_SOURCES.values(), ALIGNER_SOURCE]
    if not sources or any(not repository or not re.fullmatch(r"[0-9a-f]{40}", revision) for repository, revision in sources):
        raise ValueError("model and aligner sources require repositories and full snapshot revisions")

    def mcp_requirement(values: list[str]) -> Requirement:
        requirements = [Requirement(value) for value in values]
        matches = [requirement for requirement in requirements if requirement.name == "mcp"]
        if len(matches) != 1:
            raise ValueError("exactly one MCP dependency is required in each declaration")
        return matches[0]
    optional = mcp_requirement(pyproject["project"]["optional-dependencies"]["mcp"])
    development = mcp_requirement(pyproject["dependency-groups"]["dev"])
    if optional.specifier != development.specifier:
        raise ValueError("optional and development MCP constraints differ")
    packages = {package["name"]: package for package in lock["package"]}
    mcp_version = Version(packages["mcp"]["version"])
    if mcp_version.major != 1 or mcp_version not in optional.specifier or Version("2") in optional.specifier:
        raise ValueError("locked MCP must satisfy the declared stable v1 constraint")
    metadata = packages["asr-server"]["metadata"]
    locked_optional = [item for item in metadata["requires-dist"] if item["name"] == "mcp"]
    locked_development = [item for item in metadata["requires-dev"]["dev"] if item["name"] == "mcp"]
    if (len(locked_optional) != 1 or len(locked_development) != 1
            or SpecifierSet(locked_optional[0]["specifier"]) != optional.specifier
            or SpecifierSet(locked_development[0]["specifier"]) != development.specifier):
        raise ValueError("lock metadata does not match the MCP dependency declarations")
    return {"status": "ok", "gpu_git_pins": len(git_lines), "model_snapshots": len(sources),
            "mcp_constraint": str(optional.specifier), "locked_mcp_version": str(mcp_version)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    try:
        result = check_release_dependencies(arguments.root)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"release dependency check failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
