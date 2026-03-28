"""Generate Python contributor-search validation artifacts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.contrib_search_validation_cases import (  # noqa: E402
    GENERATED_AT,
    VALIDATION_CASES,
    build_validation_artifact_for_case,
)


async def main() -> None:
    generated_files: list[str] = []

    for validation_case in VALIDATION_CASES:
        artifact, _resolution = await build_validation_artifact_for_case(validation_case)
        output_path = REPO_ROOT / validation_case.artifact_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{json.dumps(artifact.model_dump(by_alias=True), indent=2)}\n",
            encoding="utf-8",
        )
        generated_files.append(str(output_path.relative_to(REPO_ROOT)))

    print(
        json.dumps(
            {
                "generatedAt": GENERATED_AT,
                "caseCount": len(generated_files),
                "files": generated_files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
