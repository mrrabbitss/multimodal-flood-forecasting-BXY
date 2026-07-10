from __future__ import annotations

import re
from pathlib import Path


def test_documentation_does_not_claim_normalized_threshold_in_centimeters() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = [root / "README.md", root / "PROJECT.md", root / "ARCHITECTURE_EXPERIMENTS.md"]
    pattern = re.compile(r"\b28\s*cm\b", flags=re.IGNORECASE)
    for path in documents:
        assert pattern.search(path.read_text(encoding="utf-8")) is None, path
