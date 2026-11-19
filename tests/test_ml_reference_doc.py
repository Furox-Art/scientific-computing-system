"""The ML reference page and the frozen-constant script must name one sklearn.

``docs/ml_reference.md`` advertised its table as "scikit-learn 1.9.0" while
the prose said the figures were "last regenerated against scikit-learn 1.9.1".
The frozen constants in ``scripts/verify_ml_reference.py`` were computed with
1.9.0; the stray 1.9.1 made the provenance of every reference number ambiguous.
Nothing compared the two files, so the drift was silent. This pins them.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_MENTION = re.compile(r"(?:scikit-learn|sklearn) (\d+\.\d+\.\d+)")


def test_ml_reference_doc_and_script_agree_on_sklearn_version() -> None:
    doc = (ROOT / "docs" / "ml_reference.md").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "verify_ml_reference.py").read_text(encoding="utf-8")
    doc_versions = set(VERSION_MENTION.findall(doc))
    script_versions = set(VERSION_MENTION.findall(script))
    assert doc_versions, "ml_reference.md must state the reference sklearn version"
    assert script_versions, "verify_ml_reference.py must state the reference sklearn version"
    assert doc_versions == script_versions, (
        f"sklearn version drift: docs say {sorted(doc_versions)}, "
        f"the frozen-constant script says {sorted(script_versions)}"
    )


def test_ml_reference_table_header_matches_prose() -> None:
    doc = (ROOT / "docs" / "ml_reference.md").read_text(encoding="utf-8")
    headers = set(re.findall(r"Reference \(sklearn (\d+\.\d+\.\d+)\)", doc))
    prose = set(re.findall(r"scikit-learn (\d+\.\d+\.\d+)", doc))
    assert headers, "the reference table must carry a versioned header"
    assert headers <= prose, f"table header versions {headers} absent from prose {prose}"
