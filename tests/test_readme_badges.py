"""README badge contract.

The `img.shields.io/pypi/dm/<pkg>.svg` downloads badge was removed because its
upstream data source (pypistats.org) is rate limited and returns HTTP 500. When
that happens shields.io does not fail: it serves a 200 SVG whose text reads
"downloads: rate limited by upstream service", so the broken badge looks
present in the rendered README while showing no number at all.

No dynamic downloads badge may come back, and no hand-written download count
may replace it. Both are silent regressions: the README still renders, still
links to PyPI, and nobody notices until a reader spots the dead badge.

These checks read files only. They never touch the network, so they stay
deterministic in CI and cannot be satisfied by an endpoint that happens to be
healthy on the day the suite runs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Any shields.io path segment that renders a download count from an upstream
# API. `pypi/dm` is monthly, `pypi/dd` is daily, `pypi/dw` is weekly. All three
# depend on pypistats and fail the same way.
UNSTABLE_DOWNLOAD_BADGE = re.compile(r"img\.shields\.io/pypi/d[mdw]")

# A digit glued to a download unit, e.g. "482/month", "1,204 downloads".
HAND_WRITTEN_DOWNLOAD_COUNT = re.compile(
    r"\d[\d,._]*\s*(?:/month|/day|/week|downloads?\b)",
    re.IGNORECASE,
)

PYPI_PROJECT_URL = "https://pypi.org/project/scientific-computing-system/"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_has_no_unstable_dynamic_downloads_badge() -> None:
    """The rate-limited pypistats-backed badge must not be back in the README."""
    match = UNSTABLE_DOWNLOAD_BADGE.search(_readme())
    assert match is None, f"dynamic downloads badge reintroduced in README.md: {match.group(0)}"


def test_readme_advertises_no_shields_pypi_download_metric() -> None:
    """Belt-and-braces: no `pypi/<download metric>/` shield survives, whatever it is called."""
    shields_pypi_metrics = set(re.findall(r"https://img\.shields\.io/pypi/([a-z]+)", _readme()))
    assert not shields_pypi_metrics & {"dm", "dd", "dw"}, (
        f"unstable shields.io pypi download badge(s) present: "
        f"{sorted(shields_pypi_metrics & {'dm', 'dd', 'dw'})}"
    )


def test_readme_points_readers_at_a_real_pypi_stats_link() -> None:
    """Download information must still be reachable, as a link rather than a broken image."""
    readme = _readme()
    match = re.search(r"\[PyPI stats\]\((\S+?)\)", readme)
    assert match is not None, "README.md no longer links to PyPI stats"
    assert match.group(1) == PYPI_PROJECT_URL, f"unexpected PyPI stats href: {match.group(1)}"


def test_readme_states_no_hand_copied_download_number() -> None:
    """A number copied by hand goes stale silently; the link is the honest form."""
    match = HAND_WRITTEN_DOWNLOAD_COUNT.search(_readme())
    assert match is None, f"README.md hard-codes a download count: {match.group(0)!r}"


def test_docs_pages_carry_no_unstable_download_badge() -> None:
    """The docs site must not reintroduce the badge on any page."""
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "docs").rglob("*.md")
        if UNSTABLE_DOWNLOAD_BADGE.search(path.read_text(encoding="utf-8"))
    )
    assert not offenders, f"unstable downloads badge present in docs: {offenders}"


def test_readme_download_badge_free_of_shields_error_fallback_text() -> None:
    """Guards against pasting shields' own failure string into the README as a caption."""
    readme = _readme().lower()
    for phrase in ("rate limited by upstream service", "downloads inaccessible"):
        assert phrase not in readme, (
            f"README.md carries a captured badge failure string: {phrase!r}"
        )
