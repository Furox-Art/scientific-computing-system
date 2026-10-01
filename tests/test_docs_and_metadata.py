"""Documentation and package-metadata contract tests.

These are deterministic, network-free checks that pin the *discoverability*
claims the README and `pyproject.toml` make. Without them, a docs refactor can
silently rot: a README code block that no longer imports, a badge pointing at a
workflow that does not exist, a `project.urls` entry with a typo, or a docs page
that exists on disk but is unreachable from the mkdocs navigation.

What is asserted here is deliberately narrow -- structural agreement between
files that already exist in the repository. Nothing reaches the network, so the
suite is reproducible in CI and offline.

Covered:

* ``README.md`` import statements resolve against the installed package, so the
  quickstart cannot advertise a ``scs`` module that does not exist.
* Every *imported name* -- not just every module -- resolves, across the README
  and the top-level docs pages. Resolving the module alone would still have
  passed the original README's invented ``bayesian_posterior``.
* The README transcript block is byte-identical to what
  ``examples/quickstart_demo.py`` actually prints.
* README badges point at workflows and files that are present in this repo.
* Every ``project.urls`` target is a real repository-relative path or a
  well-formed absolute URL.
* Metadata self-consistency: `requires-python`, the Python classifiers, the
  license, the zero-dependency claim and the PyPI install command.
* Classifiers are published PyPI trove codes, not merely well-formed strings.
* The "N domain modules" headline equals the real count under ``src/cds``, and
  the docs module table lists every shipped module.
* Every ``python`` block on the entry-point docs pages executes as a standalone
  script.
* Every docs page on disk is reachable from the mkdocs ``nav``.
"""

from __future__ import annotations

import importlib
import io
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlparse

import pytest

import cds

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
MKDOCS = ROOT / "mkdocs.yml"
DOCS = ROOT / "docs"
QUICKSTART = ROOT / "examples" / "quickstart_demo.py"

DISTRIBUTION = "scientific-computing-system"
IMPORT_NAME = "cds"


# --------------------------------------------------------------------------
# Minimal pyproject reader
#
# ``tomllib`` is deliberately not used, for the same reason documented in
# tests/test_dev_numpy_pin_contract.py: it only exists on Python 3.11+ and CI
# runs this file on 3.10 too. Rather than take a backport dependency, the few
# scalar/array fields asserted below are read textually.
# --------------------------------------------------------------------------


def _table_body(text: str, header: str) -> str:
    """Return the lines of a single TOML table, stopping at the next header."""
    body: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if inside:
                break
            inside = stripped == header
            continue
        if inside:
            body.append(line)
    if not inside:
        raise AssertionError(f"pyproject.toml has no {header} table")
    return "\n".join(body)


def _scalar(field: str, *, header: str = "[project]") -> str:
    """Return a quoted scalar field from a table."""
    match = re.search(
        rf'^{re.escape(field)}\s*=\s*["\']([^"\']+)["\']',
        _table_body(_pyproject(), header),
        re.MULTILINE,
    )
    assert match is not None, f"pyproject.toml [{header}] is missing {field}"
    return match.group(1)


def _string_array(field: str, *, header: str = "[project]") -> list[str]:
    """Return a possibly multi-line TOML string array field.

    Handles both the single-line inline form (``keywords = ["a", "b"]``) and the
    multi-line form with one entry per line (``keywords = [\\n  "a",\\n  "b",\\n]``).
    """
    body = _table_body(_pyproject(), header)
    match = re.search(rf"^{re.escape(field)}\s*=\s*\[(.*?)\]", body, re.MULTILINE | re.DOTALL)
    assert match is not None, f"pyproject.toml [{header}] is missing {field}"
    return re.findall(r'"([^"]+)"', match.group(1))


def _pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _project_urls() -> dict[str, str]:
    """Parse the ``[project.urls]`` table into label -> URL."""
    urls: dict[str, str] = {}
    pattern = re.compile(r'^(?:"([^"]+)"|([\w \-]+?))\s*=\s*"(https://[^"]+)"')
    for line in _table_body(_pyproject(), "[project.urls]").splitlines():
        match = pattern.match(line.strip())
        if match:
            urls[match.group(1) or match.group(2)] = match.group(3)
    return urls


def _raw_dependencies() -> str:
    """Return the raw text of the ``dependencies`` array in ``[project]``."""
    match = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]",
        _table_body(_pyproject(), "[project]"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "pyproject.toml [project] is missing dependencies"
    return match.group(1)


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _fenced_blocks(text: str, language: str) -> list[str]:
    """Return every fenced code block body for ``language``."""
    blocks: list[str] = []
    pattern = re.compile(rf"^```{language}\n(.*?)^```$", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        blocks.append(match.group(1))
    return blocks


def _python_import_statements(text: str) -> list[str]:
    """Every ``import x`` / ``from x import y`` line across all py blocks."""
    pattern = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+))\s*$")
    statements: list[str] = []
    for block in _fenced_blocks(text, "python"):
        for line in block.splitlines():
            match = pattern.match(line)
            if match:
                statements.append(line.strip())
    return statements


def _imported_symbols(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Every ``from <module> import <names>`` as (module, names) pairs.

    Resolving the module alone is not enough. ``from cds.stats import
    bayesian_posterior`` imports a real module and then fails on the missing
    attribute, so the *names* have to be checked too. This is the exact defect
    the original README shipped: ``from scs.stats import bayesian_posterior``
    looked plausible and named a function that does not exist.

    Parenthesised multi-line imports are joined first, and ``as`` aliases are
    reduced to their original name.
    """
    joined = re.sub(r"\(\s*([^)]*?)\s*\)", r"\1", text)
    pattern = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$", re.MULTILINE)
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for line in joined.splitlines():
        # Strip a trailing line comment before parsing; a name is an identifier,
        # so anything after `#` is prose, not part of the import list.
        code = line.split("#", 1)[0]
        match = pattern.match(code)
        if not match:
            continue
        module, raw = match.group(1), match.group(2)
        if raw.strip() == "*":
            continue
        names: list[str] = []
        for part in raw.split(","):
            token = part.strip()
            if not token or not token.isidentifier():
                continue
            # `name as alias` -> `name`
            names.append(token.split(" as ")[0].strip())
        if names:
            pairs.append((module, tuple(names)))
    return pairs


def _resolve_imported_names(text: str, *, source: str, require_imports: bool = False) -> None:
    """Every documented ``from cds... import y`` must have a real ``cds.y``.

    Only first-party imports are checked. Third-party references such as
    ``from sklearn.cluster import KMeans`` in the ML reference page name the
    optional ``[scientific]`` backends, which are legitimately absent from a
    core-only environment -- asserting on them would make this test depend on
    which extras happen to be installed.

    Raises ``AssertionError`` listing all unresolvable names at once, so a
    single run reports every broken snippet rather than one per attempt.

    ``require_imports`` guards against the check silently going vacuous on a
    page that *should* carry importable code (the README). Prose-only pages
    pass ``False`` and are simply skipped when they have nothing to verify.
    """
    broken: list[str] = []
    checked = 0
    for module_name, names in _imported_symbols(text):
        if module_name != IMPORT_NAME and not module_name.startswith(f"{IMPORT_NAME}."):
            continue
        checked += 1
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            broken.append(f"{module_name} (cannot import: {exc})")
            continue
        for name in names:
            if not hasattr(module, name):
                broken.append(f"{module_name}.{name}")
    if require_imports:
        assert checked, f"{source} has no `from cds... import ...` statements to verify"
    assert not broken, f"{source} imports names that do not exist: {sorted(broken)}"


# --------------------------------------------------------------------------
# README: the quickstart must be runnable
# --------------------------------------------------------------------------


def test_readme_python_blocks_exist() -> None:
    """Guard the guard: a regex that silently matches nothing is not a test."""
    blocks = _fenced_blocks(_readme_text(), "python")
    assert len(blocks) >= 1, "README has no ```python block; the quickstart is missing"


def test_readme_imported_names_all_resolve() -> None:
    """Every ``from cds.X import name`` in the README must name a real symbol.

    ``test_readme_import_statements_resolve`` only checks that the *module*
    imports. That is too weak: the original README's
    ``from scs.stats import bayesian_posterior`` failed at the module level,
    but a plausible-looking ``from cds.stats import bayesian_posterior``
    would import fine and then raise ``AttributeError`` at the next line.
    """
    readme = _readme_text()
    _resolve_imported_names(readme, source="README.md", require_imports=True)


@pytest.mark.parametrize("page", sorted(p.name for p in DOCS.glob("*.md")))
def test_docs_getting_started_and_index_imported_names_resolve(page: str) -> None:
    """Every documented import name on the top-level docs pages must resolve.

    Covers ``docs/getting-started.md`` and ``docs/index.md``, the two pages a
    new reader is most likely to copy code from.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    _resolve_imported_names(text, source=f"docs/{page}")


@pytest.mark.parametrize("statement", _python_import_statements(_readme_text()))
def test_readme_import_statements_resolve(statement: str) -> None:
    """Each README import must target the real package, not an invented one.

    This is the regression pin for the original README, which advertised
    ``from scs.linear_algebra import svd`` and ``from scs.ode import
    solve_ivp``. Neither module exists: the distribution is
    ``scientific-computing-system`` but the import name is ``cds``, and the
    modules are ``cds.math_utils`` and ``cds.diffeq``. Every reader who copied
    that snippet got an immediate ``ModuleNotFoundError``.
    """
    assert not statement.startswith("import scs"), (
        f"README imports 'scs', which does not exist; the import name is {IMPORT_NAME!r}: "
        f"{statement}"
    )
    assert " from scs." not in f" {statement}", (
        f"README imports from 'scs', which does not exist; the import name is "
        f"{IMPORT_NAME!r}: {statement}"
    )

    # Resolve the module part of the statement for real.
    match = re.match(r"^from\s+([\w.]+)\s+import", statement)
    if match:
        module_name = match.group(1)
    else:
        module_name = statement.split()[1]
    __import__(module_name)


def test_readme_declares_the_correct_import_name() -> None:
    """The README must tell readers the distribution name != import name."""
    text = _readme_text()
    assert f"import {IMPORT_NAME}" in text, (
        f"README never shows the real import name {IMPORT_NAME!r}"
    )
    assert "pip install scientific-computing-system" in text


def test_readme_troubleshooting_covers_the_wrong_import_name() -> None:
    """The known first-run failure must appear in the troubleshooting table."""
    assert "No module named 'scs'" in _readme_text()


def test_readme_has_the_required_discoverability_sections() -> None:
    """First-screen structure: scope, install, quickstart, links, badges."""
    text = _readme_text()
    required = (
        "## Why this library",
        "## Install",
        "## Quickstart",
        "## Scientific domains covered",
        "## Documentation",
        "## Adoption path",
        "## Troubleshooting",
        "## Project links",
        "## License",
    )
    missing = [heading for heading in required if heading not in text]
    assert not missing, f"README is missing required sections: {missing}"


def test_readme_makes_no_unverifiable_social_proof_claims() -> None:
    """Reject fabricated stars, downloads, dependents or contributor counts.

    ``pip install`` must never carry a download badge whose number the project
    cannot actually vouch for, and no "trusted by N teams" line may appear
    without a verifiable source.
    """
    text = _readme_text()
    forbidden = (
        "trusted by",
        "used by companies",
        "stars on GitHub",
        "weekly downloads",
        "monthly downloads",
        "dependents",
        "battle-tested in production at",
    )
    present = [phrase for phrase in forbidden if phrase.lower() in text.lower()]
    assert not present, f"README contains unverifiable social-proof claims: {present}"


# --------------------------------------------------------------------------
# README: the transcript must be real output, not a plausible-looking fiction
# --------------------------------------------------------------------------


def _quickstart_stdout() -> str:
    """Run the quickstart script and return its stdout, stripped."""
    completed = subprocess.run(
        [sys.executable, str(QUICKSTART)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert completed.returncode == 0, (
        f"examples/quickstart_demo.py exited {completed.returncode}:\n{completed.stderr}"
    )
    return completed.stdout.strip()


def test_quickstart_script_exists_and_runs() -> None:
    """The advertised transcript must come from a script that actually runs."""
    assert QUICKSTART.exists(), f"{QUICKSTART.name} is missing"
    assert _quickstart_stdout(), "quickstart produced no output"


def test_readme_transcript_matches_real_script_output() -> None:
    """The README ``text`` transcript must equal the script's actual stdout.

    Byte equality is the point: a hand-edited or invented transcript would fail
    here, which is what makes "real runnable output" an enforced claim rather
    than an aspiration.
    """
    transcripts = [block.strip() for block in _fenced_blocks(_readme_text(), "text")]
    expected = _quickstart_stdout()
    assert expected in transcripts, (
        "README does not contain the current stdout of "
        "examples/quickstart_demo.py. Re-run the script and paste its exact "
        "output into the README transcript block."
    )


def test_readme_transcript_is_reproducible_across_runs() -> None:
    """The quickstart must be deterministic: seeded or closed-form only."""
    assert _quickstart_stdout() == _quickstart_stdout()


def test_readme_transcript_reports_the_current_version() -> None:
    """The transcript header must not advertise a stale version."""
    expected = _quickstart_stdout()
    assert f"CDS {cds.__version__} --" in expected
    assert f"CDS {cds.__version__} --" in _readme_text()


def test_readme_inline_output_comments_match_real_execution() -> None:
    """Every ``# expected`` comment on a README ``print(...)`` must be real.

    The quickstart block annotates each result with the value it actually
    produces. Executing the block and comparing each annotated print against
    the corresponding stdout line pins those comments to reality, so a rounded,
    stale or invented number cannot survive in the README.
    """
    readme = _readme_text()
    blocks = _fenced_blocks(readme, "python")
    assert blocks, "README has no python block to verify"

    checked = 0
    for index, block in enumerate(blocks):
        namespace: dict[str, object] = {}
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(compile(block, f"<README block {index}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - failure path reports below
            pytest.fail(f"README python block {index} does not run: {type(exc).__name__}: {exc}")

        printed = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]

        # Annotated prints, in source order, map 1:1 onto stdout lines.
        claims: list[tuple[int, str]] = []
        for lineno, line in enumerate(block.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("print("):
                comment = stripped.partition("#")[2].strip()
                if comment:
                    claims.append((lineno, comment))

        assert len(claims) == len(printed), (
            f"README python block {index}: {len(claims)} annotated prints produced "
            f"{len(printed)} output lines; the block and its annotations disagree"
        )
        for (lineno, claim), got in zip(claims, printed):
            normalised_claim = claim.replace(" ", "")
            assert normalised_claim in got.replace(" ", ""), (
                f"README python block {index}, line {lineno}: annotated "
                f"{claim!r} but the code printed {got!r}"
            )
            checked += 1

    assert checked >= 5, (
        f"only {checked} inline output claims verified; expected the quickstart set"
    )


# --------------------------------------------------------------------------
# README: badges must point at things that exist
# --------------------------------------------------------------------------


def _badge_urls(text: str) -> list[str]:
    """Every ``<img src=...>`` URL in the README badge block."""
    return re.findall(r'<img\s+src="([^"]+)"', text)


def test_readme_badges_exist() -> None:
    """Guard the guard, and require the accuracy-relevant badges."""
    urls = _badge_urls(_readme_text())
    assert urls, "README has no badge images"
    joined = " ".join(urls)
    for service in ("pypi", "npm", "codecov", "LICENSE"):
        assert service.lower() in joined.lower(), f"README badge set is missing {service}"


def test_readme_ci_badge_targets_a_real_workflow() -> None:
    """The CI badge must reference a workflow file present in this repo."""
    text = _readme_text()
    referenced = re.findall(
        r"https://github\.com/Furox-Art/scientific-computing-system/"
        r"actions/workflows/([\w.\-/]+\.ya?ml)",
        text,
    )
    assert referenced, "README has no CI workflow badge"
    for workflow in referenced:
        assert (ROOT / ".github" / "workflows" / workflow).is_file(), (
            f"README badge references a workflow that does not exist: {workflow}"
        )


def test_readme_documentation_link_is_the_deployed_site() -> None:
    """Docs badge must point at the mkdocs site_url, not a placeholder."""
    site_url = re.search(r"^site_url:\s*(\S+)", MKDOCS.read_text(encoding="utf-8"), re.MULTILINE)
    assert site_url is not None, "mkdocs.yml has no site_url"
    assert site_url.group(1) in _readme_text(), (
        f"README does not link the deployed docs site {site_url.group(1)}"
    )


def test_readme_makes_no_scorecard_or_provenance_claims() -> None:
    """No OpenSSF Scorecard or SLSA badge without a real published scorecard.

    The repository has no ``.github/scorecard.yml`` result published, so a
    Scorecard badge would render as a false claim of a passing audit.
    """
    text = _readme_text().lower()
    for phrase in ("scorecard", "slsa", "sigstore", "provenance badge"):
        assert phrase not in text, (
            f"README advertises {phrase!r} without a verifiable public attestation"
        )


def test_readme_license_badge_matches_the_license_file() -> None:
    """License badge text must match the declared and actual license."""
    assert _scalar("license") == "MIT"
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text


# --------------------------------------------------------------------------
# pyproject metadata
# --------------------------------------------------------------------------


def test_project_declares_zero_runtime_dependencies() -> None:
    """The README's headline differentiator must be true in the metadata."""
    declared = re.findall(r'["\']([^"\']+)["\']', _raw_dependencies())
    assert declared == [], (
        "README advertises zero runtime dependencies; pyproject must declare none, "
        f"but found: {declared}"
    )


def _advertised_python_versions() -> list[tuple[int, int]]:
    """Version-specific ``Programming Language :: Python :: X.Y`` classifiers.

    The bare ``:: 3`` and ``:: 3 :: Only`` classifiers are intentionally
    excluded: they carry no minor version to check against.
    """
    pattern = re.compile(r"Programming Language :: Python :: (\d+\.\d+)\s*$", re.MULTILINE)
    found: list[tuple[int, int]] = []
    for classifier in _string_array("classifiers"):
        match = pattern.match(classifier)
        if match:
            major, minor = match.group(1).split(".")
            found.append((int(major), int(minor)))
    return found


def test_requires_python_matches_the_classifiers() -> None:
    """``requires-python`` and the advertised Python versions must agree."""
    requires = _scalar("requires-python")
    match = re.match(r"^>=(\d+)\.(\d+)$", requires)
    assert match is not None, f"unsupported requires-python format: {requires!r}"
    floor = (int(match.group(1)), int(match.group(2)))

    declared = sorted((int(major), int(minor)) for major, minor in _advertised_python_versions())
    assert declared, "no version-specific Python classifiers declared"
    assert declared[0] == floor, (
        f"lowest Python classifier {declared[0]} disagrees with requires-python {requires}"
    )
    for version in declared:
        assert version >= floor, f"classifier Python {version} is below requires-python {requires}"


def test_claimed_python_versions_are_actually_tested_in_ci() -> None:
    """Every advertised Python version must appear in the CI matrix.

    Advertising 3.13 support without a 3.13 CI cell would be a claim about
    something never verified.
    """
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python-version:\s*\[(.*?)\]", workflow)
    assert matrix is not None, "could not find the CI python-version matrix"
    tested = set(re.findall(r'"?(\d+\.\d+)"?', matrix.group(1)))

    advertised = {f"{major}.{minor}" for major, minor in _advertised_python_versions()}
    unverified = sorted(advertised - tested)
    assert not unverified, f"classifiers advertise Python {unverified} but CI does not test them"


def test_project_urls_are_well_formed_and_resolve_in_repo() -> None:
    """Each ``project.urls`` entry must be absolute, https, and non-decorative."""
    urls = _project_urls()
    assert len(urls) >= 5, "project.urls is too sparse to aid discovery"
    for label, value in urls.items():
        parsed = urlparse(value)
        assert parsed.scheme == "https", f"{label}: must be https, got {value!r}"
        assert parsed.netloc, f"{label}: missing host in {value!r}"

        if parsed.netloc == "github.com":
            parts = [p for p in parsed.path.split("/") if p]
            assert parts[0] == "Furox-Art", f"{label}: unexpected GitHub org {value!r}"
            # A blob/tree path must reference a tracked file in the repo.
            if len(parts) >= 4 and parts[2] in {"blob", "tree"}:
                repo_path = "/".join(parts[4:])
                assert repo_path, f"{label}: no path component in {value!r}"
                assert (ROOT / repo_path).exists(), (
                    f"{label}: points at {repo_path}, which is not in the repository"
                )


def test_project_urls_cover_the_key_discoverability_targets() -> None:
    """The metadata must expose docs, issues, security and source."""
    labels = set(_project_urls())
    for required in ("Homepage", "Repository", "Documentation", "Issue Tracker"):
        assert required in labels, f"project.urls is missing {required}"


def test_keywords_cover_the_scientific_domains_the_package_actually_has() -> None:
    """Keywords must not drift away from the shipped module set."""
    keywords = set(_string_array("keywords"))
    assert keywords, "no keywords declared"
    for domain in (
        "scientific-computing",
        "linear-algebra",
        "differential-equations",
        "monte-carlo",
        "uncertainty-quantification",
        "statistics",
        "quantum-computing",
        "signal-processing",
        "machine-learning",
        "pure-python",
    ):
        assert domain in keywords, f"keywords missing domain term {domain!r}"

    # Each shipped domain subpackage should be findable by at least one keyword
    # or a close morphological variant of it.
    shipped = {
        path.name
        for path in (ROOT / "src" / IMPORT_NAME).iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    assert len(shipped) >= 30, f"expected a broad module surface, found {len(shipped)}"

    # Compare on alphanumeric-normalised tokens so that a keyword written with a
    # hyphen ("monte-carlo", "differential-equations") matches its package
    # directory ("montecarlo", "diffeq" -> "diff"+"eq" is covered by the
    # explicit synonym map below).
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    keyword_tokens = {_norm(kw) for kw in keywords}
    # Package directories whose user-facing search term differs from the
    # directory name. Each is a legitimate, verified alias.
    synonyms = {
        "bayes": "bayesian-inference",
        "cli": "pure-python",
        "core": "pure-python",
        "data_analysis": "data-analysis",
        "data_io": "data-analysis",
        "diffeq": "differential-equations",
        "fractals": "physics",
        "genetics": "chemistry",
        "graph": "graph-algorithms",
        "hypothesis": "hypothesis-generation",
        "infotheory": "information-theory",
        "interpolate": "interpolation",
        "knowledge": "provenance",
        "math_utils": "linear-algebra",
        "ml": "machine-learning",
        "modeling": "symbolic-math",
        "montecarlo": "monte-carlo",
        "nlp": "natural-language-processing",
        "numerical_integration": "numerical-integration",
        "plot": "matplotlib",
        "quantum": "quantum-simulation",
        "scientific": "physics",
        "signals": "signal-processing",
        "stats": "statistics",
        "tools": "scientific-computing",
        "uncertainty": "uncertainty-quantification",
        "validation": "model-validation",
        "wavelets": "wavelet",
        "workflow": "scientific-workflows",
    }
    # Hyphenated synonyms must be normalised the same way as the keywords.
    synonym_tokens = {module: _norm(alias) for module, alias in synonyms.items()}
    undiscoverable = sorted(
        name
        for name in shipped
        if _norm(name) not in keyword_tokens and synonym_tokens.get(name, "") not in keyword_tokens
    )
    assert not undiscoverable, (
        f"shipped modules with no discoverable keyword alias: {undiscoverable}. "
        "Add a matching keyword, or a verified entry in the `synonyms` map above."
    )


def test_classifiers_are_well_formed() -> None:
    """Classifiers must be well-formed ``A :: B`` codes with real top-levels."""
    top_levels = {
        "Development Status",
        "Environment",
        "Framework",
        "Intended Audience",
        "License",
        "Natural Language",
        "Operating System",
        "Programming Language",
        "Topic",
        "Typing",
    }
    classifiers = _string_array("classifiers")
    assert classifiers, "no classifiers declared"
    for classifier in classifiers:
        assert " :: " in classifier, f"malformed classifier: {classifier!r}"
        assert classifier.split(" :: ")[0] in top_levels, (
            f"unknown classifier top-level: {classifier!r}"
        )


def test_classifiers_are_published_trove_codes() -> None:
    """Every classifier must be a real, published PyPI trove code.

    ``trove-classifiers`` is the canonical registry PyPI itself validates
    against. An invented or misspelled code does not fail a build -- PyPI's
    upload API rejects it, or the classifier is silently dropped from the
    rendered project page, which quietly removes the package from those
    search facets.

    ``trove-classifiers`` is available in the dev/docs environment via
    ``hatchling``'s dependency, but is not a declared test dependency, so the
    check skips rather than fails where it is genuinely absent.
    """
    classifiers = pytest.importorskip(
        "trove_classifiers",
        reason="trove-classifiers is not installed in this environment",
    )
    published = set(classifiers.classifiers)
    invalid = sorted(code for code in _string_array("classifiers") if code not in published)
    assert not invalid, (
        f"classifiers not present in the published trove-classifiers registry: {invalid}. "
        "PyPI will reject or drop these."
    )


def test_classifiers_do_not_claim_an_unheld_license() -> None:
    """SPDX ``license`` plus a License classifier must not contradict."""
    license_classifiers = [c for c in _string_array("classifiers") if c.startswith("License ::")]
    if _scalar("license") == "MIT":
        # PEP 639 SPDX string is authoritative; a License classifier alongside it
        # is redundant but only acceptable if it agrees.
        for classifier in license_classifiers:
            assert classifier == "License :: OSI Approved :: MIT License", (
                f"MIT project declares a non-MIT license classifier: {classifier!r}"
            )


def test_typed_claim_matches_the_shipped_py_typed_marker() -> None:
    """``Typing :: Typed`` requires an actual PEP 561 marker in the wheel."""
    assert "Typing :: Typed" in _string_array("classifiers")
    assert (ROOT / "src" / IMPORT_NAME / "py.typed").is_file()


def test_readme_install_command_matches_the_distribution_name() -> None:
    """The install command must name the real distribution, not the old alias."""
    text = _readme_text()
    installs = set(re.findall(r"pip install\s+[\"']?([\w.\-]+)", text))
    assert installs, "README has no pip install command"
    assert DISTRIBUTION in installs
    for name in installs:
        assert name != "scs", f"README tells users to pip install 'scs': {name!r}"


def test_description_is_a_single_nonempty_sentence() -> None:
    """The PyPI summary renders in search results; keep it tight and truthful."""
    description = _scalar("description")
    assert 40 <= len(description) <= 200, (
        f"description length {len(description)} outside the 40-200 window"
    )
    assert description.endswith(".")
    assert "zero runtime dependencies" in description.lower()


def test_readme_claims_match_the_description() -> None:
    """The headline differentiator must agree across README and metadata."""
    assert "zero runtime dependencies" in _readme_text().lower()
    assert "zero runtime dependencies" in _scalar("description").lower()


# --------------------------------------------------------------------------
# mkdocs navigation
# --------------------------------------------------------------------------


def _nav_pages() -> set[str]:
    """Every docs ``.md`` path referenced by the mkdocs ``nav``."""
    text = MKDOCS.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1]
    return set(re.findall(r"([\w\-/]+\.md)", nav))


def test_mkdocs_nav_parses_and_lists_the_core_journey() -> None:
    """A visitor must be able to reach every stage of the journey."""
    pages = _nav_pages()
    for required in (
        "index.md",
        "getting-started.md",
        "tutorials/quick_start.md",
        "cookbook.md",
        "api.md",
        "benchmarks.md",
    ):
        assert required in pages, f"mkdocs nav does not expose {required}"


def test_mkdocs_nav_exposes_tutorials_cookbook_benchmarks_and_case_studies() -> None:
    """The discoverability targets must each be reachable from the nav."""
    pages = _nav_pages()
    tutorials = sorted(p for p in pages if p.startswith("tutorials/"))
    assert len(tutorials) >= 20, f"only {len(tutorials)} tutorials in the nav"
    assert any(p.startswith("tutorials/") for p in pages)
    assert "cookbook.md" in pages
    assert "benchmarks.md" in pages
    assert any("CASE_STUDY" in p for p in pages), "no case studies in the nav"
    assert "api.md" in pages


def test_every_docs_page_on_disk_is_reachable_from_the_nav() -> None:
    """No page may be orphaned: an unlinked page is invisible to readers.

    ``why-pure-python.md`` shipped on disk but was absent from the nav, so it
    was never rendered into the site navigation despite being the single best
    answer to "why not NumPy?".
    """
    on_disk = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")}
    in_nav = _nav_pages()
    orphans = sorted(on_disk - in_nav)
    assert not orphans, (
        f"docs pages exist but are missing from the mkdocs nav: {orphans}. "
        "Either add them to `nav:` or delete them."
    )


def test_every_nav_page_exists_on_disk() -> None:
    """The inverse check: a nav entry pointing at a missing file breaks the build."""
    missing = sorted(page for page in _nav_pages() if not (DOCS / page).is_file())
    assert not missing, f"mkdocs nav references non-existent pages: {missing}"


def test_docs_internal_links_resolve() -> None:
    """Relative markdown links inside docs/ must point at real files."""
    broken: list[str] = []
    pattern = re.compile(r"\[[^\]]*\]\((?!https?://|#|mailto:)([^)#\s]+)")
    for page in sorted(DOCS.rglob("*.md")):
        for target in pattern.findall(page.read_text(encoding="utf-8")):
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not broken, f"broken relative links in docs: {broken}"


def test_readme_relative_links_resolve() -> None:
    """README links to repo files (LICENSE, SECURITY, examples/) must exist."""
    pattern = re.compile(r"\[[^\]]*\]\((?!https?://|#|mailto:)([^)\s]+)\)")
    broken = [
        target
        for target in pattern.findall(_readme_text())
        if not (ROOT / target.split("#", 1)[0]).exists()
    ]
    assert not broken, f"README links to missing paths: {broken}"


def test_site_metadata_is_consistent_with_pyproject() -> None:
    """mkdocs site_name/description must not contradict the package."""
    text = MKDOCS.read_text(encoding="utf-8")
    assert re.search(r"^site_name:\s*\S", text, re.MULTILINE)
    assert "zero runtime dependencies" in text.lower()
    assert re.search(r"^site_url:\s*https://", text, re.MULTILINE)


# --------------------------------------------------------------------------
# Module counts must be derived from the source tree, not remembered
# --------------------------------------------------------------------------


def _source_domain_modules() -> dict[str, str]:
    """Map every importable ``cds`` domain module to "subpackage" or "module".

    Private internals and the version/`__main__`` plumbing are excluded: they
    are not user-facing surface. ``cds.cli`` *is* included because it is
    documented and ships as the ``cds`` console script.
    """
    package = ROOT / "src" / IMPORT_NAME
    found: dict[str, str] = {}
    for entry in package.iterdir():
        if entry.is_dir() and not entry.name.startswith((".", "_")):
            found[entry.name] = "subpackage"
        elif entry.suffix == ".py" and entry.stem not in {"__init__", "__main__", "_version"}:
            found[entry.stem] = "module"
    return found


def test_source_domain_module_count_matches_docs_headline() -> None:
    """The "N domain modules" headline must equal the real module count.

    The README and docs/index.md both lead with a module count. Both drifted
    independently before (``docs/index.md`` said 19, the README said 34,
    ``cds modules`` listed 26), so the headline is now computed from
    ``src/cds`` and required to match.
    """
    modules = _source_domain_modules()
    assert len(modules) >= 30, f"expected a broad module surface, found {len(modules)}"

    # Every module must actually be importable; a stale directory would
    # otherwise inflate the count.
    for name in modules:
        importlib.import_module(f"{IMPORT_NAME}.{name}")

    expected = len(modules)
    for page in ("README.md", "docs/index.md"):
        text = (ROOT / page).read_text(encoding="utf-8")
        match = re.search(r"(\d+)\s+domain modules", text)
        assert match is not None, f"{page} does not state a 'domain modules' count"
        assert int(match.group(1)) == expected, (
            f"{page} claims {match.group(1)} domain modules but src/cds has {expected} "
            f"({len([k for k, v in modules.items() if v == 'subpackage'])} subpackages + "
            f"{len([k for k, v in modules.items() if v == 'module'])} single-file modules)"
        )


def test_docs_module_table_lists_every_shipped_module() -> None:
    """The docs module table must cover every shipped domain module.

    ``docs/index.md`` shipped a 19-row table that silently omitted 17 modules
    -- including whole advertised capabilities such as uncertainty
    quantification, units and workflow orchestration. A reader scanning the
    table had no way to know they existed.
    """
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"`cds\.([a-z_]+)`", index))

    modules = _source_domain_modules()
    missing = sorted(set(modules) - listed)
    assert not missing, (
        f"docs/index.md module table omits shipped modules: {missing}. "
        "Add a row for each, or update the table deliberately."
    )

    # Nothing may be advertised that does not exist.
    unknown = sorted(listed - set(modules))
    assert not unknown, f"docs/index.md lists modules that are not in src/cds: {unknown}"


def test_readme_module_table_matches_the_cli_inventory() -> None:
    """The README's quoted ``cds modules`` table must match the real CLI output.

    The README embeds the CLI inventory verbatim so readers see it without
    installing. Any drift is a false claim, so the row set is compared
    against the live ``cds modules`` output.
    """
    completed = subprocess.run(
        [sys.executable, "-m", IMPORT_NAME, "modules"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert completed.returncode == 0, (
        f"`python -m cds modules` failed ({completed.returncode}): {completed.stderr}"
    )

    cli_rows = set(re.findall(r"\|\s*(cds\.[a-z_]+)\s*\|", completed.stdout))
    assert cli_rows, "could not parse any module rows from `cds modules` output"

    readme_rows = set(re.findall(r"\|\s*(cds\.[a-z_]+)\s*\|", _readme_text()))
    assert readme_rows, "README has no `cds modules` inventory table"

    assert readme_rows == cli_rows, {
        "missing_from_readme": sorted(cli_rows - readme_rows),
        "extra_in_readme": sorted(readme_rows - cli_rows),
    }

    # The prose count beside the table must match the table too.
    count_match = re.search(r"(\d+)\s+scientific modules", _readme_text())
    assert count_match is not None, "README does not state a 'scientific modules' count"
    assert int(count_match.group(1)) == len(cli_rows), (
        f"README says {count_match.group(1)} scientific modules but the table lists {len(cli_rows)}"
    )


# --------------------------------------------------------------------------
# docs/getting-started.md must actually run
# --------------------------------------------------------------------------


def _run_python_block_as_script(
    block: str, index: int, *, encoding: str = "utf-8"
) -> subprocess.CompletedProcess[str]:
    """Execute a code block as a standalone script in a temp directory.

    Running as a real script (rather than ``exec`` inside the test process) is
    the honest check: it reproduces ``__name__ == "__main__"``, which is what
    ``multiprocessing`` spawn-based platforms require. It also keeps a runaway
    snippet from taking the test session down with it.

    ``encoding`` pins the child's ``PYTHONIOENCODING``. Windows CI runners
    default to a legacy console codec, so a snippet printing a non-ASCII
    character raises ``UnicodeEncodeError`` there and nowhere else -- exactly
    the kind of platform-specific docs breakage that a Linux-only local run
    never surfaces. See ``test_docs_python_blocks_run_on_a_legacy_console_codec``.
    """
    with tempfile.TemporaryDirectory(prefix="cds-docs-block-") as temp_dir:
        script = Path(temp_dir) / f"block_{index}.py"
        script.write_text(block, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = encoding
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
            env=env,
        )


ENTRY_DOC_PAGES = ["getting-started.md", "index.md", "why-pure-python.md"]


@pytest.mark.parametrize("page", ENTRY_DOC_PAGES)
def test_docs_python_blocks_execute(page: str) -> None:
    """Every ``python`` block on the entry-point docs pages must run clean.

    ``docs/getting-started.md`` shipped a block that raised on Windows and
    macOS: it called ``estimate_pi`` at module scope, and that function uses
    ``ProcessPoolExecutor``, which re-imports ``__main__`` on spawn platforms.
    The block now guards the call and the fix is enforced here.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    blocks = _fenced_blocks(text, "python")
    for index, block in enumerate(blocks):
        result = _run_python_block_as_script(block, index)
        assert result.returncode == 0, (
            f"docs/{page} python block {index} exited {result.returncode}:\n{result.stderr[-2000:]}"
        )


@pytest.mark.parametrize("page", ENTRY_DOC_PAGES)
def test_docs_python_blocks_run_on_a_legacy_console_codec(page: str) -> None:
    """Docs blocks must not depend on a UTF-8 console to run.

    A default Windows console uses a legacy single-byte codec that cannot
    encode symbols like ``π``, ``∫`` or ``≈``. A reader who pastes a snippet
    that prints one gets ``UnicodeEncodeError`` instead of a result. This
    reproduced on the Windows CI matrix and on no other platform, so the
    blocks are run under ``cp1252`` explicitly.

    Note the guard does not rescue the snippet: the child process dies before
    stdout is produced, so this fails loudly rather than silently truncating.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    blocks = _fenced_blocks(text, "python")
    for index, block in enumerate(blocks):
        result = _run_python_block_as_script(block, index, encoding="cp1252")
        assert result.returncode == 0, (
            f"docs/{page} python block {index} fails on a legacy (cp1252) console codec, "
            "so a reader on a default Windows terminal cannot run it:\n"
            f"{result.stderr[-2000:]}\n"
            "Use ASCII labels in printed output."
        )


def test_readme_python_blocks_run_on_a_legacy_console_codec() -> None:
    """The README quickstart must also be console-codec independent."""
    readme = _readme_text()
    for index, block in enumerate(_fenced_blocks(readme, "python")):
        result = _run_python_block_as_script(block, index, encoding="cp1252")
        assert result.returncode == 0, (
            f"README python block {index} fails on a legacy (cp1252) console codec:\n"
            f"{result.stderr[-2000:]}\n"
            "Use ASCII labels in printed output."
        )
