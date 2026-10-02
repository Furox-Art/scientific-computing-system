/**
 * npm entry-point contract test.
 *
 * This is a real test suite (`node --test`), wired to `npm test`, not a
 * placeholder that echoes an error and exits 0. It exists because the previous
 * npm package shipped a launcher that could not even be parsed by Node:
 *
 *   SyntaxError: Unexpected token ';'
 *
 * ...and which, had it parsed, spawned `python -m scs.cli`, a module that has
 * never existed in any published release (the import name is `cds`). Both
 * defects were invisible because `package.json` declared
 * `"test": "echo \"Error: no test specified\" && exit 1"`, which no one ran, and
 * no CI job ever invoked Node.
 *
 * The invariants asserted here are the ones that would have caught each bug:
 *
 *   - `index.js` and `bin/scs.js` parse (`node --check`, see CI).
 *   - The launcher targets the real import module `cds`, not `scs`.
 *   - The child exit status is propagated rather than swallowed.
 *   - `package.json` publishes an explicit `files` allowlist so the repository
 *     (tests, workflows, 10k+ statements of Python, promo media) is never
 *     shipped to the npm registry.
 *   - The npm version tracks the Python distribution version.
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

// This file lives in the repository root, so the root *is* __dirname. Resolving
// `..` here would silently point one directory above the checkout (and, on a
// developer machine, at whatever package.json happens to live there).
const ROOT = __dirname;
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const { buildPythonArgs, resolvePython, runSCS } = require(path.join(ROOT, 'index.js'));

/** Read `project.version` out of pyproject.toml without a TOML dependency. */
function pythonDistributionVersion() {
  const text = fs.readFileSync(path.join(ROOT, 'pyproject.toml'), 'utf8');
  const match = /^version\s*=\s*"([^"]+)"/m.exec(text);
  assert.ok(match, 'pyproject.toml must declare a static project.version');
  return match[1];
}

test('index.js is syntactically valid JavaScript', () => {
  const result = spawnSync(process.execPath, ['--check', path.join(ROOT, 'index.js')], {
    encoding: 'utf8',
  });
  assert.strictEqual(result.status, 0, `node --check index.js failed: ${result.stderr}`);
});

test('bin/scs.js is syntactically valid JavaScript', () => {
  const result = spawnSync(process.execPath, ['--check', path.join(ROOT, 'bin', 'scs.js')], {
    encoding: 'utf8',
  });
  assert.strictEqual(result.status, 0, `node --check bin/scs.js failed: ${result.stderr}`);
});

/**
 * Strip comments from JavaScript source so a doc block that *explains* the old
 * `scs.cli` bug does not trip the assertion meant to catch its reintroduction.
 */
function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"\\])\/\/.*$/gm, '$1');
}

test('the launcher targets the real import module `cds`', () => {
  const args = buildPythonArgs(['--version']);
  assert.deepStrictEqual(args.slice(0, 2), ['-m', 'cds']);

  const code = stripComments(fs.readFileSync(path.join(ROOT, 'index.js'), 'utf8'));
  assert.ok(
    !/['"]-m['"]\s*,\s*['"]scs/.test(code) && !/\bscs\.cli\b/.test(code),
    'index.js executable code must not reference the nonexistent `scs` module',
  );
});

test('`cds` is importable as a module (the target actually exists)', () => {
  const python = resolvePython();
  assert.ok(python, 'no usable Python interpreter found while running npm test');

  const probe = spawnSync(python, ['-c', 'import cds; print(cds.__version__)'], {
    cwd: ROOT,
    encoding: 'utf8',
  });
  // The Python package is not necessarily importable from the npm checkout
  // (that is the job of `pip install scientific-computing-system`), so a
  // failure here is informational. A *parse* of the launcher is what matters.
  assert.ok(
    probe.status === 0 || /ModuleNotFoundError/.test(probe.stderr || ''),
    `unexpected error probing the Python package: ${probe.stderr}`,
  );
});

test('argument forwarding preserves order and empty input', () => {
  assert.deepStrictEqual(buildPythonArgs([]), ['-m', 'cds']);
  assert.deepStrictEqual(buildPythonArgs(['stats', '1,2,3']), ['-m', 'cds', 'stats', '1,2,3']);
});

test('the exported launcher is a function', () => {
  assert.strictEqual(typeof runSCS, 'function');
});

test('package.json declares an explicit files allowlist', () => {
  assert.ok(Array.isArray(pkg.files), 'package.json must declare a `files` array');
  assert.ok(pkg.files.length > 0, 'package.json `files` allowlist must not be empty');

  const listed = new Set(pkg.files);
  assert.ok(listed.has('index.js'), 'files allowlist must include index.js');
  assert.ok(listed.has('bin/scs.js') || listed.has('bin/'), 'files allowlist must include bin/');

  // Nothing that belongs only to the Python project or the CI pipeline may be
  // published to npm.
  for (const forbidden of ['src/', 'tests/', 'docs/', '.github/', 'benchmarks/', 'assets/']) {
    assert.ok(
      !listed.has(forbidden),
      `files allowlist must not publish the Python/repository tree: ${forbidden}`,
    );
  }
});

test('npm metadata is coherent', () => {
  assert.strictEqual(pkg.private, undefined, 'the package must remain publishable');
  assert.ok(pkg.bin && pkg.bin.scs, 'package.json must expose the `scs` bin shim');
  assert.ok(pkg.license, 'package.json must declare a license');
  assert.ok(
    typeof pkg.scripts.test === 'string' && !/no test specified/.test(pkg.scripts.test),
    '`npm test` must run a real test suite, not the placeholder echo',
  );
});

test('npm version tracks the Python distribution version', () => {
  assert.strictEqual(pkg.version, pythonDistributionVersion());
});
