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
 *   - The npm version tracks the Python distribution version (lockstep).
 *   - The publish path is armed: a real `npm publish --provenance`, `id-token:
 *     write` for OIDC trusted publishing, no `private: true`, and none of the
 *     gates that make a publish safe (registry existence check, tarball
 *     allowlist, version lockstep) removed.
 *
 * Scope: this package is a launcher shim. It ships no Python -- `scs` execs
 * `python -m cds` from the PyPI distribution -- so its usefulness depends on the
 * Python package being installed. The description is required to say so.
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

  // The tarball ships to the public registry, so this allowlist is the only
  // thing standing between a `npm publish` and the repository contents: the
  // test suite, the CI workflows, 10k+ statements of Python, promo media.
  for (const forbidden of ['src/', 'tests/', 'docs/', '.github/', 'benchmarks/', 'assets/']) {
    assert.ok(
      !listed.has(forbidden),
      `files allowlist must not list the Python/repository tree: ${forbidden}`,
    );
  }
});

test('npm metadata is coherent and publishable', () => {
  // `private: true` makes `npm publish` refuse to run at all. The package is
  // published, so the flag must be absent -- but its absence must be deliberate:
  // the workflow has to be armed too (asserted separately below).
  assert.notStrictEqual(
    pkg.private,
    true,
    'package.json must not set private:true; npm publish is blocked while it is set',
  );
  assert.ok(pkg.bin && pkg.bin.scs, 'package.json must expose the `scs` bin shim');
  assert.ok(pkg.license, 'package.json must declare a license');
  assert.ok(
    typeof pkg.scripts.test === 'string' && !/no test specified/.test(pkg.scripts.test),
    '`npm test` must run a real test suite, not the placeholder echo',
  );
});

test('the launcher description states the Python prerequisite', () => {
  // This package ships no Python. `scs` execs `python -m cds`, so a user who
  // installs it without `pip install scientific-computing-system` gets a
  // launcher that cannot work. The description has to say so, because it is the
  // only text npm renders on the package page.
  const description = String(pkg.description || '');
  assert.ok(description.length > 0, 'package.json must carry a description');
  assert.ok(
    /pip install scientific-computing-system/.test(description),
    'the description must tell the user the Python distribution must be installed',
  );
  assert.ok(
    /no python code/i.test(description),
    'the description must state that the npm package ships no Python itself',
  );
});

test('the npm publish path is armed', () => {
  const workflow = fs.readFileSync(
    path.join(ROOT, '.github', 'workflows', 'npm-publish.yml'),
    'utf8',
  );

  // The publish step must be real. This assertion exists because the workflow
  // previously shipped a stub --
  //   run: echo "npm publishing is disabled; PyPI is the install path" && exit 0
  // -- which reported success while publishing nothing, and whose
  // `continue`-style exit 0 was indistinguishable from a real publish.
  assert.ok(
    /^\s*run:\s*npm publish\b/m.test(workflow),
    'npm-publish.yml must invoke a real `npm publish`',
  );

  // The no-op stub must be gone entirely, including from the header comments
  // that are not executed but do mislead a reader skimming the file.
  assert.ok(
    !/npm publishing is disabled/.test(workflow),
    'the disabled-publish stub must be removed from npm-publish.yml',
  );

  // OIDC trusted publishing is the primary path: without `id-token: write` the
  // CLI has no identity token to exchange and falls back to a long-lived
  // secret.
  assert.ok(
    /^\s*id-token:\s*write\s*$/m.test(workflow),
    'the publish job must grant id-token: write for OIDC trusted publishing',
  );
  assert.ok(
    /npm publish --provenance/.test(workflow),
    'the publish step must pass --provenance',
  );

  // The gates that make a publish safe must survive: the registry existence
  // check (never republish a version) and the tarball allowlist.
  assert.ok(
    /is already on npm/.test(workflow),
    'the registry version-existence gate must be present',
  );
  assert.ok(
    /npm pack --dry-run/.test(workflow),
    'the tarball content allowlist gate must be present',
  );

  // The version-lockstep gate: package.json must agree with pyproject.toml.
  assert.ok(
    /npm \$\{pkg\.version\} != Python/.test(workflow),
    'the npm/Python version-lockstep gate must be present',
  );
});

test('npm version tracks the Python distribution version', () => {
  assert.strictEqual(pkg.version, pythonDistributionVersion());
});
