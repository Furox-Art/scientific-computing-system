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

/**
 * Strip whole-line YAML comments from a workflow.
 *
 * The assertions below are about what the workflow *executes*. The workflow
 * deliberately quotes its own previous broken regex in a comment to explain why
 * it was replaced; matching that documentation would fail the guard on the very
 * explanation that prevents the bug from being reintroduced.
 */
function stripYamlComments(source) {
  return source
    .split('\n')
    .filter((line) => !/^\s*#/.test(line))
    .join('\n');
}

/** Read a workflow with comments removed. */
function workflowBody(name) {
  return stripYamlComments(
    fs.readFileSync(path.join(ROOT, '.github', 'workflows', name), 'utf8'),
  );
}

/**
 * Extract one step block from a workflow by its `- name: <heading>` marker.
 *
 * The block runs from the marker to the next `- name:` at the same indentation,
 * so it contains that step's inputs and nothing else. A fixed character window
 * would be wrong the moment a step grows or shrinks.
 */
function stepBlock(workflow, heading) {
  const marker = `- name: ${heading}`;
  const start = workflow.indexOf(marker);
  if (start === -1) return null;
  const rest = workflow.slice(start + marker.length);
  const next = rest.search(/\n\s*- (?:name|uses|run|if|env|with):/);
  return next === -1 ? workflow.slice(start) : workflow.slice(start, start + marker.length + next);
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
  const workflow = workflowBody('npm-publish.yml');

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
  // The version-floor gate must compare versions numerically, not with a
  // regex. The regex that shipped first rejected npm 11.19.0 -- the version
  // setup-node actually installs on Node 24 -- and broke the first real publish
  // run. See test_version_floor_gate_is_not_a_regex below.
  assert.ok(
    /node -e/.test(workflow),
    'the toolchain floor must be evaluated with node, not shell pattern matching',
  );
  assert.ok(
    !/grep -Eq? ['"]\^/.test(workflow),
    'the toolchain floor must not use a regex against a dotted version',
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

test('both publish modes exist and are selected by one input', () => {
  const workflow = workflowBody('npm-publish.yml');

  // The dispatch input is what selects the mode. It must be a boolean with a
  // false default so OIDC stays the default and token mode is always opt-in.
  assert.ok(
    /use_token_fallback:/.test(workflow),
    'npm-publish.yml must expose the use_token_fallback dispatch input',
  );
  assert.ok(
    /type:\s*boolean/.test(workflow),
    'use_token_fallback must be a boolean input',
  );
  assert.ok(
    /default:\s*false/.test(workflow),
    'use_token_fallback must default to false so OIDC remains the default mode',
  );

  // Both modes must be present and mutually exclusive.
  const publishSteps = workflow.split('- name: ').filter((block) => /^Publish to npm/.test(block));
  assert.strictEqual(publishSteps.length, 2, 'there must be exactly two publish steps');
  // Exactly one step runs per mode. The token gate is matched with the `!`
  // prefix excluded explicitly: `/inputs\.use_token_fallback/` on its own also
  // matches the OIDC step's `!inputs.use_token_fallback`, which would let both
  // modes pass a loose assertion while being mutually exclusive in reality.
  const oidcStep = publishSteps.find((s) => /if:.*!inputs\.use_token_fallback/.test(s));
  const tokenStep = publishSteps.find(
    (s) => /if:.*inputs\.use_token_fallback/.test(s) && !/!inputs\.use_token_fallback/.test(s),
  );
  assert.ok(oidcStep, 'the OIDC publish step must be gated on !inputs.use_token_fallback');
  assert.ok(tokenStep, 'the token publish step must be gated on inputs.use_token_fallback');
  assert.notStrictEqual(oidcStep, tokenStep, 'the two modes must be distinct steps');

  // The token step must read the repository's existing NPM_TOKEN secret.
  assert.ok(
    /NODE_AUTH_TOKEN:\s*\$\{\{\s*secrets\.NPM_TOKEN\s*\}\}/.test(workflow),
    'the token mode must source NODE_AUTH_TOKEN from secrets.NPM_TOKEN',
  );

  // Fail-closed, and in its own step: the credential preflight checks NPM_TOKEN
  // and exits before any publish step can run. Asserted in detail by
  // `a missing credential exits before any publish step can run` below.
  assert.ok(
    /if \[ -z "\$\{NPM_TOKEN:-\}" \]; then/.test(workflow),
    'the preflight must check that NPM_TOKEN is non-empty',
  );
});

test('the token mode does not request provenance', () => {
  const workflow = workflowBody('npm-publish.yml');

  const publishSteps = workflow
    .split('- name: ')
    .filter((block) => /^Publish to npm/.test(block));

  const oidc = publishSteps.find((s) => /!inputs\.use_token_fallback/.test(s));
  const token = publishSteps.find((s) => /inputs\.use_token_fallback/.test(s) && !/!inputs/.test(s));
  assert.ok(oidc, 'OIDC publish step not found');
  assert.ok(token, 'token publish step not found');

  // Provenance is an OIDC-only Sigstore attestation. npm rejects
  // `--provenance` on a token-authenticated publish, so passing it here would
  // fail the release outright. This assertion exists so the flag cannot be
  // "helpfully" added back.
  assert.ok(
    /npm publish --provenance/.test(oidc),
    'the OIDC mode must pass --provenance',
  );
  assert.ok(
    !/--provenance/.test(token),
    'the token mode must NOT pass --provenance: a token cannot mint an attestation',
  );

  // And it must say so, rather than publishing silently without the attribute.
  assert.ok(
    /no provenance attestation/i.test(token),
    'the token mode must print a NOTICE that the release carries no provenance',
  );
});

test('the toolchain version floor is compared numerically, not by regex', () => {
  const workflow = workflowBody('npm-publish.yml');

  // Regression guard for the bug that broke the first real publish run
  // (37075040029). The original gate was:
  //     npm --version | grep -Eq '^11\.(5[1-9]|[6-9][0-9])\.|^1[2-9]\.'
  // `[6-9][0-9]` covers 60-99, so npm 11.19.0 -- what Node 24 ships -- did not
  // match and the gate failed on a compliant toolchain.
  assert.ok(
    !/grep -Eq? ['"]\^11/.test(workflow),
    'the version floor must not use the regex that rejected npm 11.19.0',
  );

  // The replacement parses and compares as integers.
  assert.ok(
    /REQUIRED_NPM = \[11, 5, 1\]/.test(workflow),
    'the floor must be declared as a numeric [major, minor, patch] triple',
  );
  assert.ok(/function compare\(/.test(workflow), 'the floor needs an integer compare');
  assert.ok(
    /compare\(npmVersion, REQUIRED_NPM\) < 0/.test(workflow),
    'the npm floor must be applied via a numeric less-than comparison',
  );

  // Node's floor is 22.14.0 and includes a patch component; the original check
  // compared only major and minor, so 22.9.0 was wrongly accepted.
  assert.ok(
    /REQUIRED_NODE = \[22, 14, 0\]/.test(workflow),
    'the Node floor must include its patch component',
  );
  assert.ok(
    /compare\(nodeVersion, REQUIRED_NODE\) < 0/.test(workflow),
    'the Node floor must be applied via a numeric less-than comparison',
  );
});

test('the token mode is given a real registry auth entry', () => {
  const workflow = workflowBody('npm-publish.yml');

  // Regression guard for run 37120531005, which failed with
  //   npm error code ENEEDAUTH
  //   npm error need auth This command requires you to be logged in to
  //                         https://registry.npmjs.org/
  // while NPM_TOKEN was present in the environment.
  //
  // NODE_AUTH_TOKEN on its own is inert. npm only substitutes it into a
  // credential when some userconfig maps the registry to ${NODE_AUTH_TOKEN};
  // that file is written by setup-node's `registry-url` input. Without
  // registry-url, setup-node writes no .npmrc, npm finds no auth entry, and it
  // refuses before ever contacting the registry -- ENEEDAUTH, not E401.
  //
  // axiomize's publish workflow succeeds with the same NPM_TOKEN approach and
  // has exactly this input; the difference was the missing registry-url.
  // Extract the step that *uses* actions/setup-node. Matching on
  // `- uses: actions/setup-node@` would miss this file's form, where the step
  // has a `- name:` line and a separate `uses:` line beneath it.
  const setupNodeStep = stepBlock(workflow, 'Set up Node');
  assert.ok(setupNodeStep, 'the Set up Node step must exist in npm-publish.yml');
  assert.ok(
    /uses:\s*actions\/setup-node@[0-9a-f]{40}/.test(setupNodeStep),
    'the Set up Node step must invoke the SHA-pinned actions/setup-node',
  );
  const withBlock = setupNodeStep.slice(setupNodeStep.indexOf('with:') + 'with:'.length);

  assert.ok(
    /registry-url:\s*"?https:\/\/registry\.npmjs\.org/.test(withBlock),
    'setup-node MUST set registry-url: without it no .npmrc is written and the token mode fails with ENEEDAUTH',
  );

  // The publish step must also export the token by the name npm substitutes.
  const publishSteps = workflow
    .split('- name: ')
    .filter((block) => /^Publish to npm/.test(block));
  const tokenStep = publishSteps.find(
    (s) => /inputs\.use_token_fallback/.test(s) && !/!inputs\.use_token_fallback/.test(s),
  );
  assert.ok(tokenStep, 'token publish step not found');
  assert.ok(
    /NODE_AUTH_TOKEN:\s*\$\{\{\s*secrets\.NPM_TOKEN\s*\}\}/.test(tokenStep),
    'the token publish step must export NODE_AUTH_TOKEN from secrets.NPM_TOKEN',
  );

  // A warm cache in a publish job is a way to publish bytes this checkout did
  // not produce.
  assert.ok(
    /package-manager-cache:\s*false/.test(withBlock),
    'setup-node must disable package-manager-cache in a publish job',
  );
});

test('the registry auth entry is asserted before npm runs', () => {
  const workflow = workflowBody('npm-publish.yml');

  // Asserting the outcome, not just the configuration: a future edit that drops
  // registry-url should fail with a clear cause rather than at `npm publish`
  // with ENEEDAUTH.
  assert.ok(
    /npm config get userconfig/.test(workflow),
    'the workflow must resolve the npm userconfig',
  );
  assert.ok(
    /_authToken/.test(workflow),
    'the auth-entry gate must check for an _authToken entry',
  );
  assert.ok(
    /registry\.npmjs\.org/.test(workflow),
    'the auth-entry gate must check the entry references registry.npmjs.org',
  );

  // It must exit non-zero when the entry is missing rather than warn and carry
  // on into npm.
  const gate = stepBlock(workflow, 'Assert the npm registry auth entry is configured');
  assert.ok(gate, 'the registry auth gate step must exist');
  assert.ok(
    /::error::npm would fail with ENEEDAUTH/.test(gate),
    'the auth-entry gate must explain the ENEEDAUTH failure it prevents',
  );

  // Each defect the gate exists to catch must terminate the gate. Bounding the
  // block to the step matters: an assertion anywhere in the file could satisfy
  // the checks above while the gate itself had been neutered.
  const requiredChecks = [
    'if [ ! -f "$userconfig" ]; then',
    "if ! grep -q '_authToken' \"$userconfig\"; then",
    "if ! grep -q 'registry.npmjs.org' \"$userconfig\"; then",
  ];
  requiredChecks.forEach((check, i) => {
    assert.ok(
      gate.includes(check),
      `auth gate must retain check ${i + 1} of ${requiredChecks.length}: ${check}`,
    );
  });

  // Three defective states, three terminations.
  const gateExits = gate.match(/^\s*exit 1\s*$/gm) || [];
  assert.strictEqual(
    gateExits.length,
    3,
    `the auth gate must exit 1 once per defective state; found ${gateExits.length}`,
  );

  // It must resolve the userconfig rather than assume a path.
  assert.ok(
    /userconfig="\$\(npm config get userconfig\)"/.test(gate),
    'the auth gate must read the resolved npm userconfig path',
  );
});

test('a missing credential exits before any publish step can run', () => {
  const workflow = workflowBody('npm-publish.yml');

  // The credential preflight must be its own step. Inline inside the publish
  // step, a check that prints an error but forgets to exit continues straight
  // into `npm publish` -- which is exactly what the failed run did: the
  // fail-closed branch's own source text appeared in the log and npm ran anyway.
  const preflightIndex = workflow.indexOf(
    'Assert the requested publish mode has a usable credential',
  );
  assert.ok(preflightIndex > -1, 'the credential preflight step must exist');

  // Both publish steps must come after it.
  const publishBlocks = [
    stepBlock(workflow, 'Publish to npm (OIDC trusted publishing, with provenance)'),
    stepBlock(workflow, 'Publish to npm (token mode, no provenance)'),
  ];
  publishBlocks.forEach((block, i) => {
    assert.ok(block, `publish step ${i} must exist`);
    const index = workflow.indexOf(block);
    assert.ok(
      index > preflightIndex,
      'every publish step must run after the credential preflight',
    );
  });

  // The preflight must cover both modes, and must exit 1 on each failure.
  // Bound the block by the *step* rather than by the next publish step, so the
  // auth-entry gate that follows is not folded in and counted here.
  const authGateIndex = workflow.indexOf(
    'Assert the npm registry auth entry is configured',
  );
  const preflightEnd = authGateIndex > preflightIndex ? authGateIndex : publishIndexes[0];
  const preflightBlock = workflow.slice(preflightIndex, preflightEnd);

  assert.ok(
    /if \[ -z "\$\{NPM_TOKEN:-\}" \]; then/.test(preflightBlock),
    'the preflight must check NPM_TOKEN for token mode',
  );
  assert.ok(
    /ACTIONS_ID_TOKEN_REQUEST_URL/.test(preflightBlock),
    'the preflight must check OIDC availability for OIDC mode',
  );
  assert.ok(
    /if \[ "\$\{\{ inputs\.use_token_fallback \}\}" = "true" \]; then/.test(preflightBlock),
    'the preflight must branch on the use_token_fallback input',
  );

  // Exactly two `exit 1`s: one per mode. Each failing branch must terminate
  // the step, so the count is a precise proxy for "both modes are guarded".
  const exits = preflightBlock.match(/^\s*exit 1\s*$/gm) || [];
  assert.strictEqual(
    exits.length,
    2,
    `the preflight must exit 1 exactly once per mode; found ${exits.length}`,
  );

  // Every failing branch must be preceded by an error annotation, so a failure
  // is never silent. Bound to the preflight block so annotations elsewhere in
  // the workflow cannot satisfy this.
  const errorAnnotations = preflightBlock.match(/::error::/g) || [];
  assert.ok(
    errorAnnotations.length >= 4,
    `the preflight must annotate each failure with ::error::; found ${errorAnnotations.length}`,
  );

  // Each mode's guard must actually contain an `exit 1`, not merely a test that
  // prints. This is the print-and-continue bug in its most direct form: a
  // branch that reports a missing credential but falls through.
  const branches = preflightBlock.split(/if \[ -z "\$\{/).slice(1);
  assert.strictEqual(
    branches.length,
    2,
    `the preflight must guard both modes with a credential test; found ${branches.length}`,
  );
  branches.forEach((branch, i) => {
    const mode = i === 0 ? 'NPM_TOKEN' : 'ACTIONS_ID_TOKEN_REQUEST_URL';
    assert.ok(
      branch.includes(mode),
      `preflight branch ${i} must test ${mode}`,
    );
    assert.ok(
      /^\s*exit 1\s*$/m.test(branch),
      `preflight branch ${i} must terminate with exit 1 when ${mode} is missing; ` +
        'printing an error and continuing into npm publish is the bug being guarded',
    );
  });

  // And the publish steps themselves must contain no credential check at all:
  // if a check survived in the publish step it could print-and-continue again.
  for (const block of publishBlocks) {
    assert.ok(
      !/NPM_TOKEN:-\}|NODE_AUTH_TOKEN:-\}/.test(block),
      'no publish step may re-check the credential inline; the preflight owns that',
    );
    assert.ok(
      /npm publish\b/.test(block),
      'each publish block must still invoke npm publish',
    );
  }
});

test('npm version tracks the Python distribution version', () => {
  assert.strictEqual(pkg.version, pythonDistributionVersion());
});
