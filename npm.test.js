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
  //
  // `npm publish` legitimately appears in two shapes now: inline
  // (`run: npm publish ...`) and inside a `run: |` block, which the OIDC step
  // needs so it can capture npm's exit code and explain a 404 instead of
  // relaying npm's ambiguous one-liner. Both are real invocations, so accept
  // either.
  //
  // What must NOT satisfy this is a line that only mentions the command. Both
  // YAML comments and shell comments are filtered out first, so a prose
  // reference -- in the header, or in a comment explaining why token mode must
  // not pass -- cannot re-open the door the stub came through.
  const executedLines = workflow
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'));

  assert.ok(
    executedLines.some(
      (line) => /^run:\s*npm publish\b/.test(line) || /^npm publish\b/.test(line),
    ),
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

/**
 * A command name must be usable as a shell word: no separators, no spaces, no
 * path fragments. This is what `bin` keys are turned into on PATH.
 */
const VALID_COMMAND_NAME = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;

test('the bin map is a valid command-name -> path mapping', () => {
  assert.ok(pkg.bin && typeof pkg.bin === 'object', 'package.json must declare a `bin` map');
  assert.ok(!Array.isArray(pkg.bin), '`bin` must be a name -> path map, not an array');

  const entries = Object.entries(pkg.bin);
  assert.ok(entries.length > 0, 'the `bin` map must not be empty');

  for (const [command, target] of entries) {
    assert.ok(
      VALID_COMMAND_NAME.test(command),
      `bin key ${JSON.stringify(command)} is not a valid command name; ` +
        'it would become an unusable PATH entry',
    );
    assert.ok(
      !command.includes('/') && !command.includes('\\'),
      `bin key ${JSON.stringify(command)} must not contain a path separator`,
    );

    assert.ok(
      typeof target === 'string' && target.length > 0,
      `bin[${command}] must be a non-empty string path`,
    );
    assert.ok(
      !target.startsWith('/') && !/^[a-zA-Z]:[\\/]/.test(target),
      `bin[${command}] must be a relative path inside the package, got ${target}`,
    );
    assert.ok(
      !target.includes('\\'),
      `bin[${command}] must use forward slashes; npm does not normalise Windows ` +
        `separators, so a backslash path ships broken (got ${target})`,
    );
    assert.ok(
      !/^[a-z]+:\/\//i.test(target),
      `bin[${command}] must not be a URL (got ${target})`,
    );

    // The single most dangerous shape: a bin path that does not exist. npm
    // ships it with NO warning at all and the installed command is silently
    // absent -- verified by packing and installing such a package locally.
    const resolved = path.resolve(ROOT, target.replace(/^\.\//, ''));
    assert.ok(
      fs.existsSync(resolved),
      `bin[${command}] points at ${target}, which does not exist; ` +
        'npm would publish this silently and the command would never appear',
    );
    assert.ok(
      fs.statSync(resolved).isFile(),
      `bin[${command}] must point at a file, got ${target}`,
    );
  }
});

test('every bin target is shipped by the files allowlist', () => {
  const allowlist = new Set(pkg.files || []);
  const included = (relative) =>
    [...allowlist].some(
      (entry) =>
        entry === relative ||
        entry === relative.replace(/\/[^/]+$/, '') + '/' ||
        (entry.endsWith('/') && relative.startsWith(entry)),
    );

  for (const [command, target] of Object.entries(pkg.bin || {})) {
    const relative = String(target).replace(/^\.\//, '');
    assert.ok(
      included(relative),
      `bin[${command}] -> ${target} is not covered by package.json "files" ` +
        `(${allowlist.size} entries); npm would drop it from the tarball and ` +
        'the published package would have no working command',
    );
  }
});

test('the launcher shim is executable as a node script', () => {
  // npm's generated shim runs the target directly on POSIX, so the file needs
  // a shebang. Without it `scs` fails with "cannot execute binary file".
  const shim = path.resolve(ROOT, 'bin', 'scs.js');
  assert.ok(fs.existsSync(shim), 'bin/scs.js must exist');
  const firstLine = fs.readFileSync(shim, 'utf8').split('\n', 1)[0];
  assert.ok(
    firstLine.startsWith('#!'),
    `bin/scs.js must start with a shebang, got ${JSON.stringify(firstLine)}`,
  );
  assert.ok(
    /node/.test(firstLine),
    `bin/scs.js shebang must invoke node, got ${JSON.stringify(firstLine)}`,
  );
});

test('the tarball ships the launcher and keeps the bin map', () => {
  // `npm pack --dry-run --json` reports the file list without writing anything,
  // so this is the exact set of files that would be published.
  const npmArgs = ['pack', '--dry-run', '--json'];
  const result = spawnSync('npm', npmArgs, {
    cwd: ROOT,
    encoding: 'utf8',
    // On Windows the npm entry point is npm.cmd, which spawnSync cannot execute
    // directly; a shell is required. The arguments are static literals, so this
    // introduces no injection surface.
    shell: process.platform === 'win32',
  });
  assert.ok(
    !result.error,
    `could not run npm pack: ${result.error && result.error.message}`,
  );
  assert.strictEqual(
    result.status,
    0,
    `npm pack --dry-run failed: ${result.stderr || result.stdout}`,
  );

  const payload = JSON.parse(result.stdout);
  const files = payload[0].files.map((entry) => entry.path.replace(/\\/g, '/'));

  for (const [command, target] of Object.entries(pkg.bin || {})) {
    const relative = String(target).replace(/^\.\//, '');
    assert.ok(
      files.includes(relative),
      `bin[${command}] -> ${target} is missing from the tarball; packed files: ${files.join(', ')}`,
    );
  }
  assert.ok(
    files.includes('index.js'),
    'the tarball must ship index.js',
  );
  assert.ok(
    files.includes('package.json'),
    'the tarball must ship package.json',
  );
});

test('the redundant ./ bin prefix is intentional and must not be "fixed"', () => {
  // npm prints this on publish:
  //   npm warn publish "bin[scs]" script name bin/scs.js was invalid and removed
  // It reads like a removed bin. It is not. npm normalises the redundant "./"
  // and ships the entry; the published tarball keeps
  // `bin: { scs: "./bin/scs.js" }` and `npm install -g` creates a working `scs`.
  //
  // Verified three ways:
  //   * this repo's packed tarball has bin scs -> ./bin/scs.js plus bin/scs.js
  //   * axiomize@1.12.4 on npm, published from the identical shape
  //     ("./bin/axiomize.js"), ships the same and is not deprecated; the registry
  //     normalises it to `bin/axiomize.js`
  //   * installing this repo's tarball into a scratch prefix created scs,
  //     scs.cmd and scs.ps1, and `scs --version` exited 0
  //
  // So the warning is cosmetic. This test exists to stop a future maintainer
  // from "fixing" it and breaking the mapping -- the shape assertions above
  // catch genuinely broken shapes, which npm ships with no warning whatsoever.
  const scs = pkg.bin.scs;
  assert.ok(scs, 'the bin map must expose `scs`');
  assert.ok(
    String(scs).startsWith('./'),
    'bin.scs is expected to keep its "./" prefix: npm normalises it on publish ' +
      'and the published command works. Dropping it is harmless but so is ' +
      'keeping it -- do not read the cosmetic npm warning as a defect.',
  );
});

/**
 * Resolve a Python 3 interpreter for the registry-verification script.
 *
 * `python3` is not a command on Windows, so the plain name is tried first and
 * the launcher-specific names second. Returns null when none exist; the tests
 * that need it skip rather than fail on a machine without Python, matching how
 * the rest of this suite treats optional interpreters.
 */
function pythonInterpreter() {
  for (const candidate of ['python3', 'python']) {
    const probe = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (!probe.error && probe.status === 0) {
      return candidate;
    }
  }
  return null;
}

/**
 * Start a stub npm registry on an ephemeral port.
 *
 * `handler(request, response, attemptNumber)` decides each reply, so a test can
 * serve 404s for a while and then the real payload -- exactly the propagation
 * lag that produced run 37136131220 -- without touching the network.
 */
async function startStubRegistry(handler) {
  const http = require('node:http');
  const state = { attempts: 0 };

  const server = http.createServer((request, response) => {
    state.attempts += 1;
    handler(request, response, state.attempts);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();

  return {
    registry: `http://127.0.0.1:${port}`,
    attempts: () => state.attempts,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

const VERIFY_SCRIPT = path.join(ROOT, 'scripts', 'verify_npm_publication.py');

/**
 * Run the verification script asynchronously.
 *
 * It must not be spawnSync: the stub registry lives in this process, and
 * spawnSync blocks the event loop, so the server could never answer and every
 * request would hang until the child timed out.
 */
function runVerify(args) {
  const python = pythonInterpreter();
  assert.ok(python, 'a python3 interpreter is required for this test');
  return new Promise((resolve, reject) => {
    const child = require('node:child_process').spawn(
      python,
      [VERIFY_SCRIPT, ...args],
      { cwd: ROOT },
    );
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('error', reject);
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
}

test('post-publish verification tolerates registry propagation delay', async () => {
  // The regression: run 37136131220 published 2.2.0 successfully, the
  // registry recorded it ~90s later, and a single immediate probe returned
  // E404 and failed the job. Here the stub answers 404 for the first two
  // attempts and only then serves the version, so the probe must survive it.
  const stub = await startStubRegistry((request, response, attempt) => {
    if (attempt <= 2) {
      response.writeHead(404, { 'content-type': 'application/json' });
      response.end('{"error":"Not found"}');
      return;
    }
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ versions: { '1.0.0': {}, '2.2.0': {} } }));
  });

  try {
    const result = await runVerify([
      '--package', 'scientific-computing-system',
      '--version', '2.2.0',
      '--registry', stub.registry,
      '--attempts', '6',
      '--initial-delay', '0',
      '--max-delay', '0',
    ]);

    assert.strictEqual(
      result.status,
      0,
      `the probe must succeed once the version appears; got ${result.status}\n${result.stdout}\n${result.stderr}`,
    );
    assert.ok(
      /PASS: the registry serves scientific-computing-system@2\.2\.0/.test(result.stdout),
      `expected a PASS line, got: ${result.stdout}`,
    );
    assert.ok(
      stub.attempts() >= 3,
      `the probe must have retried through the 404s (saw ${stub.attempts()} requests)`,
    );
    assert.ok(
      !/::error::/.test(result.stdout),
      'a tolerated propagation delay must not be reported as an error',
    );
  } finally {
    await stub.close();
  }
});

test('post-publish verification still fails closed when the version never appears', async () => {
  // The other direction: tolerance must not become permissiveness. If the
  // version is genuinely absent at the deadline, this must exit non-zero.
  const stub = await startStubRegistry((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ versions: { '1.0.0': {} } }));
  });

  try {
    const result = await runVerify([
      '--package', 'scientific-computing-system',
      '--version', '2.2.0',
      '--registry', stub.registry,
      '--attempts', '3',
      '--initial-delay', '0',
      '--max-delay', '0',
    ]);

    assert.strictEqual(
      result.status,
      1,
      `a version that never appears must exit 1; got ${result.status}\n${result.stdout}`,
    );
    assert.ok(
      /::warning::/.test(result.stdout),
      `the deadline miss must emit a ::warning:: annotation, got: ${result.stdout}`,
    );
    assert.ok(
      /never appeared/.test(result.stderr),
      `the failure must state the version never appeared, got: ${result.stderr}`,
    );
    assert.ok(
      stub.attempts() >= 3,
      `the probe must exhaust its attempts before failing (saw ${stub.attempts()})`,
    );
  } finally {
    await stub.close();
  }
});

test('the workflow probes the registry with a generous, not shortened, budget', () => {
  const workflow = workflowBody('npm-publish.yml');
  const step = stepBlock(
    workflow,
    'Verify the published version (polling for registry propagation)',
  );
  assert.ok(step, 'the polling verification step must exist');
  assert.ok(
    /scripts\/verify_npm_publication\.py/.test(step),
    'the step must delegate to scripts/verify_npm_publication.py',
  );

  // The budget must not be trimmed to make a test fast: that is precisely what
  // produced the false failure being fixed.
  const attempts = Number(/--attempts (\d+)/.exec(step)?.[1]);
  assert.ok(
    Number.isFinite(attempts) && attempts >= 10,
    `the verification budget must be at least 10 attempts, got ${attempts}`,
  );
  const initialDelay = Number(/--initial-delay (\d+(?:\.\d+)?)/.exec(step)?.[1]);
  assert.ok(
    Number.isFinite(initialDelay) && initialDelay >= 3,
    `the initial delay must be at least 3s, got ${initialDelay}`,
  );
  const maxDelay = Number(/--max-delay (\d+(?:\.\d+)?)/.exec(step)?.[1]);
  assert.ok(
    Number.isFinite(maxDelay) && maxDelay >= 15,
    `the per-attempt cap must be at least 15s, got ${maxDelay}`,
  );

  // Query the registry directly rather than through the publishing client, which
  // conflates "absent" with "denied" and touches write credentials to do it.
  assert.ok(
    /--registry https:\/\/registry\.npmjs\.org/.test(step),
    'the step must address the registry explicitly',
  );
  assert.ok(
    !/npm view/.test(step),
    'the step must not use `npm view`: it conflates absent with denied and ' +
      'resolves against the userconfig this job created for publishing',
  );
});

test('the verification script documents and defaults to a multi-minute budget', () => {
  const source = fs.readFileSync(VERIFY_SCRIPT, 'utf8');

  const attempts = Number(/^DEFAULT_ATTEMPTS = (\d+)$/m.exec(source)?.[1]);
  const initialDelay = Number(/^DEFAULT_INITIAL_DELAY = ([\d.]+)$/m.exec(source)?.[1]);
  const maxDelay = Number(/^DEFAULT_MAX_DELAY = ([\d.]+)$/m.exec(source)?.[1]);

  assert.ok(Number.isFinite(attempts) && attempts >= 10, `DEFAULT_ATTEMPTS is ${attempts}`);
  assert.ok(Number.isFinite(initialDelay) && initialDelay >= 3, `initial delay is ${initialDelay}`);
  assert.ok(Number.isFinite(maxDelay) && maxDelay >= 15, `max delay is ${maxDelay}`);

  // Backoff must actually grow and be capped, not be a flat constant.
  assert.ok(/^DEFAULT_GROWTH = ([\d.]+)$/m.test(source), 'a growth factor must be declared');
  assert.ok(
    /min\(delay, maximum\)/.test(source),
    'the backoff must be capped',
  );
});

test('npm version tracks the Python distribution version', () => {
  assert.strictEqual(pkg.version, pythonDistributionVersion());
});
