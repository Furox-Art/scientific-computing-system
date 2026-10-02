/**
 * Node entry point for the Scientific Computing System CLI.
 *
 * This module is a thin, dependency-free launcher. It locates a Python
 * interpreter and forwards every argument to the package's own CLI, which is
 * the single authoritative implementation:
 *
 *     python -m cds [args...]
 *
 * Two rules are enforced by `npm test` (npm.test.mjs) and by CI:
 *
 *   1. The interpreter invocation must target the `cds` module. The PyPI
 *      distribution is named `scientific-computing-system`, but its *import*
 *      name is `cds` (see pyproject.toml `[tool.hatch.build.targets.wheel]`
 *      -> `packages = ["src/cds"]`). An earlier revision of this file spawned
 *      `python -m scs.cli`, a module that has never existed in any release;
 *      every `npx scs ...` invocation therefore died with ModuleNotFoundError.
 *   2. The child's exit status must be propagated. Swallowing it would make
 *      `scs` report success for failing scientific runs, which is worse than
 *      not shipping a Node wrapper at all.
 */

'use strict';

const { spawn, spawnSync } = require('node:child_process');

/** Argument orders to try when resolving a Python interpreter. */
const PYTHON_CANDIDATES = process.platform === 'win32'
  ? ['python', 'python3', 'py']
  : ['python3', 'python'];

/**
 * Resolve the first interpreter that actually runs.
 *
 * `spawn` alone cannot be used for discovery: if the binary is missing, the
 * error surfaces asynchronously *after* the caller has already been handed a
 * child object. Probing with `spawnSync` keeps resolution synchronous and lets
 * the launcher produce one actionable message instead of an ENOENT event.
 *
 * @param {string[]} [extraArgs] Arguments used only to prove the interpreter works.
 * @returns {string|null} The resolved interpreter, or null when none work.
 */
function resolvePython(extraArgs) {
  const args = extraArgs || ['--version'];
  for (const candidate of PYTHON_CANDIDATES) {
    const probe = spawnSync(candidate, args, { stdio: 'ignore' });
    if (!probe.error && probe.status === 0) {
      return candidate;
    }
  }
  return null;
}

/**
 * Build the argument vector handed to the Python CLI.
 *
 * Kept exported (and pure) so the contract test can assert the module target
 * without spawning a subprocess.
 *
 * @param {string[]} [args] CLI arguments forwarded from Node.
 * @returns {string[]} The `python -m cds ...` argument vector.
 */
function buildPythonArgs(args) {
  return ['-m', 'cds'].concat(args || []);
}

/**
 * Run the Scientific Computing System CLI.
 *
 * @param {string[]} [args] CLI arguments forwarded from Node.
 * @returns {number} The child's exit code (1 when no interpreter was found).
 */
function runSCS(args) {
  const python = resolvePython();
  if (python === null) {
    process.stderr.write(
      'scientific-computing-system: no usable Python interpreter found.\n' +
      'Tried: ' + PYTHON_CANDIDATES.join(', ') + '\n' +
      'Install Python 3.10 or newer, or invoke the CLI directly with ' +
      '`python -m cds` after `pip install scientific-computing-system`.\n'
    );
    return 1;
  }

  const child = spawn(python, buildPythonArgs(args), {
    stdio: 'inherit',
    cwd: __dirname,
  });

  child.on('error', (error) => {
    process.stderr.write(
      'scientific-computing-system: failed to start `' + python + '`: ' + error.message + '\n'
    );
    process.exitCode = 1;
  });

  // Propagate the child's exit status verbatim. `127` mirrors the shell
  // convention for "command could not be executed".
  child.on('exit', (code, signal) => {
    if (signal) {
      process.stderr.write('scientific-computing-system: terminated by signal ' + signal + '\n');
      process.exitCode = 1;
      return;
    }
    process.exitCode = code === null ? 127 : code;
  });

  return child.pid === undefined ? 1 : 0;
}

module.exports = { runSCS, buildPythonArgs, resolvePython, PYTHON_CANDIDATES };

if (require.main === module) {
  runSCS(process.argv.slice(2));
}
