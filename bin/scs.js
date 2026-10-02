#!/usr/bin/env node
/**
 * `scs` console shim.
 *
 * Resolves the launcher relative to this file rather than the process CWD so
 * the shim works when invoked through an npx-installed symlink from any
 * directory.
 */

'use strict';

require('../index.js').runSCS(process.argv.slice(2));
