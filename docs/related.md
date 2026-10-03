# Related projects

This documentation site is the front door. The other public repositories are not additional numerical libraries.

| Repository | What it is | PyPI | npm |
|---|---|---|---|
| [scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system) | This package. Pure Python, no runtime dependencies. | `pip install scientific-computing-system` | `npm i -g scientific-computing-system` — Node launcher shim, needs the PyPI package too |
| [scientific-computing-system-2.0](https://github.com/Furox-Art/scientific-computing-system-2.0) | The same work on NumPy, SciPy, pandas, and matplotlib. | `pip install scientific-computing-system-2.0` | published on npm |
| [axiomize](https://github.com/Furox-Art/axiomize) | Versioned models with units and SBML, CellML, and Modelica export. MCP server: `axiomize mcp`. | `pip install axiomize` | published on npm |
| [axiomize-quantum-skills-2.0](https://github.com/Furox-Art/axiomize-quantum-skills-2.0) | axiomize and the reasoning skill shipped together. Not a separate product. | Prefer the two repositories above. | Prefer the two repositories above. |
| [quantum-reasoning-skill](https://github.com/Furox-Art/quantum-reasoning-skill) | Agent skill. Copy `SKILL.md`. No published accuracy claim. | `pip install quantum-reasoning-skill` | published on npm |
| [plan-auditor](https://github.com/Furox-Art/plan-auditor) | Fail-closed check for AI coding plans. Also a GitHub Action. | `pip install plan-auditor` | published on npm |

All six names are published on npm as of this writing. The npm channel is the
real distribution path for each; for the two `*-2.0` names check the package
page for which artifact a given version corresponds to.

!!! warning "npm and PyPI are not equally verifiable for this project"
    For `scientific-computing-system` specifically, the currently published npm
    release carries **no** build provenance attestation, while the PyPI release
    does. See [SECURITY.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/SECURITY.md#distribution-channels-and-provenance)
    for the details, and prefer PyPI for anything security-sensitive.
