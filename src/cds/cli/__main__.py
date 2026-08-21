"""Allow ``python -m cds.cli`` to run the CLI directly."""

import sys

from cds.cli import main

if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
