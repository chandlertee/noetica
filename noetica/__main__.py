"""Enable ``python -m noetica``."""

from __future__ import annotations

import sys

from noetica.cli import main

if __name__ == "__main__":
    sys.exit(main())
