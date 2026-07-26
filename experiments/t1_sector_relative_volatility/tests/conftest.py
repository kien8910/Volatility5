from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
