"""MaixVision project entry point for native material detection."""

# MaixVision normally starts this file from the project root, but some
# versions execute it from a temporary upload directory.  Make package
# imports independent of that current-working-directory detail.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.material_detect import main


if __name__ == "__main__":
    main()
