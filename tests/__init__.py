import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.getLogger("controlsguard.audit").setLevel(logging.CRITICAL)
