"""
Pytest configuration and session-wide fixtures.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.utils.logger import set_seed

set_seed(42)
