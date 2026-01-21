# tests\conftest.py
import sys
from pathlib import Path

# Add repo root to sys.path for pytest
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    
print("sys.path contains:", sys.path)
print("Files in scripts:", list((repo_root / "scripts").glob("*")))
