"""Test suite for JBG transkribering.

Run from the repository root:

    python -m unittest discover -s tests -t .

torch, transformers and tiktoken are replaced by stand-ins before anything
from src is imported, so the tests need no GPU, no model downloads and no
network. OpenAI is replaced by a fake client in tests.support.
"""
import sys
import types
from unittest.mock import MagicMock

for _name in ("torch", "transformers", "tiktoken"):
    sys.modules[_name] = MagicMock(name=_name)
sys.modules["torch"].cuda.is_available = lambda: False
