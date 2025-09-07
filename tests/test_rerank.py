import pytest
from extras import rerank

def test_rerank_module_exists():
    assert hasattr(rerank, "__file__")
